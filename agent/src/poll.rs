//! Driving a task forward: one bounded Rune slice or one adapter handoff at a
//! time, the round-robin over a tree, and the worker call that hands a step to
//! a provider, a tool or a child.

use super::*;

impl<A: Authority> Agent<A> {
    /// At most one bounded Rune slice or one adapter handoff. A host can drive
    /// other tasks while this one awaits a provider, tool or child result.
    pub fn poll(&self, id: Id, adapter: &mut impl Adapter) -> Result<Progress> {
        let _guard = self
            .store
            .lock
            .lock()
            .map_err(|_| "Soma coordinator poisoned")?;
        self.poll_locked(id, adapter)
    }
    /// Body-independent tree scheduling: parent control first, one descendant
    /// slice in round-robin order, then parent adoption. The cursor is a local
    /// fairness hint; durable task state and budgets remain authoritative.
    pub fn poll_tree(&self, root: Id, adapter: &mut impl Adapter) -> Result<Progress> {
        let _guard = self
            .store
            .lock
            .lock()
            .map_err(|_| "Soma coordinator poisoned")?;
        let progress = self.poll_locked(root, adapter)?;
        if matches!(
            progress,
            Progress::Completed(_) | Progress::Cancelled | Progress::Failed(_)
        ) {
            return Ok(progress);
        }
        let (_, state) = self.store.load()?;
        let mut seen = std::collections::BTreeSet::from([root]);
        let mut children = state
            .tasks
            .get(&hex(&root))
            .ok_or("missing root task")?
            .children
            .clone();
        let mut index = 0;
        while index < children.len() {
            if children.len() > MAX_TASKS || !seen.insert(children[index]) {
                return Err("task tree bound/cycle".into());
            }
            children.extend(
                state
                    .tasks
                    .get(&hex(&children[index]))
                    .ok_or("missing descendant")?
                    .children
                    .iter()
                    .copied(),
            );
            index += 1;
        }
        if children.is_empty() {
            return Ok(progress);
        }
        let index = self
            .tree_cursor
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % children.len();
        self.poll_locked(children[index], adapter)?;
        self.poll_locked(root, adapter)
    }
    fn poll_locked(&self, id: Id, adapter: &mut impl Adapter) -> Result<Progress> {
        let mut task = self.task(id)?;
        let view = self.engine.inspect(self.profile.subject).map_err(err)?;
        let job = view
            .state
            .invocations
            .get(&id)
            .ok_or("missing task invocation")?;
        if job.prog != task.prog {
            return Err("task program mismatch".into());
        }
        if !job.status.terminal() && !task.control.cancelled {
            let input: Input = self.store.read(task.input, "soma/input/1")?;
            if let Some(parent) = input.parent {
                if self.task(parent)?.control.cancelled {
                    self.mark_cancelled(id)?;
                    task = self.task(id)?;
                }
            }
        }
        if task.control.cancelled && !job.status.terminal() {
            let children_ready = self.cancel_children(&task)?;
            if let Some(p) = &job.pending {
                if p.stage == 1 {
                    if let Some(observed) = self.observations(id)?.into_iter().find(|o| {
                        o.attempt.operation == p.id && Some(o.attempt.attempt) == p.attempt
                    }) {
                        self.reconcile(id, &observed)?;
                        return Ok(Progress::Running);
                    }
                    return Ok(Progress::Unknown {
                        operation: p.id,
                        attempt: p.attempt.ok_or("missing attempt")?,
                    });
                }
            }
            if !children_ready {
                return Ok(Progress::Waiting);
            }
            self.engine
                .cancel(self.profile.subject, id, "Soma control cancellation".into())
                .map_err(err)?;
            return Ok(Progress::Cancelled);
        }
        match self
            .engine
            .tick_invocation(self.profile.subject, id)
            .map_err(err)?
        {
            JobProgress::Finished { status, .. } => match status {
                Status::Completed => task
                    .answer
                    .map(Progress::Completed)
                    .ok_or("engine completed without application terminal result".into()),
                Status::Cancelled => Ok(Progress::Cancelled),
                other => Ok(Progress::Failed(
                    task.failure.unwrap_or_else(|| other.name().into()),
                )),
            },
            JobProgress::Yielded { .. } => Ok(Progress::Running),
            JobProgress::Idle | JobProgress::Event { .. } => Ok(Progress::Waiting),
            JobProgress::Unknown {
                operation, attempt, ..
            } => {
                if let Some(observed) = self
                    .observations(id)?
                    .into_iter()
                    .find(|o| o.attempt.operation == operation && o.attempt.attempt == attempt)
                {
                    self.reconcile(id, &observed)?;
                    Ok(Progress::Running)
                } else {
                    Ok(Progress::Unknown { operation, attempt })
                }
            }
            JobProgress::Awaiting { tag, .. } => {
                if tag != HOST {
                    return Err("unexpected task host act".into());
                }
                let step: Step = self.store.read(task.step, "soma/step/1")?;
                let joined = if matches!(step.phase, Phase::Join { .. }) {
                    match self.join_locked(&task, &step)? {
                        Some(output) => Some(output),
                        None => return Ok(Progress::Waiting),
                    }
                } else {
                    None
                };
                let task = self.task(id)?;
                let token = self
                    .worker
                    .prepare(&self.engine, self.profile.subject, id)
                    .map_err(err)?;
                let mut callback_error = None;
                let execution = self
                    .worker
                    .execute(&self.engine, token, |dispatch| {
                        match self.call(&task, &step, dispatch, adapter, joined) {
                            Ok(outcome) => outcome,
                            Err(e) => {
                                callback_error = Some(e);
                                EffectOutcome::Unknown("Soma observation persistence failed".into())
                            }
                        }
                    })
                    .map_err(err)?;
                if let Some(e) = callback_error {
                    return Err(e);
                }
                if let Some(reconciliation) = execution.reconciliation {
                    reconciliation.map_err(err)?;
                    Ok(Progress::Running)
                } else {
                    Ok(Progress::Unknown {
                        operation: execution.dispatch.operation,
                        attempt: execution.dispatch.attempt,
                    })
                }
            }
        }
    }
    fn call(
        &self,
        task: &Task,
        step: &Step,
        d: &Dispatch,
        adapter: &mut impl Adapter,
        joined: Option<Output>,
    ) -> Result<EffectOutcome> {
        if d.invocation != task.id
            || d.prog != task.prog
            || d.neuron != self.profile.subject
            || d.network != self.profile.network
            || d.arguments != handle(task.step)?
            || step.admission != task.admission
        {
            return Err("worker task/step mismatch".into());
        }
        let context_id = if matches!(step.phase, Phase::Model) {
            task.control.context
        } else {
            step.context
        };
        let context = self.context(context_id)?;
        self.context_scope(&context)?;
        let attempt = Attempt {
            operation: d.operation,
            attempt: d.attempt,
            step: task.step,
            context: context_id,
            control: task.control.revision,
            phase: step.phase.clone(),
        };
        let (head, mut state) = self.store.load()?;
        state
            .tasks
            .get_mut(&hex(&task.id))
            .ok_or("missing task")?
            .pending = Some(attempt.clone());
        // Already inside Authority::with_current: do not recursively acquire it.
        self.store.save(head, &state, vec![])?;
        let input: Input = self.store.read(task.input, "soma/input/1")?;
        let request = Request {
            task: task.id,
            operation: d.operation,
            attempt: d.attempt,
            context,
            input: input.text,
            phase: step.phase.clone(),
            history: self.observations(task.id)?,
            steering: task.control.steering.clone(),
            memory: self.memory(&self.context(context_id)?)?,
        };
        let outcome =
            if !matches!(step.phase, Phase::Model) && task.control.revision != step.control {
                AdapterOutcome::Observed(Output::Superseded)
            } else if step.round >= request.context.rounds {
                AdapterOutcome::Observed(Output::Failed {
                    reason: "task round limit".into(),
                })
            } else if let Some(output) = joined {
                AdapterOutcome::Observed(output)
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| adapter.call(&request)))
                    .unwrap_or(AdapterOutcome::Pending)
            };
        match outcome {
            AdapterOutcome::Pending => Ok(EffectOutcome::Unknown("adapter outcome pending".into())),
            AdapterOutcome::Observed(output) => {
                let observed = self.observe_locked(task.id, attempt, output)?;
                Self::effect(&observed)
            }
        }
    }
    pub(crate) fn effect(observed: &Observed) -> Result<EffectOutcome> {
        match &observed.output {
            Output::Failed { reason } => Ok(EffectOutcome::Failed(reason.clone())),
            _ => Ok(EffectOutcome::Value(match observed.next {
                Some(next) => handle(next)?,
                None => zero(),
            })),
        }
    }
}
