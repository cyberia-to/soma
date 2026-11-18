//! Taking a result back in: validating an output against the phase that asked
//! for it, recording it durably against its attempt, and reconciling what the
//! durable task now knows with the engine that is waiting on it.

use super::*;

impl<A: Authority> Agent<A> {
    pub(crate) fn reconcile(&self, id: Id, observed: &Observed) -> Result<()> {
        match Self::effect(observed)? {
            EffectOutcome::Value(value) => self
                .engine
                .record_outcome(
                    self.profile.subject,
                    id,
                    observed.attempt.operation,
                    observed.attempt.attempt,
                    value,
                )
                .map_err(err)?,
            EffectOutcome::Failed(reason) => self
                .engine
                .record_failure(
                    self.profile.subject,
                    id,
                    observed.attempt.operation,
                    observed.attempt.attempt,
                    reason,
                )
                .map_err(err)?,
            EffectOutcome::Unknown(_) => unreachable!(),
        };
        Ok(())
    }
    /// Trusted adapter receipt ingestion, never a tool exposed to the model.
    /// It retains an already dispatched observation after cancellation/revocation.
    /// Engine reconciliation still requires current authority and never replays.
    pub fn observe(&self, id: Id, operation: Id, attempt: Id, output: Output) -> Result<()> {
        let _guard = self
            .store
            .lock
            .lock()
            .map_err(|_| "Soma coordinator poisoned")?;
        let task = self.task(id)?;
        let raw_output = self.store.content("soma/adapter-output/1", &output)?.id();
        if let Some(prior) = self
            .observations(id)?
            .into_iter()
            .find(|o| o.attempt.operation == operation)
        {
            return if prior.attempt.attempt == attempt && prior.raw_output == raw_output {
                Ok(())
            } else {
                Err("conflicting adapter receipt".into())
            };
        }
        let pending = task.pending.ok_or("no captured adapter request")?;
        if pending.operation != operation || pending.attempt != attempt {
            return Err("adapter receipt lineage mismatch".into());
        }
        self.observe_locked(id, pending, output)?;
        Ok(())
    }
    pub(crate) fn observe_locked(
        &self,
        id: Id,
        attempt: Attempt,
        output: Output,
    ) -> Result<Observed> {
        let (head, mut state) = self.store.load()?;
        let task = state.tasks.get_mut(&hex(&id)).ok_or("missing task")?;
        if task.pending.as_ref() != Some(&attempt) {
            return Err("observation request mismatch".into());
        }
        let context = self.context(attempt.context)?;
        let step: Step = self.store.read(attempt.step, "soma/step/1")?;
        let raw_output = self.store.content("soma/adapter-output/1", &output)?.id();
        let output = match self.validate_output(&attempt.phase, &context, &output) {
            Ok(()) => output,
            Err(_) => Output::Failed {
                reason: "adapter output rejected by admitted schema or limits".into(),
            },
        };
        for prior in &task.observations {
            let prior: Observed = self.store.read(*prior, "soma/observation/1")?;
            if prior.attempt.operation == attempt.operation {
                return if prior.attempt == attempt && prior.raw_output == raw_output {
                    Ok(prior)
                } else {
                    Err("conflicting adapter observation".into())
                };
            }
        }
        if task.observations.len() >= MAX_ROUNDS as usize * 2 + 1 {
            return Err("observation capacity".into());
        }
        let next_phase = match &output {
            Output::Model {
                decision: Decision::Final { answer },
            } => {
                if !task.control.cancelled && task.control.revision != attempt.control {
                    Phase::Model
                } else {
                    task.answer = Some(answer.clone());
                    Phase::Terminal
                }
            }
            Output::Model {
                decision:
                    Decision::Tool {
                        name,
                        schema,
                        arguments,
                    },
            } => Phase::Tool {
                name: name.clone(),
                schema: *schema,
                arguments: arguments.clone(),
            },
            Output::Model {
                decision: Decision::Delegate { children, join },
            } => Phase::Join {
                children: children.clone(),
                policy: join.clone(),
            },
            Output::Failed { reason } => {
                task.failure = Some(reason.clone());
                Phase::Terminal
            }
            _ => Phase::Model,
        };
        let mut content = Vec::new();
        let next = if next_phase == Phase::Terminal {
            None
        } else {
            let context = if output == Output::Superseded {
                task.control.context
            } else {
                attempt.context
            };
            let next = self.store.content(
                "soma/step/1",
                &Step {
                    admission: task.admission,
                    previous: Some(attempt.step),
                    context,
                    round: step.round.checked_add(1).ok_or("round overflow")?,
                    control: attempt.control,
                    phase: next_phase,
                },
            )?;
            let id = next.id();
            content.push(next);
            task.step = id;
            Some(id)
        };
        let observed = Observed {
            attempt,
            output,
            raw_output,
            next,
        };
        let record = self.store.content("soma/observation/1", &observed)?;
        task.observations.push(record.id());
        content.push(record);
        self.store.save(head, &state, content)?;
        Ok(observed)
    }
    fn validate_output(&self, phase: &Phase, context: &Context, output: &Output) -> Result<()> {
        match (phase, output) {
            (_, Output::Failed { reason }) => {
                if reason.len() > 4096 {
                    return Err("adapter failure limit".into());
                }
            }
            (_, Output::Superseded) => {}
            (Phase::Model, Output::Model { decision }) => match decision {
                Decision::Final { answer } => {
                    text(answer)?;
                }
                Decision::Tool {
                    name,
                    schema,
                    arguments,
                } => {
                    let tool = context
                        .tools
                        .iter()
                        .find(|t| t.name == *name && t.schema == *schema)
                        .ok_or("tool not admitted")?;
                    if arguments.len() > tool.max_arguments as usize {
                        return Err("tool argument limit".into());
                    }
                }
                Decision::Delegate { children, .. } => {
                    if children.is_empty() || children.len() > 16 {
                        return Err("delegation limit".into());
                    }
                    for child in children {
                        text(&child.input)?;
                        if child.allowance == 0
                            || child.allowance > 1_000_000
                            || child.disclosure.len() > 64
                            || child
                                .disclosure
                                .iter()
                                .any(|s| !context.disclosure.contains(s))
                        {
                            return Err("child scope/allowance".into());
                        }
                    }
                }
            },
            (Phase::Tool { name, schema, .. }, Output::Tool { value }) => {
                let tool = context
                    .tools
                    .iter()
                    .find(|t| t.name == *name && t.schema == *schema)
                    .ok_or("tool no longer admitted")?;
                if value.len() > tool.max_result as usize {
                    return Err("tool result limit".into());
                }
            }
            (Phase::Join { children, .. }, Output::Joined { children: results }) => {
                if results.len() != children.len() {
                    return Err("join result count".into());
                }
            }
            _ => return Err("output phase mismatch".into()),
        };
        Ok(())
    }
}
