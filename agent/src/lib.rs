//! Durable task strategy over the existing neuron/Rune machine and worker ABI.
//! The host supplies custody and adapters. Tasks and progs never acquire keys.
mod adapters;
mod controls;
mod observe;
mod poll;
mod reader;
mod store;
mod types;
pub use adapters::*;
pub use controls::{Change, LearningProposal};
use cybergraph::{
    application::{Database, Head},
    content::Content,
};
pub use neuron_engine::Authority;
use neuron_engine::{Action, Admission, Dispatch, JobProgress, Neuron, ProgramConfig};
use neuron_model::execution::Status;
use neuron_node::{EffectOutcome, Graph, LocalWorker, Rune};
pub use reader::TaskReader;
use store::{Prepared, State, Store, err};
pub use store::{derive, hash, hex};
pub use types::*;

pub const TASK_PROGRAM: &str = r#"
let task = event;
loop {
    let reply = host(task);
    if reply == 0 { 0 } else { rebind task = reply; continue }
}
"#;
#[derive(Clone)]
pub struct Profile {
    pub subject: Id,
    pub network: Id,
    pub policy: Id,
    pub epoch: u64,
    pub attachment: Id,
    pub binding_revision: u64,
    pub device: Id,
    pub worker: Id,
    pub boot: Id,
    pub slots: usize,
    pub total_steps: u64,
}
pub struct Agent<A: Authority> {
    pub engine: Neuron<Graph, Rune, A>,
    worker: LocalWorker,
    store: Store,
    pub profile: Profile,
    tree_cursor: std::sync::atomic::AtomicUsize,
}
fn handle(id: Id) -> Result<Vec<u8>> {
    let mut words = [0; 4];
    for (i, bytes) in id.chunks_exact(8).enumerate() {
        words[i] = u64::from_le_bytes(bytes.try_into().unwrap());
    }
    neuron_rune::value(&format!(
        "[[{} {}] [{} {}]]",
        words[0], words[1], words[2], words[3]
    ))
    .map_err(err)
}
fn zero() -> Vec<u8> {
    neuron_rune::value("0").expect("literal zero")
}
impl<A: Authority> Agent<A> {
    /// Explicit local worker placement. Opening a new body generation fences
    /// the old worker; it does not retry its unresolved effects.
    pub fn open(database: Database, authority: A, mut profile: Profile) -> Result<Self> {
        if profile.slots == 0 || profile.slots > 64 || profile.total_steps == 0 {
            return Err("invalid task capacity".into());
        }
        let engine =
            Neuron::with_authority(Graph::from_database(database.clone()), Rune, authority);
        let view = match engine.inspect(profile.subject) {
            Ok(view) => view,
            Err(neuron_engine::Error::Missing) => {
                engine
                    .activate(
                        profile.subject,
                        profile.network,
                        profile.policy,
                        profile.total_steps,
                    )
                    .map_err(err)?;
                engine.inspect(profile.subject).map_err(err)?
            }
            Err(e) => return Err(err(e)),
        };
        if view.state.network != profile.network
            || view.state.policy != profile.policy
            || view.state.epoch != profile.epoch
        {
            return Err("runtime authority profile mismatch".into());
        }
        profile.total_steps = view.state.limit;
        let worker = LocalWorker::bind(
            &engine,
            profile.subject,
            view.state.writer_generation,
            LocalWorker::descriptor(
                profile.network,
                profile.worker,
                profile.device,
                profile.boot,
                vec![HOST],
                1024,
            ),
        )
        .map_err(err)?;
        let store = Store::new(database, profile.subject, profile.network)?;
        let agent = Self {
            engine,
            worker,
            store,
            profile,
            tree_cursor: std::sync::atomic::AtomicUsize::new(0),
        };
        {
            let _guard = agent
                .store
                .lock
                .lock()
                .map_err(|_| "Soma coordinator poisoned")?;
            let (mut head, mut state) = agent.store.load()?;
            if !state.slots.is_empty() && state.slots.len() > agent.profile.slots {
                return Err("configured task pool smaller than persisted pool".into());
            }
            for slot in state.slots.len()..agent.profile.slots {
                let prog = agent
                    .engine
                    .install(
                        agent.profile.subject,
                        derive(
                            "soma/prog-slot/1",
                            &[&agent.profile.network, &(slot as u64).to_le_bytes()],
                        ),
                        TASK_PROGRAM.as_bytes().to_vec(),
                        zero(),
                        ProgramConfig {
                            step_limit: 1_000_000,
                            max_inflight: 1,
                            allowed_acts: vec![HOST],
                        },
                    )
                    .map_err(err)?;
                state.slots.push(prog);
                head = Some(agent.publish(head, &state, vec![], "soma/slots", Some(prog), None)?);
            }
        }
        Ok(agent)
    }
    fn publish(
        &self,
        head: Option<Head>,
        state: &State,
        content: Vec<Content>,
        kind: &str,
        prog: Option<Id>,
        task: Option<Id>,
    ) -> Result<Head> {
        let statement = self.store.content("soma/state/1", state)?.id();
        let p = &self.profile;
        let action = Action {
            neuron: p.subject,
            network: p.network,
            policy: p.policy,
            epoch: p.epoch,
            prog,
            invocation: task,
            act: Some(HOST),
            kind,
            statement,
            allowed_acts: &[HOST],
        };
        let mut content = Some(content);
        let mut result = None;
        self.engine
            .authority
            .with_current(&action, &mut || {
                result = Some(self.store.save(
                    head,
                    state,
                    content.take().ok_or(neuron_engine::Error::Conflict)?,
                ));
                Ok(())
            })
            .map_err(err)?;
        result.ok_or("Soma publication was not called")?
    }
    fn context_scope(&self, context: &Context) -> Result<()> {
        context.validate()?;
        let p = &self.profile;
        if context.subject != p.subject
            || context.network != p.network
            || context.policy != p.policy
            || context.epoch != p.epoch
            || context.attachment != p.attachment
            || context.binding_revision != p.binding_revision
        {
            return Err("context does not match captured authority".into());
        }
        Ok(())
    }
    pub fn submit(&self, input: Input) -> Result<Id> {
        let _guard = self
            .store
            .lock
            .lock()
            .map_err(|_| "Soma coordinator poisoned")?;
        self.submit_locked(input)
    }
    fn submit_locked(&self, input: Input) -> Result<Id> {
        text(&input.text)?;
        let (mut head, mut state) = self.store.load()?;
        let input_content = self.store.content("soma/input/1", &input)?;
        let input_id = input_content.id();
        let key = hex(&input.nonce);
        if let Some(prepared) = state.prepared.get(&key) {
            if prepared.input != input_id {
                return Err("admission nonce conflict".into());
            }
            if let Some(task) = prepared.task {
                return Ok(task);
            }
        }
        self.context_scope(&input.context)?;
        if input.allowance == 0 || input.allowance > 1_000_000 {
            return Err("task step allowance".into());
        }
        let recovering = state.prepared.contains_key(&key)
            && self
                .engine
                .lookup_admission(
                    self.profile.subject,
                    derive("soma/admission/1", &[&input.nonce]),
                )
                .map_err(err)?
                .is_some();
        if let Some(parent_id) = input.parent.filter(|_| !recovering) {
            let parent = state
                .tasks
                .get(&hex(&parent_id))
                .ok_or("missing Soma parent")?;
            if parent.control.cancelled {
                return Err("parent cancelled".into());
            }
            let context = self.context(parent.control.context)?;
            if input.context.workspace != context.workspace
                || input.context.workspace_revision != context.workspace_revision
                || input.context.soul != context.soul
                || input.context.model != context.model
                || input.context.rounds > context.rounds
                || input
                    .context
                    .tools
                    .iter()
                    .any(|tool| !context.tools.contains(tool))
                || input
                    .context
                    .disclosure
                    .iter()
                    .any(|scope| !context.disclosure.contains(scope))
                || input
                    .context
                    .memory
                    .iter()
                    .any(|memory| !context.memory.contains(memory))
            {
                return Err("child context is not a parent subset".into());
            }
        }
        if !state.prepared.contains_key(&key) {
            let view = self.engine.inspect(self.profile.subject).map_err(err)?;
            let prog = state
                .slots
                .iter()
                .find(|prog| {
                    !view
                        .state
                        .invocations
                        .values()
                        .any(|job| job.prog == **prog && !job.status.terminal())
                        && !state
                            .prepared
                            .values()
                            .any(|p| p.prog == **prog && p.task.is_none())
                })
                .copied()
                .ok_or("task pool occupied")?;
            let context = self.store.content("soma/context/1", &input.context)?;
            // Read memory now: missing, wrong codec or undisclosed sources cannot
            // become an ambiguous provider failure after dispatch.
            self.memory(&input.context)?;
            let step = self.store.content(
                "soma/step/1",
                &Step {
                    admission: input.nonce,
                    previous: None,
                    context: context.id(),
                    round: 0,
                    control: 0,
                    phase: Phase::Model,
                },
            )?;
            state.prepared.insert(
                key.clone(),
                Prepared {
                    input: input_id,
                    prog,
                    step: step.id(),
                    task: None,
                },
            );
            head = Some(self.publish(
                head,
                &state,
                vec![input_content, context, step],
                "soma/admit",
                Some(prog),
                None,
            )?);
        }
        let prepared = state.prepared[&key].clone();
        let step: Step = self.store.read(prepared.step, "soma/step/1")?;
        let receipt = self
            .engine
            .submit(
                self.profile.subject,
                Admission {
                    prog: prepared.prog,
                    nonce: derive("soma/admission/1", &[&input.nonce]),
                    input: handle(prepared.step)?,
                    context: Some(step.context),
                    parent: input.parent,
                    allowance: input.allowance,
                },
            )
            .map_err(err)?;
        let mut parent_cancelled = false;
        if let Some(parent_id) = input.parent {
            let parent = state
                .tasks
                .get_mut(&hex(&parent_id))
                .ok_or("missing parent receipt")?;
            parent_cancelled = parent.control.cancelled;
            if !parent.children.contains(&receipt.invocation) {
                parent.children.push(receipt.invocation);
            }
        }
        let task = Task {
            id: receipt.invocation,
            prog: prepared.prog,
            admission: input.nonce,
            input: input_id,
            step: prepared.step,
            control: Control {
                revision: 0,
                context: step.context,
                steering: vec![],
                cancelled: parent_cancelled,
            },
            pending: None,
            observations: vec![],
            children: vec![],
            answer: None,
            failure: None,
        };
        state.tasks.insert(hex(&task.id), task);
        state.prepared.get_mut(&key).unwrap().task = Some(receipt.invocation);
        self.publish(
            head,
            &state,
            vec![],
            "soma/admitted",
            Some(prepared.prog),
            Some(receipt.invocation),
        )?;
        Ok(receipt.invocation)
    }
    /// Prepared admissions are durable even if the engine or receipt write
    /// failed. Re-admission uses their exact captured context and nonce.
    pub fn recover_admissions(&self) -> Result<Vec<Id>> {
        let _guard = self
            .store
            .lock
            .lock()
            .map_err(|_| "Soma coordinator poisoned")?;
        let (_, state) = self.store.load()?;
        state
            .prepared
            .values()
            .filter(|p| p.task.is_none())
            .map(|p| self.submit_locked(self.store.read(p.input, "soma/input/1")?))
            .collect()
    }
    pub fn task(&self, id: Id) -> Result<Task> {
        self.store
            .load()?
            .1
            .tasks
            .remove(&hex(&id))
            .ok_or("missing Soma task".into())
    }
    pub fn tasks(&self) -> Result<Vec<Task>> {
        Ok(self.store.load()?.1.tasks.into_values().collect())
    }
    pub fn context(&self, id: Id) -> Result<Context> {
        self.store.read(id, "soma/context/1")
    }
    pub fn observations(&self, id: Id) -> Result<Vec<Observed>> {
        self.task(id)?
            .observations
            .iter()
            .map(|id| self.store.read(*id, "soma/observation/1"))
            .collect()
    }
    fn memory(&self, context: &Context) -> Result<Vec<(Memory, String)>> {
        let mut total = 0;
        context
            .memory
            .iter()
            .map(|source| {
                if !context.disclosure.contains(&source.scope) {
                    return Err("memory scope denied".into());
                }
                // Both the scoped observation and exact content must still exist.
                self.store
                    .graph
                    .get(&source.head)
                    .map_err(err)?
                    .ok_or("missing memory source head")?;
                let content = self
                    .store
                    .graph
                    .get(&source.content)
                    .map_err(err)?
                    .ok_or("missing memory content")?;
                total += content.bytes().len();
                if content.codec() != cybergraph::content::Codec::Blob || total > MAX_TEXT {
                    return Err("memory bounds/codec".into());
                }
                Ok((
                    source.clone(),
                    std::str::from_utf8(content.bytes())
                        .map_err(err)?
                        .to_owned(),
                ))
            })
            .collect()
    }
}
