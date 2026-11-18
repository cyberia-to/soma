//! The read-only view of the same durable state: no key, no activation, no
//! worker placement and no budget — an application can look without driving.

use super::*;

/// Read-only application view: no key, activation, worker placement or budget.
pub struct TaskReader {
    store: Store,
    database: Database,
    subject: Id,
    network: Id,
}
impl TaskReader {
    pub fn status(&self, id: Id) -> Result<Status> {
        self.task(id)?;
        let view = Neuron::new(Graph::from_database(self.database.clone()), Rune)
            .inspect(self.subject)
            .map_err(err)?;
        if view.state.network != self.network {
            return Err("task runtime network mismatch".into());
        }
        view.state
            .invocations
            .get(&id)
            .map(|job| job.status)
            .ok_or("missing or archived task runtime".into())
    }
    pub fn input(&self, id: Id) -> Result<Input> {
        self.store.read(self.task(id)?.input, "soma/input/1")
    }
    pub fn new(database: Database, subject: Id, network: Id) -> Result<Self> {
        Ok(Self {
            store: Store::new(database.clone(), subject, network)?,
            database,
            subject,
            network,
        })
    }
    pub fn tasks(&self) -> Result<Vec<Task>> {
        Ok(self.store.load()?.1.tasks.into_values().collect())
    }
    pub fn task(&self, id: Id) -> Result<Task> {
        self.store
            .load()?
            .1
            .tasks
            .remove(&hex(&id))
            .ok_or("missing Soma task".into())
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
    pub fn schedules(&self) -> Result<Vec<Schedule>> {
        Ok(self.store.load()?.1.schedules.into_values().collect())
    }
    pub fn prepared(&self) -> Result<Vec<Input>> {
        self.store
            .load()?
            .1
            .prepared
            .values()
            .filter(|p| p.task.is_none())
            .map(|p| self.store.read(p.input, "soma/input/1"))
            .collect()
    }
}
