//! soma-kernel — the first running piece of soma: one cognition loop.
//!
//! A question comes in as text; soma turns it over to glia (the model layer),
//! gets an answer back, and remembers the exchange as content-addressed
//! particles ready to be linked into a cybergraph. That is the whole of phase
//! 1: think locally, leave a trace in the graph. The four loops, the market,
//! the Body budget — all of that arrives later and *around* this, because
//! whatever else soma becomes, this is the part it does all day.
//!
//! Boundaries, deliberately kept:
//! - soma decides *what* to think about; glia runs the thinking. No tensor,
//!   dtype or backend name crosses into this crate's API.
//! - soma produces particles and the *text* behind them; committing links is
//!   the host's job through its shared graph and explicitly selected neuron.
//!   This computation kernel has no keys or independent persistence writer.
//!
//! The mind runs on its own thread. Model loading takes a second and
//! generation takes a few more, and the caller is typically a UI frame loop
//! that cannot wait for either. [`Soma::ask`] returns immediately;
//! [`Soma::poll`] hands back [`SomaEvent`]s as they happen.

pub mod agent;
mod concepts;
mod model;
#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

pub use concepts::concepts_of;
pub(crate) use concepts::tidy;
pub(crate) use model::{generate_messages, interactive_thread, think, wake, LoadedMind};
pub(crate) use run::tokenizer::ChatMessage;

/// A particle is the project's canonical hash of the content. Text in,
/// 32 bytes out; the same text is the same particle on every machine, which is
/// what lets independently-running minds converge on a shared graph.
pub fn particle_of(text: &str) -> [u8; 32] {
    let h = hemera::hash(text.as_bytes());
    let b = h.as_bytes();
    let mut out = [0u8; 32];
    let n = b.len().min(32);
    out[..n].copy_from_slice(&b[..n]);
    out
}

/// The anchor every exchange hangs off: `particle_of("soma")`. One well-known
/// particle, so all of a mind's Q→A pairs are reachable from a single root
/// instead of floating as disconnected islands.
pub fn soma_anchor() -> [u8; 32] {
    particle_of("soma")
}

/// What soma has to say back, in the order it says it.
#[derive(Debug, Clone)]
pub enum SomaEvent {
    /// The model is loading. Happens once, on the first question.
    Waking,
    /// The mind is running the question. Sent as generation starts.
    Thinking,
    /// Prefill is stepping the prompt: `done` of `total` tokens already
    /// consumed. The first event is `{0, n}`; the last is `{n, n}` when
    /// decode is about to start. Without this the UI is silent for the
    /// whole prompt pass, which on a phone is long enough to look dead.
    Prefill { done: usize, total: usize },
    /// A piece of the answer, as it is being written. Concatenating every
    /// delta gives the raw generation; [`SomaEvent::Answer`] carries the
    /// cleaned-up whole.
    Delta(String),
    /// The answer, with the question it answers and the concepts it touches.
    Answer {
        question: String,
        answer: String,
        /// Recurring content words of the exchange, most frequent first.
        /// These are how one exchange weaves into the rest of the graph: the
        /// same concept in two conversations is the same particle.
        concepts: Vec<String>,
        tokens: usize,
        tok_per_s: f32,
        /// Prompt tokens / prefill seconds.
        tok_in_s: f32,
        /// Generated tokens / decode seconds.
        tok_out_s: f32,
    },
    /// The mind failed. The text says how; the mind stays up for the next ask.
    Error(String),
    /// The mind will run this model from the next question on.
    ModelChanged(PathBuf),
}

/// Where the mind's weights and habits come from.
#[derive(Debug, Clone)]
pub struct SomaConfig {
    /// Path to a glia `.model` file.
    pub model: PathBuf,
    /// Longest answer soma will produce, in tokens.
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for SomaConfig {
    fn default() -> Self {
        Self {
            model: default_model_path(),
            max_tokens: 512,
            temperature: 0.7,
        }
    }
}

/// Which model the mind runs, resolved in order of intent: the `SOMA_MODEL`
/// env var (this run only), then `~/cyb/model` (the choice made in the UI,
/// device config like identity is), then the smallest model in `~/llm` that
/// answers well enough to be worth linking.
pub fn default_model_path() -> PathBuf {
    if let Ok(p) = std::env::var("SOMA_MODEL") {
        return PathBuf::from(p);
    }
    if let Ok(chosen) = std::fs::read_to_string(chosen_model_file()) {
        let chosen = chosen.trim();
        if !chosen.is_empty() {
            return PathBuf::from(chosen);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("llm/qwen3-0.6b-abl.model")
}

/// Where the UI's model choice persists — one path, one line.
pub fn chosen_model_file() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("cyb").join("model")
}

/// Who soma is, on every question. The commander is nushell; soma is talk.
const SOMA_SELF: &str = "You are soma, the local mind of this cyb. \
The commander on every world is nushell: a typed command runs there \
(`help`, `ls`, pipes). A question (`? …` or a word the shell does not \
know) comes to you. You do not execute commands. If a command would \
help, name it so they can type it.";

/// What the host can tell the mind.
enum Directive {
    Ask {
        question: String,
        /// What the graph already holds on the subject — recalled by the
        /// host, spoken into the prompt here. Empty means answer cold.
        context: Vec<String>,
        /// Extra standing instructions from the host (command catalog, …).
        system: String,
    },
    /// Put down the current weights and pick these up on the next question.
    UseModel(PathBuf),
}

/// A handle to the mind's thread. Cheap to hold, immediate to call.
pub struct Soma {
    ask_tx: Sender<Directive>,
    event_rx: Receiver<SomaEvent>,
}

impl Soma {
    /// Start the mind. The thread is spawned now; the model is loaded lazily
    /// on the first question, so a cyb that is never asked anything never
    /// pays for the weights.
    pub fn spawn(cfg: SomaConfig) -> Self {
        let (ask_tx, ask_rx) = channel::<Directive>();
        let (event_tx, event_rx) = channel::<SomaEvent>();
        std::thread::Builder::new()
            .name("soma".into())
            .spawn(move || mind_loop(cfg, ask_rx, event_tx))
            .expect("spawn soma thread");
        Self { ask_tx, event_rx }
    }

    /// Hand the mind a question. Returns immediately; the answer arrives
    /// through [`Soma::poll`]. Questions queue — a second ask while the first
    /// is generating runs after it.
    pub fn ask(&self, question: impl Into<String>) {
        self.ask_grounded(question, Vec::new());
    }

    /// Ask with recall: `context` is what the graph already holds on the
    /// subject, and the answer is asked to build on it. This is the
    /// difference between chatting *near* a cybergraph and reasoning *over*
    /// one — the mind is told what it already knows.
    pub fn ask_grounded(&self, question: impl Into<String>, context: Vec<String>) {
        self.ask_on(question, context, String::new());
    }

    /// Ask with recall and a host-supplied standing note (the shell's
    /// command names, a body-specific fact). Folded into the system turn.
    pub fn ask_on(
        &self,
        question: impl Into<String>,
        context: Vec<String>,
        system: impl Into<String>,
    ) {
        let _ = self.ask_tx.send(Directive::Ask {
            question: question.into(),
            context,
            system: system.into(),
        });
    }

    /// Switch minds. The current weights are dropped, the new ones load
    /// lazily on the next question — switching costs nothing until it is
    /// used. Queued like an ask, so a switch after a pending question takes
    /// effect after that question is answered.
    pub fn use_model(&self, path: impl Into<PathBuf>) {
        let _ = self.ask_tx.send(Directive::UseModel(path.into()));
    }

    /// The next event, if one has happened. Non-blocking: call it from a
    /// frame loop.
    pub fn poll(&self) -> Option<SomaEvent> {
        match self.event_rx.try_recv() {
            Ok(ev) => Some(ev),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(SomaEvent::Error("soma thread is gone".into())),
        }
    }
}

// ── the mind's thread ────────────────────────────────────────────────────────

fn mind_loop(mut cfg: SomaConfig, ask_rx: Receiver<Directive>, event_tx: Sender<SomaEvent>) {
    // A plain spawned thread defaults to a QoS the scheduler is free to park
    // on efficiency cores, and everything downstream — kernel launches,
    // waits, the sampler between them — inherits that. Measured on an
    // M4 Max: the same model, same backend, same weights ran at 12 tok/s
    // from a default thread and 60+ from the main one. Thinking is what the
    // user is waiting for; say so.
    interactive_thread();

    let mut mind: Option<LoadedMind> = None;

    while let Ok(directive) = ask_rx.recv() {
        let (question, context, system) = match directive {
            Directive::Ask {
                question,
                context,
                system,
            } => (question, context, system),
            Directive::UseModel(path) => {
                mind = None;
                cfg.model = path.clone();
                let _ = event_tx.send(SomaEvent::ModelChanged(path));
                continue;
            }
        };
        // Wake on first use.
        if mind.is_none() {
            let _ = event_tx.send(SomaEvent::Waking);
            match wake(&cfg) {
                Ok(m) => mind = Some(m),
                Err(e) => {
                    let _ = event_tx.send(SomaEvent::Error(e));
                    continue;
                }
            }
        }
        let m = mind.as_mut().unwrap();

        let _ = event_tx.send(SomaEvent::Thinking);
        let delta_tx = event_tx.clone();
        let prefill_tx = event_tx.clone();
        match think(
            m,
            &question,
            &context,
            &system,
            &cfg,
            move |d| {
                let _ = delta_tx.send(SomaEvent::Delta(d));
            },
            move |done, total| {
                let _ = prefill_tx.send(SomaEvent::Prefill { done, total });
            },
        ) {
            Ok(stats) => {
                let concepts = concepts_of(&question, &stats.answer);
                let _ = event_tx.send(SomaEvent::Answer {
                    question,
                    answer: stats.answer,
                    concepts,
                    tokens: stats.gen_tokens,
                    tok_per_s: stats.tok_out_s,
                    tok_in_s: stats.tok_in_s,
                    tok_out_s: stats.tok_out_s,
                });
            }
            Err(e) => {
                let _ = event_tx.send(SomaEvent::Error(e));
            }
        }
    }
}
