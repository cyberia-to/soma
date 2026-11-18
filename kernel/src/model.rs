//! The model layer of the mind's thread: finding usable weights, waking
//! them on the fastest backend this body has, and running one generation.
//! Everything here lives on the mind's thread; nothing is shared.

use std::path::PathBuf;

use run::backend::Backend;
use run::generate::{ModelRunner, SampleConfig, SampleKind};

use crate::{tidy, ChatMessage, SomaConfig, SOMA_SELF};

/// The loaded model with everything needed to run it. Lives only on the
/// mind's thread; nothing here is shared.
pub(crate) struct LoadedMind {
    model: run::arch::decoder::LlamaModel,
    tokenizer: run::Tokenizer,
    backend: Box<dyn Backend>,
    /// The weight file `wake` actually loaded, which `resolve_model` may have
    /// taken from `~/llm` instead of the one asked for. A caller holding a
    /// pinned revision must compare this against its own path.
    pub(crate) path: PathBuf,
}

pub(crate) fn interactive_thread() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
        }
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

/// Openable and large enough to be a real .model — `exists()` is not
/// enough: an adb-pushed file on Android FUSE can stat and still refuse
/// mmap, which is how soma loads weights.
fn usable_model(path: &std::path::Path) -> bool {
    std::fs::File::open(path)
        .and_then(|f| f.metadata())
        .map(|m| m.len() > 1024 && path.extension().is_some_and(|x| x == "model"))
        .unwrap_or(false)
}

fn resolve_model(preferred: &std::path::Path) -> Result<std::path::PathBuf, String> {
    if usable_model(preferred) {
        return Ok(preferred.to_path_buf());
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let dir = std::path::PathBuf::from(home).join("llm");
    if let Ok(rd) = std::fs::read_dir(&dir) {
        let mut models: Vec<std::path::PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| usable_model(p))
            .collect();
        // Smallest first, the path breaking a size tie: read_dir order is not
        // stable between runs or machines, and model choice must be.
        models.sort_by_key(|p| (p.metadata().map(|m| m.len()).unwrap_or(u64::MAX), p.clone()));
        if let Some(p) = models.into_iter().next() {
            return Ok(p);
        }
    }
    Err("no model on this body — open models and fetch qwen3-0.6b, or put a .model in ~/llm".into())
}

pub(crate) fn wake(cfg: &SomaConfig) -> Result<LoadedMind, String> {
    let model_path = resolve_model(&cfg.model)?;
    let lm = run::LoadedModel::load(&model_path).map_err(|e| format!("model load: {e}"))?;
    let tokenizer = run::tokenizer::build_tokenizer(&lm).map_err(|e| format!("tokenizer: {e}"))?;
    let mut model = run::arch::decoder::LlamaModel::from_loaded(&lm)
        .map_err(|e| format!("model build: {e}"))?;

    // The fastest backend the body has, exactly as the CLI picks it. What
    // looked like an off-main-thread honeycrisp deadlock — and cost soma
    // weeks on the CPU backend — was two stacked bugs of ours: the weights
    // were never uploaded to the backend (the CLI calls `to_backend`, soma
    // did not), and the resulting per-op fallback spun in an infinite
    // self-recursion in glia. Both fixed; the thread was never the problem.
    // `SOMA_BACKEND=cpu|wgpu|honeycrisp` still overrides for experiments.
    let choice = std::env::var("SOMA_BACKEND").unwrap_or_else(|_| "auto".into());
    let backend: Box<dyn Backend> = match choice.as_str() {
        "cpu" => Box::new(run::backend::cpu::CpuBackend::new()),
        "wgpu" => match run::backend::wgpu::WgpuRsBackend::new() {
            Ok(b) => Box::new(b),
            Err(_) => Box::new(run::backend::cpu::CpuBackend::new()),
        },
        #[cfg(target_os = "macos")]
        "honeycrisp" => match run::backend::honeycrisp::HoneycrispBackend::new() {
            Ok(b) => Box::new(b),
            Err(_) => Box::new(run::backend::cpu::CpuBackend::new()),
        },
        _ => {
            #[cfg(target_os = "macos")]
            {
                match run::backend::honeycrisp::HoneycrispBackend::new() {
                    Ok(b) => Box::new(b) as Box<dyn Backend>,
                    Err(_) => Box::new(run::backend::cpu::CpuBackend::new()),
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                Box::new(run::backend::cpu::CpuBackend::new())
            }
        }
    };
    // The upload is what makes the fast paths eligible: without it every
    // fused op sees host tensors and takes the slow road.
    if let Err(e) = model.to_backend(backend.as_ref()) {
        return Err(format!("weight upload: {e}"));
    }
    log::info!("soma: backend {:?}", backend.kind());

    log::info!("soma: awake — {}", model_path.display());
    Ok(LoadedMind {
        model,
        tokenizer,
        backend,
        path: model_path,
    })
}

pub(crate) struct ThinkStats {
    pub(crate) answer: String,
    pub(crate) gen_tokens: usize,
    pub(crate) tok_in_s: f32,
    pub(crate) tok_out_s: f32,
}

pub(crate) fn think(
    m: &mut LoadedMind,
    question: &str,
    context: &[String],
    system: &str,
    cfg: &SomaConfig,
    on_delta: impl FnMut(String),
    on_prefill: impl FnMut(usize, usize),
) -> Result<ThinkStats, String> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut self_msg = SOMA_SELF.to_string();
    if !system.is_empty() {
        self_msg.push('\n');
        self_msg.push_str(system);
    }
    if !context.is_empty() {
        self_msg.push_str(
            "\nYour cybergraph already holds the following, from earlier \
             exchanges. Build on it; do not contradict it without saying why.",
        );
        for c in context {
            self_msg.push_str("\n- ");
            self_msg.push_str(c);
        }
    }
    messages.push(ChatMessage {
        role: "system".into(),
        content: self_msg,
    });
    messages.push(ChatMessage {
        role: "user".into(),
        content: question.into(),
    });

    generate_messages(
        m,
        &messages,
        cfg,
        on_delta,
        on_prefill,
        std::time::Instant::now() + std::time::Duration::from_secs(180),
        8192,
    )
}

pub(crate) fn generate_messages(
    m: &mut LoadedMind,
    messages: &[ChatMessage],
    cfg: &SomaConfig,
    mut on_delta: impl FnMut(String),
    mut on_prefill: impl FnMut(usize, usize),
    deadline: std::time::Instant,
    max_prompt_tokens: usize,
) -> Result<ThinkStats, String> {
    // The model's own chat template, then an empty think block appended. The
    // qwen3 family spends its whole token budget deliberating inside
    // <think> unless the block is already there and closed — this is the
    // documented way to ask it to just answer.
    let mut prompt = run::generate::build_chat_prompt(&m.tokenizer, messages);
    prompt.push_str("<think>\n\n</think>\n\n");

    let sample = SampleConfig {
        method: if cfg.temperature > 0.0 {
            SampleKind::TopP
        } else {
            SampleKind::Greedy
        },
        temperature: cfg.temperature,
        top_p: 0.95,
        top_k: 40,
    };

    // The decode loop lives here rather than in glia's `generate` because the
    // point is to see the answer *as it is written*: every new token, the
    // full sequence is re-decoded and whatever text grew is sent on. Decoding
    // from the start each step is what keeps multi-byte characters honest —
    // a token boundary is not a character boundary, and decoding tokens one
    // at a time tears UTF-8 apart exactly where it matters (any text that is
    // not English).
    m.model.reset();
    let t_prefill = std::time::Instant::now();
    let mut ids = m.tokenizer.encode(&prompt);
    if let Some(bos) = m.tokenizer.bos_token_id {
        if ids.first() != Some(&bos) {
            ids.insert(0, bos);
        }
    }
    if ids.len() > max_prompt_tokens {
        return Err("model prompt token limit".into());
    }
    let prompt_n = ids.len();
    on_prefill(0, prompt_n);
    let mut logits: Vec<f32> = Vec::new();
    let mut last_report = t_prefill;
    for (i, &t) in ids.iter().enumerate() {
        if std::time::Instant::now() >= deadline {
            return Err("model prefill deadline".into());
        }
        logits = m
            .model
            .step(t, m.backend.as_ref())
            .map_err(|e| format!("prefill: {e}"))?;
        let stepped = i + 1;
        let now = std::time::Instant::now();
        if stepped == prompt_n
            || stepped % 4 == 0
            || now.duration_since(last_report).as_millis() >= 200
        {
            on_prefill(stepped, prompt_n);
            last_report = now;
        }
    }

    let prefill_s = t_prefill.elapsed().as_secs_f32().max(1e-3);
    let prompt_tokens = prompt_n;
    let t_decode = std::time::Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut sent = String::new();
    for _ in 0..cfg.max_tokens {
        if std::time::Instant::now() >= deadline {
            return Err("model decode deadline".into());
        }
        let next = run::generate::sample(&logits, sample);
        if m.tokenizer.is_eos(next) {
            break;
        }
        generated.push(next);
        let full = m.tokenizer.decode(&generated, false);
        if full.len() > soma_agent::MAX_TEXT {
            return Err("model text result limit".into());
        }
        if full.len() > sent.len() && full.is_char_boundary(sent.len()) {
            let delta = full[sent.len()..].to_string();
            let _ = on_delta(delta);
            sent = full;
        }
        logits = m
            .model
            .step(next, m.backend.as_ref())
            .map_err(|e| format!("decode: {e}"))?;
    }

    let raw = m.tokenizer.decode(&generated, false);
    let decode_s = t_decode.elapsed().as_secs_f32().max(1e-3);
    let gen = generated.len();
    Ok(ThinkStats {
        answer: tidy(&raw),
        gen_tokens: gen,
        tok_in_s: prompt_tokens as f32 / prefill_s,
        tok_out_s: gen as f32 / decode_s,
    })
}
