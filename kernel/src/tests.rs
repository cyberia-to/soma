//! The kernel's own checks: particles, the trimming of deliberation, concept
//! extraction, and — behind the `model-integration` feature — the whole loop
//! against real weights on this body.

use super::*;

#[test]
fn particles_are_stable_and_distinct() {
    assert_eq!(particle_of("a question"), particle_of("a question"));
    assert_ne!(particle_of("a question"), particle_of("an answer"));
    assert_ne!(soma_anchor(), particle_of(""));
}

#[test]
fn tidy_strips_closed_think_blocks() {
    assert_eq!(tidy("<think>\nhmm\n</think>\n\nParis."), "Paris.");
    assert_eq!(tidy("Paris."), "Paris.");
    // Unclosed: the budget died mid-thought; there is no answer to keep.
    assert_eq!(tidy("prefix <think>endless deliberation"), "prefix");
}

#[test]
fn concepts_are_deterministic_and_ranked() {
    let a = concepts_of(
        "what is a cybergraph?",
        "a cybergraph links particles; particles form the cybergraph",
    );
    let b = concepts_of(
        "what is a cybergraph?",
        "a cybergraph links particles; particles form the cybergraph",
    );
    assert_eq!(a, b);
    assert_eq!(a[0], "cybergraph", "most frequent word ranks first: {a:?}");
    assert!(a.contains(&"particles".to_string()));
    assert!(!a.iter().any(|w| w == "what"), "stopwords excluded: {a:?}");
}

/// Switching models mid-flight: the swap is acknowledged, and the next
/// ask wakes the new weights. Part of the local integration profile:
/// selected without two models on disk, it fails.
#[cfg(feature = "model-integration")]
#[test]
fn switches_models_between_asks() {
    let first = SomaConfig::default().model;
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let second = PathBuf::from(&home).join("llm/qwen2.5-coder-1.5b-q4k.canonical.model");
    assert!(
        first.exists() && second.exists(),
        "the local integration profile requires both {first:?} and {second:?}"
    );
    let soma = Soma::spawn(SomaConfig {
        max_tokens: 16,
        temperature: 0.0,
        ..Default::default()
    });
    soma.use_model(&second);
    soma.ask("Reply with exactly one word: ready");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    let mut changed_to: Option<PathBuf> = None;
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "no answer in three minutes"
        );
        match soma.poll() {
            Some(SomaEvent::ModelChanged(p)) => changed_to = Some(p),
            Some(SomaEvent::Answer { answer, .. }) => {
                assert_eq!(
                    changed_to.as_ref(),
                    Some(&second),
                    "answered before acknowledging the switch"
                );
                assert!(!answer.trim().is_empty());
                break;
            }
            Some(SomaEvent::Error(e)) => panic!("soma error: {e}"),
            Some(_) => {}
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

/// Grounding is real, not decorative: a fact that exists only in the
/// context must surface in the answer. Part of the local integration
/// profile: selected without weights on disk, it fails.
#[cfg(feature = "model-integration")]
#[test]
fn grounded_answers_use_the_graph() {
    let cfg = SomaConfig::default();
    assert!(
        cfg.model.exists(),
        "the local integration profile requires weights at {}",
        cfg.model.display()
    );
    let soma = Soma::spawn(SomaConfig {
        max_tokens: 48,
        temperature: 0.0,
        ..cfg
    });
    soma.ask_grounded(
        "What color is the sky over cyberia? One word.",
        vec![
            "Q: describe cyberia\nA: In cyberia the sky is bright green at all hours.".to_string(),
        ],
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "no answer in two minutes"
        );
        match soma.poll() {
            Some(SomaEvent::Answer { answer, .. }) => {
                assert!(
                    answer.to_lowercase().contains("green"),
                    "the graph said green; the answer ignored it: {answer:?}"
                );
                break;
            }
            Some(SomaEvent::Error(e)) => panic!("soma error: {e}"),
            Some(_) => {}
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

/// The full loop against the real model: ask, get an answer, see the
/// events in order. Part of the local integration profile, which CI
/// machines do not select because they carry no weights; selecting it
/// without them fails rather than passing on nothing.
#[cfg(feature = "model-integration")]
#[test]
fn asks_the_real_model() {
    let cfg = SomaConfig::default();
    assert!(
        cfg.model.exists(),
        "the local integration profile requires weights at {}",
        cfg.model.display()
    );
    let soma = Soma::spawn(SomaConfig {
        max_tokens: 64,
        temperature: 0.0,
        ..cfg
    });
    soma.ask("Reply with exactly one word: ready");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut saw_waking = false;
    let mut deltas = String::new();
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "soma did not answer within two minutes"
        );
        match soma.poll() {
            Some(SomaEvent::Waking) => saw_waking = true,
            Some(SomaEvent::Thinking) => {}
            Some(SomaEvent::Prefill { .. }) => {}
            Some(SomaEvent::Delta(d)) => deltas.push_str(&d),
            Some(SomaEvent::ModelChanged(_)) => {}
            Some(SomaEvent::Answer { answer, tokens, .. }) => {
                assert!(saw_waking, "answered without waking first");
                assert!(tokens > 0);
                assert!(!answer.trim().is_empty(), "empty answer");
                assert!(
                    deltas.contains(answer.trim()),
                    "the streamed text must contain the final answer;\n deltas: {deltas:?}\n answer: {answer:?}"
                );
                break;
            }
            Some(SomaEvent::Error(e)) => panic!("soma error: {e}"),
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}
