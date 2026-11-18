//! What an exchange is *about*, and what of it is worth keeping: concept
//! extraction and the trimming of leaked deliberation. Pure text in, pure
//! text out — no model, no state, the same answer on every machine.

/// The recurring content words of an exchange, most frequent first — the
/// hooks by which it hangs onto the rest of the graph.
///
/// Deliberately dumb: lowercase words of four letters or more, minus a small
/// list of function words, ranked by how often the exchange uses them. No
/// embedding, no model — because the value is not in the extraction being
/// clever, it is in the extraction being *deterministic*: the same concept in
/// two exchanges, or two minds, hashes to the same particle, and that
/// identity is what makes separate conversations grow into one graph.
pub fn concepts_of(question: &str, answer: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        // en
        "this",
        "that",
        "with",
        "from",
        "have",
        "what",
        "which",
        "your",
        "will",
        "would",
        "could",
        "should",
        "about",
        "there",
        "their",
        "them",
        "then",
        "than",
        "these",
        "those",
        "some",
        "such",
        "also",
        "into",
        "over",
        "more",
        "most",
        "other",
        "when",
        "where",
        "here",
        "does",
        "very",
        "just",
        "like",
        "used",
        "using",
        "each",
        "between",
        "because",
        "while",
        "been",
        "being",
        "only",
        "must",
        "many",
        "much",
        "they",
        "were",
        "your",
        "yours",
        "ours",
        "it's",
        "don't",
        "can't",
        // ru
        "этот",
        "это",
        "эта",
        "как",
        "что",
        "или",
        "для",
        "если",
        "чтобы",
        "который",
        "которая",
        "может",
        "быть",
        "есть",
        "она",
        "оно",
        "они",
        "его",
        "её",
        "их",
        "нас",
        "вас",
        "при",
        "под",
        "над",
        "все",
        "всё",
        "так",
        "тоже",
        "также",
        "когда",
        "где",
        "почему",
        "потому",
    ];

    let mut counts: Vec<(String, usize)> = Vec::new();
    for word in format!("{question} {answer}")
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
    {
        if word.chars().count() < 4 || STOP.contains(&word) {
            continue;
        }
        match counts.iter_mut().find(|(w, _)| w == word) {
            Some((_, n)) => *n += 1,
            None => counts.push((word.to_string(), 1)),
        }
    }
    // Stable: frequency first, then order of first appearance (already the
    // vec order), so equal counts do not reshuffle between runs.
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    counts.into_iter().take(5).map(|(w, _)| w).collect()
}

/// Strip whatever deliberation leaked into the answer anyway, and trim.
pub(crate) fn tidy(raw: &str) -> String {
    let mut s = raw;
    // A closed think block: drop it and keep what follows.
    if let Some(open) = s.find("<think>") {
        if let Some(close) = s.find("</think>") {
            if close > open {
                return format!(
                    "{}{}",
                    &s[..open],
                    s[close + "</think>".len()..].trim_start()
                )
                .trim()
                .to_string();
            }
        }
        // An unclosed think block means the budget ran out mid-thought;
        // everything after the tag is deliberation, not an answer.
        s = &s[..open];
    }
    s.trim().to_string()
}
