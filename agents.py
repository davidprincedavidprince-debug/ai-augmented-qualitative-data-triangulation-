"""
The ONLY AI calls in this app are: (1) text embedding, used on qualitative
entries and on an indicator's target description, and (2) a narrow
entailment/contradiction check used on-demand or in batch from the UI.
Neither call decides anything on its own — the human always sees the raw
number, the raw text, and the raw verdict, and makes the actual call.
"""

import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-flash-latest")

_client = None


ENTAILMENT_PROMPT = """You are checking whether a beneficiary narrative supports, contradicts,
or is unrelated to a target statement describing what programme success looks like.

Pay specific attention to negation, restriction, and scope words — "only", "not", "few", "no",
"without", "except", "limited to" — since these can completely flip a claim's meaning even when
the narrative shares heavy topical overlap with the target (same subject matter, opposite claim).

Target: "{target}"

Narrative: "{narrative}"

Does the narrative describe the target being ACHIEVED (supports), CONTRADICTED — for example by
describing exclusion, restriction to a subgroup, or the opposite of the target despite similar
topic — or is it UNRELATED/neutral to the target?

Return strict JSON:
{{"verdict": "supports" | "contradicts" | "unrelated", "reasoning": "one sentence; quote the specific restriction or negation word if that is what drives the verdict"}}
Return ONLY the JSON object, no other text."""


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=api_key)
    return _client


def check_entailment(target_description, narrative_text):
    """A narrow, targeted LLM call — NOT a generative summarizer, and it never
    decides anything on its own. It returns one more signal (like the lexical
    score) for a human to weigh alongside the embedding similarity score."""
    client = get_client()
    prompt = ENTAILMENT_PROMPT.format(target=target_description, narrative=narrative_text)
    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


_STOPWORDS = set("""a an the is are was were be been being to of in on for with and or but not no
if then than that this these those it its as at by from into over under again further out up down
we you they he she i my your our their them us his her has have had do does did will would could
should can may might must""".split())


def _tokenize(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return set(w for w in words if w not in _STOPWORDS and len(w) > 2)


def lexical_similarity(text_a, text_b):
    """Jaccard similarity of stopword-filtered word sets. Deliberately naive —
    this is the 'just count matching keywords' baseline, included specifically
    to demonstrate where it disagrees with semantic (embedding) similarity."""
    set_a, set_b = _tokenize(text_a), _tokenize(text_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def embed_text(text):
    """Returns a plain list of floats for the given text."""
    client = get_client()
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return list(result.embeddings[0].values)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def score_qual_entries_against_indicator(indicator_target_embedding, indicator_target_description, qual_entries):
    """Returns qual_entries with 'semantic_score' (embedding cosine similarity),
    'lexical_score' (naive keyword-overlap baseline), and 'gap' (semantic minus
    lexical) added to each. Sorted by semantic_score ascending (most divergent
    first, by the embedding's own — imperfect — judgement)."""
    scored = []
    for entry in qual_entries:
        emb = json.loads(entry["embedding"])
        semantic = cosine_similarity(indicator_target_embedding, emb)
        lexical = lexical_similarity(indicator_target_description, entry["text"])
        entry_copy = dict(entry)
        entry_copy["semantic_score"] = semantic
        entry_copy["lexical_score"] = lexical
        entry_copy["gap"] = semantic - lexical
        scored.append(entry_copy)
    scored.sort(key=lambda e: e["semantic_score"])
    return scored


def combined_verdict(semantic_score, threshold, entailment_result):
    """Combines the cheap-but-unreliable cosine similarity signal with the
    slower-but-more-accurate LLM entailment check. Entailment, when available,
    OVERRIDES the raw threshold call for 'supports' and 'contradicts' — those
    are exactly the cases (agreement despite low word overlap, negation despite
    high similarity) cosine similarity is worst at. 'unrelated' verdicts, or
    entries where entailment hasn't been run yet, fall back to the semantic
    threshold, since entailment gave no strong signal either way in that case."""
    if entailment_result:
        verdict = entailment_result.get("verdict")
        if verdict == "contradicts":
            return "divergent"
        if verdict == "supports":
            return "aligned"
        # "unrelated" falls through to the semantic threshold below
    return "divergent" if semantic_score < threshold else "aligned"


def _split_sentences(text):
    """Lightweight, dependency-free sentence splitter — good enough for
    beneficiary narratives and field reports, not a full NLP tokenizer.
    Abbreviations (e.g. 'Dr.', 'approx.') can occasionally cause an extra
    split; acceptable for this use case."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def semantic_chunk_text(full_text, similarity_threshold=0.75):
    """Splits a longer document into semantically coherent chunks, instead of
    fixed-size/recursive character chunking. Method: split into sentences,
    embed each sentence, and start a new chunk whenever the cosine similarity
    between consecutive sentences drops below similarity_threshold — i.e. the
    topic appears to shift. Returns a list of chunk strings. Note: this makes
    one embedding call per sentence for boundary detection, then each
    finalized chunk is re-embedded again when stored — a deliberate tradeoff
    for consistency with how every other entry in this app is embedded as a
    whole unit, at the cost of some extra API calls on long documents."""
    sentences = _split_sentences(full_text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    sentence_embeddings = [embed_text(s) for s in sentences]

    chunks = []
    current_chunk_sentences = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(sentence_embeddings[i - 1], sentence_embeddings[i])
        if sim < similarity_threshold:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])
    chunks.append(" ".join(current_chunk_sentences))
    return chunks