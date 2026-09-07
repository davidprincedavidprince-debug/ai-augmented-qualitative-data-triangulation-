# ai-augmented-qualitative-data-triangulation-
# Programmatic Dissonance — Quant/Qual Review App

A focused demo of the "Programmatic Dissonance" concept: quantitative
baselines and qualitative narratives are kept structurally separate, and a
deterministic cosine similarity score — not an AI judgement — surfaces where
they might be diverging, for a human to review and decide on.

## Four sections

1. **Indicators** — define what success looks like: a name, a plain-language
   target description (the only thing embedded on this side), a target
   value, and a unit.
2. **Quantitative data** — Excel upload (columns: `indicator_name`, `value`,
   `source`) or manual entry. No AI call anywhere in this tab, by design.
3. **Qualitative data** — beneficiary narratives, uploaded as a `.txt` (one
   entry per line) or entered manually. Each entry is embedded on arrival —
   embedding only, never summarized or judged.
4. **Divergence review** — for a selected indicator, every qualitative entry
   is scored by cosine similarity against the indicator's target
   description, sorted most-divergent first. Nothing is recorded as
   confirmed until a human clicks **Confirm dissonance**, **Dismiss as
   noise**, or **Needs more info** — the AI never resolves this on its own.

## Why the AI boundary is drawn where it is

- The quantitative path never calls the AI — numbers move via plain
  Pandas/SQLite writes, matching the decision that structured data collection
  tools (ODK, Excel, etc.) don't need an LLM to move data into a database.
- The qualitative path's only AI call is **embedding**, a deterministic
  vector representation — not generation, not summarization, not judgement.
- The similarity score itself is a plain cosine-similarity computation in
  NumPy — not an LLM call, not a judgement call. The AI never decides
  whether something is "real" dissonance; a human always makes that call in
  section 4.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your Gemini API key from https://aistudio.google.com/apikey
streamlit run app.py
```

## Deploying free on Streamlit Community Cloud

Same process as before: push to GitHub, deploy at share.streamlit.io, add
`GEMINI_API_KEY` under Settings → Secrets. The SQLite file resets on
redeploy (ephemeral filesystem) — use the **Reset all data** button in the
sidebar to start a session cleanly rather than relying on persistence.

## Known limitations (worth naming, not hiding)

- Similarity threshold colors (🔴 < 0.6, 🟡 < 0.8, 🟢 ≥ 0.8) are illustrative
  defaults, not empirically validated cutoffs — a real deployment would need
  these calibrated against actual reviewer judgements.
- A single indicator has one target description embedding; it doesn't yet
  account for an indicator meaning different things in different cultural or
  programme contexts — worth naming if asked about the evidence-pluralism
  critique of this kind of similarity-based flagging.
- Excel import matches on exact indicator name string — no fuzzy matching.
