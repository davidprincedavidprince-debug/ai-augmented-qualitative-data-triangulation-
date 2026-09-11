"""
Programmatic Dissonance app — four sections.

1. Indicators     — define what "success" looks like (target description + value)
2. Quantitative   — Excel upload or manual entry, no AI involved at all
3. Qualitative    — beneficiary narrative upload/entry, embedded on arrival
4. Divergence     — cosine similarity between each indicator's target
                     description and every qualitative entry, sorted most-
                     divergent first, with a required human verification
                     action before anything is recorded as confirmed.

Run locally: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd

import database as db
import agents

st.set_page_config(page_title="Programmatic Dissonance", layout="wide")
db.init_db()

st.title("Programmatic Dissonance — quant/qual review")
st.caption(
    "Quantitative data never touches the AI. Qualitative data is embedded for "
    "similarity comparison only — never summarized or judged by the AI. Every "
    "divergence score is a raw number a human reviews and decides on, never an "
    "automatic verdict."
)

with st.sidebar:
    st.header("Session")
    if st.button("Reset all data", width="stretch"):
        db.reset_db()
        st.success("Cleared.")
        st.rerun()

tabs = st.tabs(["1. Indicators", "2. Quantitative data", "3. Qualitative data", "4. Divergence review"])

# ---------------- 1. Indicators ----------------
with tabs[0]:
    st.subheader("Define an indicator")
    st.caption(
        "The target description is the only thing embedded on this side — write it "
        "as a plain sentence describing what success actually looks like, since this "
        "is what qualitative narratives get compared against."
    )
    name = st.text_input("Indicator name", placeholder="e.g. Reliable water access")
    target_description = st.text_area(
        "Target description (plain sentence)",
        placeholder="e.g. Households have reliable, affordable access to clean water without financial strain."
    )
    col1, col2 = st.columns(2)
    target_value = col1.number_input("Target value", value=100.0)
    unit = col2.text_input("Unit", value="%")

    if st.button("Save indicator") and name.strip() and target_description.strip():
        with st.spinner("Embedding target description..."):
            emb = agents.embed_text(target_description)
        db.insert_indicator(name, target_description, emb, target_value, unit)
        st.success(f"Indicator '{name}' saved.")
        st.rerun()

    st.divider()
    st.subheader("Existing indicators")
    indicators = db.list_indicators()
    for ind in indicators:
        with st.expander(f"{ind['name']} — target {ind['target_value']}{ind['unit']}"):
            st.write(ind["target_description"])

# ---------------- 2. Quantitative data (no AI) ----------------
with tabs[1]:
    st.subheader("Quantitative data — Excel upload or manual entry")
    st.caption("Nothing on this tab calls the AI. Plain structured numbers only.")

    indicators = db.list_indicators()
    if not indicators:
        st.warning("Define at least one indicator first (tab 1).")
    else:
        options = {f"{i['name']} (id {i['id']})": i["id"] for i in indicators}

        st.markdown("**Option A — Excel upload**")
        excel_file = st.file_uploader("Upload .xlsx with columns: indicator_name, value, source", type=["xlsx"])
        if excel_file is not None:
            df = pd.read_excel(excel_file)
            st.dataframe(df.head())
            if st.button("Import rows"):
                name_to_id = {i["name"]: i["id"] for i in indicators}
                imported, skipped = 0, 0
                for _, row in df.iterrows():
                    ind_id = name_to_id.get(str(row.get("indicator_name", "")).strip())
                    if ind_id is None:
                        skipped += 1
                        continue
                    db.insert_quant_entry(
                        ind_id, float(row.get("value", 0)),
                        str(row.get("source", "excel_upload")), "excel_upload"
                    )
                    imported += 1
                st.success(f"Imported {imported} rows, skipped {skipped} (indicator name not found).")
                st.rerun()

        st.markdown("**Option B — Manual entry**")
        sel = st.selectbox("Indicator", list(options.keys()), key="quant_manual_indicator")
        val = st.number_input("Value", value=0.0, key="quant_manual_value")
        source = st.text_input("Source note", value="manual entry", key="quant_manual_source")
        if st.button("Add entry", key="add_quant_entry"):
            db.insert_quant_entry(options[sel], val, source, "manual_entry")
            st.rerun()

        st.divider()
        st.subheader("Recorded quantitative entries")
        for ind in indicators:
            entries = db.get_quant_entries(ind["id"])
            if entries:
                st.write(f"**{ind['name']}**")
                st.dataframe(pd.DataFrame(entries)[["value", "source", "entry_method", "created_at"]])

# ---------------- 3. Qualitative data ----------------
with tabs[2]:
    st.subheader("Qualitative beneficiary data")
    st.caption("Each entry is embedded for similarity comparison only — never summarized here.")

    uploaded_txt = st.file_uploader("Upload a .txt file (one entry per line)", type=["txt"])
    if uploaded_txt is not None and st.button("Import lines from file"):
        lines = uploaded_txt.read().decode("utf-8", errors="ignore").splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        with st.spinner(f"Embedding {len(lines)} entries..."):
            for line in lines:
                emb = agents.embed_text(line)
                db.insert_qual_entry(line, emb, uploaded_txt.name)
        st.success(f"Imported {len(lines)} entries.")
        st.rerun()

    st.markdown("**Or add one entry manually**")
    manual_text = st.text_area("Beneficiary feedback / narrative")
    manual_source = st.text_input("Source note", value="manual entry", key="qual_manual_source")
    if st.button("Add entry", key="add_qual_entry") and manual_text.strip():
        with st.spinner("Embedding..."):
            emb = agents.embed_text(manual_text)
        db.insert_qual_entry(manual_text, emb, manual_source)
        st.rerun()

    st.divider()
    st.markdown("**Option C — semantic chunking of a longer document**")
    st.caption(
        "For a full field report or transcript covering multiple topics, rather than one "
        "short entry. Splits at detected topic shifts instead of fixed character counts, "
        "then stores each chunk as its own entry."
    )
    long_doc_text = st.text_area("Paste a longer document here", key="long_doc_text", height=150)
    chunk_threshold = st.slider(
        "Chunk boundary sensitivity (lower = more, smaller chunks)",
        min_value=0.5, max_value=0.95, value=0.75, step=0.01, key="chunk_threshold",
    )
    if st.button("Preview chunks") and long_doc_text.strip():
        with st.spinner("Splitting into semantic chunks..."):
            preview_chunks = agents.semantic_chunk_text(long_doc_text, chunk_threshold)
        st.session_state["chunk_preview"] = preview_chunks

    preview = st.session_state.get("chunk_preview")
    if preview:
        st.write(f"{len(preview)} chunk(s) at this threshold:")
        for i, c in enumerate(preview):
            st.markdown(f"**Chunk {i+1}:** {c}")
        long_doc_source = st.text_input("Source note for all chunks", value="semantic chunking", key="long_doc_source")
        if st.button("Import these chunks as separate entries"):
            with st.spinner(f"Embedding {len(preview)} chunks..."):
                for c in preview:
                    emb = agents.embed_text(c)
                    db.insert_qual_entry(c, emb, long_doc_source)
            st.session_state.pop("chunk_preview", None)
            st.success(f"Imported {len(preview)} chunks.")
            st.rerun()

    st.divider()
    st.subheader("Recorded qualitative entries")
    qual_entries = db.list_qual_entries()
    st.caption(f"{len(qual_entries)} entries stored.")
    for q in qual_entries:
        st.markdown(f"> {q['text']}  \n*({q['source']})*")

# ---------------- 4. Divergence review (human verification) ----------------
with tabs[3]:
    st.subheader("Divergence review — cosine similarity, human-verified")
    st.caption(
        "For the selected indicator, every qualitative entry is scored against the "
        "indicator's target description. Lowest similarity (most divergent) appears "
        "first. Nothing is confirmed until you act on it below."
    )

    indicators = db.list_indicators()
    qual_entries = db.list_qual_entries()

    if not indicators:
        st.warning("Define an indicator first (tab 1).")
    elif not qual_entries:
        st.warning("Add qualitative entries first (tab 3).")
    else:
        options = {f"{i['name']}": i for i in indicators}
        sel_name = st.selectbox("Indicator", list(options.keys()))
        indicator = options[sel_name]

        quant_entries = db.get_quant_entries(indicator["id"])
        if quant_entries:
            latest = quant_entries[0]
            st.metric(
                f"Latest quantitative value — {indicator['name']}",
                f"{latest['value']}{indicator['unit']}",
                help=f"Target: {indicator['target_value']}{indicator['unit']}"
            )
        else:
            st.info("No quantitative entries recorded yet for this indicator.")

        target_embedding = json.loads(indicator["target_description_embedding"])
        scored = agents.score_qual_entries_against_indicator(
            target_embedding, indicator["target_description"], qual_entries
        )

        # Auto-run entailment for every entry that isn't already cached — no
        # button needed. Cached results persist for the session, so this only
        # actually calls the API once per entry, not on every rerun.
        uncached = [e for e in scored if f"entail_result_{e['id']}" not in st.session_state]
        if uncached:
            progress = st.progress(0.0, text=f"Checking {len(uncached)} entries against the target...")
            for i, entry in enumerate(uncached):
                st.session_state[f"entail_result_{entry['id']}"] = agents.check_entailment(
                    indicator["target_description"], entry["text"]
                )
                progress.progress((i + 1) / len(uncached))
            progress.empty()

        st.divider()
        st.write(f"Target: *{indicator['target_description']}*")

        st.divider()
        st.subheader("Lexical vs. semantic comparison")
        st.caption(
            "Lexical score = naive keyword overlap between the entry and the target "
            "description. Semantic score = embedding cosine similarity. A large gap "
            "means the two methods disagree — that disagreement is the point of this "
            "table, not an error to fix."
        )
        compare_df = pd.DataFrame([
            {
                "entry": (e["text"][:70] + "...") if len(e["text"]) > 70 else e["text"],
                "semantic_score": round(e["semantic_score"], 3),
                "lexical_score": round(e["lexical_score"], 3),
                "gap": round(e["gap"], 3),
            }
            for e in scored
        ])
        compare_df["abs_gap"] = compare_df["gap"].abs()
        compare_df = compare_df.sort_values("abs_gap", ascending=False).drop(columns="abs_gap")
        st.dataframe(compare_df, width="stretch")

        biggest = max(scored, key=lambda e: abs(e["gap"]))
        if abs(biggest["gap"]) > 0.15:
            if biggest["gap"] > 0:
                st.warning(
                    f"Biggest mismatch: semantic score {biggest['semantic_score']:.3f} vs. "
                    f"lexical score {biggest['lexical_score']:.3f} for: \"{biggest['text'][:100]}\" — "
                    f"the embedding recognises this as related in meaning despite sharing few or no "
                    f"exact words with the target. A pure keyword-matching approach would likely have "
                    f"missed this entry entirely. Note the red/green flags below are driven by the "
                    f"LLM entailment check, not this raw score — see the entailment verdict for the "
                    f"actual reasoning."
                )
            else:
                st.warning(
                    f"Biggest mismatch: lexical score {biggest['lexical_score']:.3f} vs. "
                    f"semantic score {biggest['semantic_score']:.3f} for: \"{biggest['text'][:100]}\" — "
                    f"this entry shares vocabulary with the target but the embedding sees it as "
                    f"meaning something different. A pure keyword-matching approach would likely "
                    f"have falsely treated this as aligned."
                )

        st.divider()
        st.subheader("Divergence result (LLM entailment — primary signal)")
        st.caption(
            "Every entry above was already checked automatically. Red/green here is decided "
            "by the entailment verdict — 'contradicts' is red and 'supports' is green, "
            "regardless of the raw cosine similarity score. Cosine similarity is used only as "
            "a fallback for entries the LLM marks 'unrelated' (no strong signal either way)."
        )

        llm_divergent = 0
        llm_aligned = 0
        for entry in scored:
            cached = st.session_state.get(f"entail_result_{entry['id']}")
            verdict = agents.combined_verdict(entry["semantic_score"], 0.70, cached)
            if verdict == "divergent":
                llm_divergent += 1
            else:
                llm_aligned += 1
        c1, c2 = st.columns(2)
        c1.metric("🔴 Flagged as divergent", llm_divergent)
        c2.metric("🟢 Treated as aligned", llm_aligned)

        with st.expander("Reference only: raw cosine similarity threshold (not used for the flags above)"):
            st.caption(
                "Kept for comparison purposes — this is what a cosine-similarity-only system "
                "would have flagged, before the entailment check corrects it."
            )
            threshold = st.slider(
                "Cosine similarity threshold (reference only)",
                min_value=0.0, max_value=1.0, value=0.70, step=0.01,
                key=f"threshold_{indicator['id']}",
            )
            divergent = [e for e in scored if e["semantic_score"] < threshold]
            aligned = [e for e in scored if e["semantic_score"] >= threshold]
            c3, c4 = st.columns(2)
            c3.metric("Flagged as divergent (cosine only)", len(divergent))
            c4.metric("Treated as aligned (cosine only)", len(aligned))
            score_df = pd.DataFrame({"semantic_score": [e["semantic_score"] for e in scored]})
            st.bar_chart(score_df["semantic_score"].sort_values(ascending=True).reset_index(drop=True))

        st.divider()

        for entry in scored:
            existing_review = db.get_review(indicator["id"], entry["id"])
            score = entry["semantic_score"]
            cached_entailment = st.session_state.get(f"entail_result_{entry['id']}")
            verdict = agents.combined_verdict(score, threshold, cached_entailment)
            flag = "🔴 divergent" if verdict == "divergent" else "🟢 aligned"

            with st.expander(f"{flag}  (semantic: {score:.3f}, lexical: {entry['lexical_score']:.3f}) — {entry['text'][:60]}..."):
                st.write(entry["text"])
                st.caption(f"Cosine similarity to target: {score:.3f} (reference only)  |  Source: {entry['source']}")
                if existing_review:
                    st.info(f"Previously reviewed: {existing_review['status']} — {existing_review['human_note'] or ''}")

                if cached_entailment:
                    verdict_icon = {"supports": "✅", "contradicts": "⚠️", "unrelated": "⬜"}.get(
                        cached_entailment["verdict"], "❓"
                    )
                    st.write(f"{verdict_icon} **{cached_entailment['verdict']}** — {cached_entailment['reasoning']}")
                    if cached_entailment["verdict"] == "contradicts" and score >= threshold:
                        st.caption(
                            "Cosine similarity alone would have missed this — it scored above the "
                            "reference threshold. The entailment check caught it instead."
                        )

                if st.button("Recheck with LLM", key=f"entail_{entry['id']}"):
                    with st.spinner("Rechecking..."):
                        result = agents.check_entailment(indicator["target_description"], entry["text"])
                    st.session_state[f"entail_result_{entry['id']}"] = result
                    st.rerun()

                note = st.text_input("Reviewer note (optional)", key=f"note_{entry['id']}")
                c1, c2, c3 = st.columns(3)
                if c1.button("Confirm dissonance", key=f"confirm_{entry['id']}"):
                    db.upsert_review(indicator["id"], entry["id"], score, "confirmed_dissonance", note)
                    st.rerun()
                if c2.button("Dismiss as noise", key=f"dismiss_{entry['id']}"):
                    db.upsert_review(indicator["id"], entry["id"], score, "dismissed_as_noise", note)
                    st.rerun()
                if c3.button("Needs more info", key=f"more_{entry['id']}"):
                    db.upsert_review(indicator["id"], entry["id"], score, "needs_more_info", note)
                    st.rerun()

        st.divider()
        st.subheader("Review summary for this indicator")
        reviews = db.get_reviews_for_indicator(indicator["id"])
        if reviews:
            st.dataframe(pd.DataFrame(reviews)[["qual_entry_id", "similarity_score", "status", "human_note", "reviewed_at"]])
        else:
            st.caption("No reviews recorded yet.")