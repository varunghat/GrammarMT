"""
icl_baseline_eval.py
Run from: c:\\Users\\Varun\\master_thesis   (project root)

Grammar-aware ICL translation baseline — no lexical substitution.
For each test sentence: retrieve top-k grammar rules by sentence embedding
similarity + dictionary lookups → Gemini 2.5 Flash translate → BLEU/ChrF/ChrF++.

Phases (run automatically in sequence; each is fully resumable):
  1. BUILD   – embed sentences, retrieve rules, build Gemini batch JSONL files
  2. SUBMIT  – upload to Gemini Batch API, poll until done, download results
  3. EVAL    – parse model output, compute + print metrics

Temp / checkpoint files live in data/icl_baseline/ so a crash mid-run can be resumed
by simply re-running the script.
"""

import json
import os
import re
import time
import numpy as np
import sacrebleu
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from google import genai


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val

# ──────────────────────────── CONFIG ────────────────────────────────────────────

LANGUAGES      = ["kalamang", "mandan", "sursilvan_romansh"]
GRANULARITIES  = ["rule", "section"]
TOP_K          = 10
GEMINI_MODEL   = "gemini-2.5-flash"
TOKEN_LIMIT    = 1_800_000   # conservative; Gemini hard-cap is 2 M
POLL_INTERVAL  = 600         # seconds between batch status checks

BASE_DIR      = Path(__file__).parent.parent        # project root
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "models"
OUT_DIR       = DATA_DIR / "icl_baseline"
LOG_FILE      = OUT_DIR / "batch_log.json"

# ─────────────────────────── PROMPT ─────────────────────────────────────────────

GRANULARITY_STATEMENT = {
    "rule":    "Codified Rules: Use the relevant codified rules below to interpret grammatical forms in the sentence.",
    "section": "Source Text: Use the relevant source text paragraphs below to interpret grammatical forms in the sentence.",
}

ICL_PROMPT = """\
You are a professional linguist for {lang}.
Task: Translate the sentence into English using the provided grammatical rules.
Output must be concise. Use fragments, not full paragraphs.

---

### Constraints
- Use the provided rules to interpret morphological forms in the sentence.
- Use the dictionary entries to identify known word meanings.
- Citation: You MUST cite Rule #s used in your reasoning.

---

### INPUTS
- **{granularity_statement}**:
{rules_data}

- **Sentence**: {sentence_text}
- **Dictionary** (source words found in lexicon):
{dictionary_entries}

---

### Step-by-step reasoning (Direct & Brief)
1. **Segmentation**: [Words + clitics/morphology identified].
2. **Rule Selection**: [Rule #s] used + [1-phrase justification].
3. **Word Meanings**: [Known words from dictionary].
4. **Translation**: [English result].

---

### Final Result
```yaml
english_translation: "the_english_translation"
```
"""

# ─────────────────────────── UTILITIES ──────────────────────────────────────────

def _fix_json(s: str) -> str:
    """Strip trailing commas before } or ] (non-standard but common in hand-edited files)."""
    return re.sub(r",\s*([\]}])", r"\1", s)


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.loads(_fix_json(f.read()), strict=False)


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(_fix_json(l), strict=False) for l in f if l.strip()]


def save_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_log():
    if LOG_FILE.exists():
        return load_json(LOG_FILE)
    return {}


def save_log(log):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


# ─────────────────────────── RULE EMBEDDING ─────────────────────────────────────

def compute_rule_embeddings(rules: list, model: SentenceTransformer) -> torch.Tensor:
    """Build the same rich text representation used in rule_selection.py."""
    texts = []
    for r in rules:
        desc       = str(r.get("description") or "")
        lrl_code   = str(r.get("lrl_code") or "")
        pos_tag    = f"POS:{r.get('target_pos', '')}"
        uni_tag    = f"{r.get('unimorph_feature', '')}:{r.get('unimorph_value', '')}"
        texts.append(" ".join([desc, pos_tag, uni_tag, lrl_code]))
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def load_or_build_embeddings(lang: str, rules: list, model: SentenceTransformer) -> torch.Tensor:
    cache = OUT_DIR / lang / "rule_embeddings.pt"
    if cache.exists():
        emb = torch.load(cache, weights_only=True)
        if emb.shape[0] == len(rules):
            print(f"  [{lang}] Loaded cached rule embeddings ({emb.shape[0]} rules).")
            return emb
        print(f"  [{lang}] Cache size mismatch ({emb.shape[0]} vs {len(rules)} rules). Recomputing.")
    print(f"  [{lang}] Computing rule embeddings for {len(rules)} rules...")
    emb = compute_rule_embeddings(rules, model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache)
    print(f"  [{lang}] Saved embeddings to {cache}.")
    return emb


# ─────────────────────────── RULE RETRIEVAL ─────────────────────────────────────

def retrieve_rules(
    sentence_text: str,
    rules: list,
    rule_embeddings: torch.Tensor,
    model: SentenceTransformer,
    top_k: int,
    granularity: str,
    sections_data: list | None = None,
) -> str:
    """
    Retrieve top-k rules by cosine similarity of the source sentence
    against pre-computed rule embeddings. No POS/feature scoring.
    """
    query_emb = model.encode(sentence_text, convert_to_tensor=True, normalize_embeddings=True)
    sims      = util.cos_sim(query_emb, rule_embeddings)[0].cpu().numpy()
    top_idx   = np.argsort(sims)[::-1][:top_k]
    top_rules = [rules[i] for i in top_idx]

    if granularity == "rule":
        parts = []
        for i, r in enumerate(top_rules):
            desc = r.get("description", "")
            code = r.get("lrl_code", "FUNCTION ApplyRule(STEM, POS): RETURN STEM")
            parts.append(
                f"--- CANDIDATE RULE {i + 1} ---\n"
                f"Description: {desc.strip()}\n"
                f"Code:\n{code.strip()}\n"
            )
        return "\n".join(parts)

    if granularity == "section":
        if sections_data is None:
            return "(section data not available)"
        seen, out = set(), []
        for r in top_rules:
            for sec_id, para_id in r.get("section_paragraph_ids", []):
                sec_id, para_id = int(sec_id), int(para_id)
                for sec in sections_data:
                    if sec.get("section_id") == sec_id:
                        paras = sec.get("paragraphs", [])
                        if 0 <= para_id < len(paras):
                            text = paras[para_id].strip()
                            if text and text not in seen:
                                seen.add(text)
                                out.append(text)
                        break
            if len(out) >= top_k * 2:
                break
        return ("\n\n---\n\n".join(out[:top_k * 2])) if out else "(no relevant sections found)"

    raise ValueError(f"Unknown granularity: {granularity!r}")


# ─────────────────────────── DICTIONARY LOOKUP ──────────────────────────────────

def lookup_source_words(sentence_text: str, dictionary: dict) -> str:
    """
    Tokenize source sentence, try to match tokens and sub-morphemes
    (split on = and -) against the dictionary. Returns a formatted
    multi-line string of matches for inclusion in the prompt.
    """
    raw_tokens = sentence_text.split()
    candidates: set[str] = set()
    for tok in raw_tokens:
        candidates.add(tok)
        parts = re.split(r"[=\-]", tok)
        for p in parts:
            if p:
                candidates.add(p)
                candidates.add("=" + p)   # clitics stored with = prefix

    entries, seen = [], set()
    for key in sorted(candidates):
        if key not in dictionary or key in seen:
            continue
        seen.add(key)
        senses = dictionary[key].get("senses", [])
        translations = [s.get("translation", "") for s in senses if s.get("translation")]
        if not translations:
            continue
        sa  = dictionary[key].get("spacy_analysis", {})
        pos = sa.get("head_pos", "")
        trans_str = "; ".join(translations[:2])
        entries.append(f"  - {key}: {trans_str}" + (f" [{pos}]" if pos else ""))

    return "\n".join(entries) if entries else "  (no dictionary matches found)"


# ─────────────────────────── PHASE 1: BUILD ─────────────────────────────────────

def phase_build(model: SentenceTransformer):
    print("\n" + "=" * 60)
    print("PHASE 1: Building batch prompt files")
    print("=" * 60)

    for lang in LANGUAGES:
        print(f"\n[{lang}]")
        lang_dir = OUT_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        # ── Load data ─────────────────────────────────────────────
        rules_path = DATA_DIR / "extracted_rules" / f"{lang}_extracted_rules_final.json"
        dict_path  = DATA_DIR / "dictionary"       / f"{lang}_dictionary_with_metadata.json"
        test_path  = DATA_DIR / "gemini_finetune_data" / lang / "test_inference.jsonl"
        sects_path = DATA_DIR / "sections_split"   / f"{lang}_sections_classified_split.json"

        rules      = load_json(rules_path)
        dictionary = load_json(dict_path)
        test_lines = load_jsonl(test_path)
        sections   = load_json(sects_path) if sects_path.exists() else None
        print(f"  Rules: {len(rules)}  |  Test sentences: {len(test_lines)}")

        rule_embeddings = load_or_build_embeddings(lang, rules, model)

        # ── Build one JSONL per granularity ───────────────────────
        for gran in GRANULARITIES:
            out_file = lang_dir / f"{lang}_icl_{gran}_batch.jsonl"

            if out_file.exists():
                existing = load_jsonl(out_file)
                if len(existing) == len(test_lines):
                    print(f"  [{gran}] Already complete ({len(existing)} prompts). Skipping.")
                    continue
                print(f"  [{gran}] Partial file ({len(existing)}/{len(test_lines)}). Rebuilding.")

            print(f"  [{gran}] Building {len(test_lines)} prompts...")
            batch_requests = []

            for entry in tqdm(test_lines, desc=f"  {lang}/{gran}", leave=False):
                key         = entry["key"]
                source_text = entry["request"]["contents"][0]["parts"][0]["text"]
                # Strip "Translate to english from {lang}: " prefix
                sentence_text = source_text.split(":", 1)[1].strip() if ":" in source_text else source_text.strip()

                rules_data  = retrieve_rules(sentence_text, rules, rule_embeddings, model, TOP_K, gran, sections)
                dict_str    = lookup_source_words(sentence_text, dictionary)
                lang_label  = lang.replace("_", " ")

                prompt = ICL_PROMPT.format(
                    lang=lang_label,
                    granularity_statement=GRANULARITY_STATEMENT[gran],
                    rules_data=rules_data,
                    sentence_text=sentence_text,
                    dictionary_entries=dict_str,
                )

                batch_requests.append({
                    "key":    key,
                    "method": "generateContent",
                    "model":  GEMINI_MODEL,
                    "request": {
                        "contents": [{"parts": [{"text": prompt}]}]
                    },
                })

            save_jsonl(out_file, batch_requests)
            total_tok = sum(estimate_tokens(r["request"]["contents"][0]["parts"][0]["text"]) for r in batch_requests)
            print(f"  [{gran}] Saved {len(batch_requests)} prompts  (~{total_tok:,} tokens).")

    print("\nPHASE 1 COMPLETE")


# ─────────────────────────── PHASE 2: SUBMIT ────────────────────────────────────

def _chunks_from_jsonl(path: Path, token_limit: int) -> list[list[str]]:
    """Split a JSONL file into chunks that each stay under token_limit."""
    chunks, current, cur_tok = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                text  = json.loads(line)["request"]["contents"][0]["parts"][0]["text"]
                toks  = estimate_tokens(text)
            except Exception:
                continue
            if current and cur_tok + toks > token_limit:
                chunks.append(current)
                current, cur_tok = [], 0
            current.append(line)
            cur_tok += toks
    if current:
        chunks.append(current)
    return chunks


def _poll_until_done(client, job_name: str, label: str, log: dict, log_key: str) -> str:
    """Poll a Gemini batch job until terminal state. Returns 'JOB_STATE_SUCCEEDED' or error state."""
    wait = POLL_INTERVAL
    while True:
        try:
            status = client.batches.get(name=job_name)
            state  = status.state.name
            print(f"  [{label}] Status: {state}")
            if state == "JOB_STATE_SUCCEEDED":
                log[log_key]["status"] = "COMPLETED"
                save_log(log)
                return state
            if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                log[log_key]["status"] = state
                save_log(log)
                return state
            wait = POLL_INTERVAL
        except Exception as exc:
            print(f"  [{label}] Connection error: {exc}. Retrying in {wait}s...")
            wait = min(wait * 2, POLL_INTERVAL)
        print(f"  [{label}] Waiting {wait}s...")
        time.sleep(wait)


def _submit_and_wait(client, chunk_lines: list[str], label: str, temp_path: Path) -> tuple[str, str]:
    """Upload chunk, submit batch, wait for success. Returns (job_name, final_state)."""
    # Write temp JSONL
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        f.writelines(chunk_lines)

    # Submit with retry on quota errors
    while True:
        try:
            print(f"  [{label}] Uploading {len(chunk_lines)} lines...")
            uploaded = client.files.upload(file=str(temp_path), config={"mime_type": "application/jsonl"})
            job = client.batches.create(
                model=GEMINI_MODEL,
                src=uploaded.name,
                config={"display_name": label},
            )
            temp_path.unlink(missing_ok=True)
            return job.name
        except Exception as e:
            if "429" in str(e):
                print(f"  [{label}] Quota limit (429). Waiting 15 min...")
                time.sleep(900)
            else:
                raise


def phase_submit():
    print("\n" + "=" * 60)
    print("PHASE 2: Submitting batches to Gemini Batch API")
    print("=" * 60)

    client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
    log    = load_log()

    for lang in LANGUAGES:
        lang_dir = OUT_DIR / lang
        for gran in GRANULARITIES:
            batch_file   = lang_dir / f"{lang}_icl_{gran}_batch.jsonl"
            results_file = lang_dir / f"{lang}_icl_{gran}_results.jsonl"

            if results_file.exists():
                print(f"\n[{lang}/{gran}] Results already exist. Skipping.")
                continue
            if not batch_file.exists():
                print(f"\n[{lang}/{gran}] Batch file not found. Run phase 1 first.")
                continue

            print(f"\n[{lang}/{gran}] Processing...")
            chunks = _chunks_from_jsonl(batch_file, TOKEN_LIMIT)
            print(f"  Split into {len(chunks)} chunk(s).")

            all_result_lines = []

            for ci, chunk_lines in enumerate(chunks):
                chunk_key    = f"{lang}_{gran}_chunk{ci}"
                res_path     = lang_dir / f"{lang}_icl_{gran}_chunk{ci}_results.jsonl"
                temp_path    = lang_dir / f"_temp_{lang}_{gran}_chunk{ci}.jsonl"

                # ── Already fully done ──
                if res_path.exists() and log.get(chunk_key, {}).get("status") == "COMPLETED":
                    print(f"  [chunk {ci}] Already completed. Loading results.")
                    all_result_lines.extend(load_jsonl(res_path))
                    continue

                # ── Already submitted, just poll ──
                job_name = log.get(chunk_key, {}).get("job_name")
                if not job_name:
                    job_name = _submit_and_wait(client, chunk_lines, chunk_key, temp_path)
                    log[chunk_key] = {
                        "job_name": job_name,
                        "status": "RUNNING",
                        "lang": lang,
                        "granularity": gran,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    save_log(log)
                    print(f"  [chunk {ci}] Submitted: {job_name}")

                final_state = _poll_until_done(client, job_name, f"chunk {ci}", log, chunk_key)

                if final_state != "JOB_STATE_SUCCEEDED":
                    print(f"  [chunk {ci}] Job ended with {final_state}. Skipping download.")
                    continue

                # ── Download results ──
                status  = client.batches.get(name=job_name)
                content = client.files.download(file=status.dest.file_name)
                with open(res_path, "wb") as f:
                    f.write(content)
                log[chunk_key]["status"] = "COMPLETED"
                save_log(log)
                print(f"  [chunk {ci}] Downloaded results to {res_path}.")
                all_result_lines.extend(load_jsonl(res_path))

            # ── Merge chunks → single results file ──
            if all_result_lines:
                save_jsonl(results_file, all_result_lines)
                print(f"[{lang}/{gran}] Merged {len(all_result_lines)} results → {results_file}.")

    print("\nPHASE 2 COMPLETE")


# ─────────────────────────── PHASE 3: EVAL ──────────────────────────────────────

def _parse_translation(text: str) -> str:
    """
    Extract english_translation from the model's YAML output block.
    Falls back to the last non-empty line.
    """
    # Try YAML: english_translation: "..." (with or without quotes)
    m = re.search(
        r'english_translation\s*:\s*["\']?(.*?)["\']?\s*(?:```|$)',
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip('"').strip("'")
    # Fallback: last non-empty line of response
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def phase_eval():
    print("\n" + "=" * 60)
    print("PHASE 3: Computing evaluation metrics")
    print("=" * 60)

    all_metrics: dict[str, dict] = {}

    for lang in LANGUAGES:
        # Build key → reference mapping from test.jsonl (ground-truth) aligned
        # positionally with test_inference.jsonl (which carries the keys).
        inf_path = DATA_DIR / "gemini_finetune_data" / lang / "test_inference.jsonl"
        gt_path  = DATA_DIR / "gemini_finetune_data" / lang / "test.jsonl"
        inf_lines = load_jsonl(inf_path)
        gt_lines  = load_jsonl(gt_path)

        key_to_ref: dict[str, str] = {}
        for inf_line, gt_line in zip(inf_lines, gt_lines):
            key = inf_line["key"]
            try:
                ref = gt_line["contents"][1]["parts"][0]["text"]
            except (KeyError, IndexError):
                continue
            key_to_ref[key] = ref.strip()
        print(f"\n[{lang}] {len(key_to_ref)} reference sentences loaded.")

        for gran in GRANULARITIES:
            results_file = OUT_DIR / lang / f"{lang}_icl_{gran}_results.jsonl"
            if not results_file.exists():
                print(f"  [{gran}] Results not found. Run phase 2 first.")
                continue

            results  = load_jsonl(results_file)
            hyps, refs, skipped = [], [], 0

            for entry in results:
                key = entry.get("key", "")
                try:
                    raw = entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    skipped += 1
                    continue

                ref = key_to_ref.get(key, "")
                if not ref:
                    skipped += 1
                    continue

                hyps.append(_parse_translation(raw))
                refs.append(ref)

            if not hyps:
                print(f"  [{gran}] No valid predictions to evaluate.")
                continue

            bleu   = sacrebleu.corpus_bleu(hyps, [refs])
            chrf   = sacrebleu.corpus_chrf(hyps, [refs])
            chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)

            label = f"{lang}_{gran}"
            all_metrics[label] = {
                "language":    lang,
                "granularity": gran,
                "n_evaluated": len(hyps),
                "n_skipped":   skipped,
                "BLEU":   round(bleu.score,   2),
                "ChrF":   round(chrf.score,   2),
                "ChrF++": round(chrfpp.score, 2),
            }
            print(
                f"  [{gran}]  BLEU={bleu.score:.2f}  "
                f"ChrF={chrf.score:.2f}  ChrF++={chrfpp.score:.2f}  "
                f"(n={len(hyps)}, skipped={skipped})"
            )

    # ── Save & print summary table ─────────────────────────────────────────────
    metrics_out = OUT_DIR / "icl_baseline_metrics.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_out}")

    print("\n" + "-" * 62)
    print(f"{'Condition':<35} {'BLEU':>6} {'ChrF':>6} {'ChrF++':>7}")
    print("-" * 62)
    for label, m in all_metrics.items():
        print(f"{label:<35} {m['BLEU']:>6.2f} {m['ChrF']:>6.2f} {m['ChrF++']:>7.2f}")
    print("-" * 62)

    print("\nPHASE 3 COMPLETE")


# ─────────────────────────── MAIN ───────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    phase_build(embed_model)
    phase_submit()
    phase_eval()
