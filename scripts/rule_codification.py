import json
import os
import time
import yaml
from collections import Counter
from pathlib import Path
from google import genai
from google.genai import types
import typer


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val


app = typer.Typer(
    name="rule_codification",
    help="Deduplicate, merge, and codify extracted grammatical rules using Gemini batch API",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def merge_rules_to_canonical(extracted_rules):
    rule_groups = {}
    for rule in extracted_rules:
        if isinstance(rule["target_pos"], list):
            rule["target_pos"] = ",".join(sorted(rule["target_pos"]))
        key = (
            rule["target_pos"],
            rule["morpheme"],
            rule["unimorph_feature"],
            rule["unimorph_value"],
        )
        if key not in rule_groups:
            rule_groups[key] = []
        rule_groups[key].append(rule)

    canonical_rules = []
    for key, group in rule_groups.items():
        canonical_rule = max(
            group,
            key=lambda r: len(r["context_dependency"]) if r["context_dependency"] != "N/A" else 0,
        )

        all_descriptions = {r["description"] for r in group}
        all_semantic_triggers = {r["semantic_trigger"] for r in group}

        canonical_rule["semantic_trigger"] = " | ".join(sorted(all_semantic_triggers))
        canonical_rule["description"] = " | ".join(sorted(all_descriptions))

        all_context_dependencies = {
            r["context_dependency"] for r in group if r["context_dependency"] != "N/A"
        }
        if all_context_dependencies:
            canonical_rule["context_dependency"] = " AND ".join(sorted(all_context_dependencies))
        else:
            canonical_rule["context_dependency"] = "N/A"

        section_paragraph_ids = []
        for r in group:
            sec_id = r.get("section_id")
            para_id = r.get("paragraph_id")
            if sec_id is not None and para_id is not None:
                section_paragraph_ids.append((sec_id, para_id))

        canonical_rule["section_paragraph_ids"] = section_paragraph_ids
        canonical_rule.pop("section_id", None)
        canonical_rule.pop("paragraph_id", None)

        canonical_rules.append(canonical_rule)

    return canonical_rules


CODIFICATION_PROMPT = """
You are a computational linguist.
Task: Translate the single input rule into a concise pseudo-code function, ensuring the output YAML is perfectly parsable.

## Output Schema (YAML)
Output MUST be a single YAML block containing these two fields:
1.  **canonical_description:** A single, non-redundant sentence synthesizing the unique meaning, usage, and all conditions from the input fields. **MUST be enclosed in double quotes if it contains colons (:) or starts with a YAML reserved character.**
2.  **lrl_code:** The Function that applies the rule, based on the synthesized conditions.

## Codification Rules
1.  Function MUST be named `ApplyRule(STEM, POS)`.
2.  Affixation must use the **'application_string'** field.
3.  Implement all unique conditions using `IF/ELSE IF` checks (`IF STEM_ENDS_IN('x')`, `IF POS == 'Y'`, etc.).
4.  If the morpheme should not be applied (conditions not met), the function MUST return the original `STEM`.

## Example (Fixing the Colon Issue)

### Input Rule (YAML):
category: surface_rule
description: Applies the clitic =ka to NOUN to mark Lative Case (LAT).
target_pos: NOUN
morpheme: =ka
application_string: STEM + '=ka'
unimorph_value: LAT
context_dependency: Morpheme variation: If stem ends in a vowel, clitic becomes 'nga'.
semantic_trigger: Use when the noun indicates a destination (e.g., 'to the house').

### Expected Output:
canonical_description: "The clitic =ka marks Lative Case (LAT) on a NOUN, indicating a destination. Allomorphy is based on the final phoneme: it becomes 'nga' after vowels and remains 'ka' otherwise."
lrl_code: |
  FUNCTION ApplyRule(STEM, POS):
    IF POS != 'NOUN':
      RETURN STEM

    FINAL_PHONEME = GET_LAST_PHONEME(STEM)

    IF IS_VOWEL(FINAL_PHONEME):
      RETURN STEM + 'nga'
    ELSE:
      RETURN STEM + 'ka'

## Input Rule (YAML)
Process this rule.

---
{input_rule}
---
"""

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@app.command()
def codify(
    filename: str = typer.Argument(
        None, help="Path to extracted rules JSON (e.g. data/extracted_rules/kalamang_extracted_rules.json)"
    ),
    dry_run: bool = typer.Option(False, help="Build batch file only, skip API submission"),
):
    with open(filename, encoding="utf-8") as f:
        cleaned_rules_raw = json.load(f)

    language_name = Path(filename).stem.split("_")[0]
    print(f"Language: {language_name}")

    # Flatten — input may be list-of-lists (from rule_extraction) or flat list
    if cleaned_rules_raw and isinstance(cleaned_rules_raw[0], list):
        flattened = [rule for section in cleaned_rules_raw for rule in section]
    else:
        flattened = cleaned_rules_raw
    print(f"Total rules loaded: {len(flattened)}")

    # Deduplicate by description
    descriptions_seen = set()
    categories = Counter()
    flattened_unique = []
    for rule in flattened:
        desc = rule.get("description", "")
        category = rule.get("category", "")
        if desc in descriptions_seen:
            continue
        categories[category] += 1
        descriptions_seen.add(desc)
        flattened_unique.append(rule)
    print(f"After deduplication: {len(flattened_unique)} rules")
    print(f"Category breakdown: {dict(categories)}")

    surface_rules = [r for r in flattened_unique if r.get("category") == "surface_rule"]
    print(f"Surface rules: {len(surface_rules)}")

    # Merge rules that share (target_pos, morpheme, unimorph_feature, unimorph_value)
    final_rules = merge_rules_to_canonical(surface_rules)
    print(f"After canonical merge: {len(final_rules)} rules")

    # Assign rule IDs and build codification batch
    rule_id = 0
    batch_prompts = []
    for rule in final_rules:
        rule.pop("rule_id", None)
        ignore_keys = {"section_paragraph_ids", "category"}
        rule_filtered = {k: v for k, v in rule.items() if k not in ignore_keys}
        rule_yaml = yaml.dump(rule_filtered, default_flow_style=False)
        prompt = CODIFICATION_PROMPT.format(input_rule=rule_yaml)
        batch_prompts.append((rule_id, prompt))
        rule["rule_id"] = rule_id
        rule_id += 1

    Path("scratch").mkdir(exist_ok=True)

    # Save rules with IDs for later merge
    rules_for_cleanup_path = f"scratch/{language_name}_rules_for_cleanup.json"
    with open(rules_for_cleanup_path, "w", encoding="utf-8") as f:
        json.dump(final_rules, f, ensure_ascii=False, indent=4)

    # Build batch JSONL
    batch_jsonl_path = f"scratch/{language_name}_codification_batch_requests.jsonl"
    batch_file = []
    for rule_id_val, prompt in batch_prompts:
        batch_file.append({
            "key": f"rule_{rule_id_val}",
            "request": {"contents": [{"parts": [{"text": prompt}]}]},
        })

    with open(batch_jsonl_path, "w", encoding="utf-8") as f:
        for item in batch_file:
            f.write(json.dumps(item) + "\n")
    print(f"Batch file written: {batch_jsonl_path} ({len(batch_file)} requests)")

    if dry_run:
        print("Dry run — skipping Gemini batch submission.")
        return

    client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))

    # Upload
    uploaded_file = client.files.upload(
        file=batch_jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"{language_name}_codification_batch_requests",
            mime_type="jsonl",
        ),
    )
    print(f"Uploaded file: {uploaded_file.name}")

    # Submit batch job
    batch_job = client.batches.create(
        model="gemini-2.5-flash",
        src=uploaded_file.name,
        config={"display_name": f"{language_name}_rule_codification_batch_job"},
    )
    print(f"Batch job created: {batch_job.name}")

    # Poll
    batch_job = client.batches.get(name=batch_job.name)
    while batch_job.state.name not in COMPLETED_STATES:
        print(f"Status: {batch_job.state.name} — waiting 5 min...")
        time.sleep(300)
        batch_job = client.batches.get(name=batch_job.name)
    print(f"Job finished: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Error: {batch_job.error}")
        raise typer.Exit(code=1)

    # Download results
    result_file_name = batch_job.dest.file_name
    file_content = client.files.download(file=result_file_name)
    raw_results_path = f"scratch/{language_name}_codification_batch_results.txt"
    with open(raw_results_path, "wb") as f:
        f.write(file_content)
    print(f"Raw results saved to {raw_results_path}")

    # Parse results
    gemini_results = []
    for res_line in file_content.decode("utf-8").splitlines():
        res = json.loads(res_line)
        key = res["key"]
        text = res["response"]["candidates"][0]["content"]["parts"][0]["text"]
        gemini_results.append({"rule_id": key, "cleaned_rule": text})

    # Parse YAML from each response
    for result in gemini_results:
        raw = result["cleaned_rule"].strip()
        # Strip markdown code fences
        if "```yaml" in raw:
            raw = raw[raw.find("```yaml") + 7 : raw.rfind("```")]
        elif "```" in raw:
            raw = raw[raw.find("```") + 3 : raw.rfind("```")]
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            print(f"YAML parse error for {result['rule_id']}: {e}")
            result["cleaned_rule_dict"] = None
        else:
            result["cleaned_rule_dict"] = parsed

    with open(f"scratch/{language_name}_cleaned_rules_with_code.json", "w", encoding="utf-8") as f:
        json.dump(gemini_results, f, ensure_ascii=False, indent=4)

    # Reload rules and merge in codification output
    with open(rules_for_cleanup_path, encoding="utf-8") as f:
        final_rules_with_ids = json.load(f)

    results_by_id = {r["rule_id"]: r for r in gemini_results}

    for rule in final_rules_with_ids:
        rule_key = f"rule_{rule['rule_id']}"
        result = results_by_id.get(rule_key)
        if result and result.get("cleaned_rule_dict"):
            d = result["cleaned_rule_dict"]
            rule["description"] = d.get("canonical_description", rule["description"])
            rule["context_dependency"] = d.get("context_dependency", rule["context_dependency"])
            rule["semantic_trigger"] = d.get("semantic_trigger", rule["semantic_trigger"])
            rule["lrl_code"] = d.get("lrl_code", "")
            rule.pop("transform_templates", None)
            rule.pop("batch_id", None)
            rule.pop("cleaned_rule", None)
        else:
            print(f"No result for {rule_key}")

    Path("data/extracted_rules").mkdir(parents=True, exist_ok=True)
    output_path = f"data/extracted_rules/{language_name}_extracted_rules_final.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_rules_with_ids, f, ensure_ascii=False, indent=4)
    print(f"Final rules saved to {output_path} ({len(final_rules_with_ids)} rules)")


if __name__ == "__main__":
    app()
