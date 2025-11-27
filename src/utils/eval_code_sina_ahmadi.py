response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=api_messages,
    extra_body={"thinking": {"type": "disabled", "budget_tokens": 0}},
)
import json
import gc
import torch
import json
import re
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint


def remove_tripple_quotes(text):
    # based on https://github.com/ZurichNLP/romansh_mt_eval/blob/main/wmt-collect-translations/main_romansh.py
    # check if there are exactly two occurences of ``` in the text
    if text.count("```") == 2:
        # get only the text inbetween the tripple quotes
        text = text.split("```")[1]

    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    # replace new lines and tabs with spaces
    text = text.replace("\n", " ").replace("\r", "").replace("\t", " ")

    return text


model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
bleu = BLEU(
    lowercase=False, effective_order=True, max_ngram_order=6, smooth_method="none"
)
chrf = CHRF(char_order=6, word_order=0, beta=2, lowercase=False)

varieties = ["puter", "rumgr", "surmiran", "sursilv", "sutsilv", "vallader"]
base_dir = "/home/seahma/ROMANSH/prompting/outputs/"

# Configure what to evaluate
# prompts = ["prompt_basic_0", "prompt_basic_1", "prompt_lemmata", "prompt_lemmata_rare",
# 		  "prompt_few_shot", "prompt_hybrid", "prompt_hybrid_rare"]
# suffixes = ["", "_no_variety", "_no_source"]  # Change to "_no_variety", "_no_source", "_with_reasoning", etc.

# prompts = ["prompt_hybrid", "prompt_hybrid_rare"]
# suffixes = ["", "_only_hybrid_no_variety", "_only_hybrid_no_source", "_only_hybrid_with_reasoning"]

# prompts = ["prompt_basic_0", "prompt_basic_1", "prompt_lemmata", "prompt_lemmata_rare"]#"prompt_few_shot", "prompt_hybrid", "prompt_hybrid_rare"
# suffixes = ["_with_reasoning"]

base_dir = "/home/seahma/ROMANSH/"
prompts = ["prompt_few_shot", "prompt_hybrid", "prompt_hybrid_rare"]
suffixes = ["_with_reasoning"]

for suffix in suffixes:

    prompt_outputs = {}
    for prompt in prompts:
        filename = f"{prompt}{suffix}.json"
        prompt_outputs[prompt] = f"{base_dir}{filename}"

    # Write header to results.md
    with open("results-new.md", "a") as f:
        f.write(f"\n## Results for suffix: '{suffix}'\n")
        f.write(f"Prompts evaluated: {', '.join(prompts)}\n\n")

    for prompt in prompt_outputs:
        print(f"\n=== {prompt} ===")

        with open(prompt_outputs[prompt], "r") as f:
            json_file = json.load(f)

        performance_table = [f"### {prompt}", f"| Variety | BLEU | chrF | XComet |"]
        performance_table.append("| -------- | ------- |-------- | ------- |")

        for variety in varieties:
            if variety not in json_file:
                continue
            print(variety)
            refs, sys = list(), list()

            for entry in json_file[variety]:
                refs.append(entry["de"])
                sys.append(remove_tripple_quotes(entry["sys"]))

            if not sys:
                continue

            bleu_score = bleu.corpus_score(sys, [refs])
            bleu_value = float(
                re.search(r"BLEU = (\d+\.\d+)", str(bleu_score)).group(1)
            )
            chrf_score = chrf.corpus_score(sys, [refs])
            chrf_value = float(
                re.search(r"chrF2 = (\d+\.\d+)", str(chrf_score)).group(1)
            )

            data = []
            for sys_sent, ref_sent in zip(sys, refs):
                data.append({"mt": sys_sent, "ref": ref_sent})

            model_output = model.predict(data, batch_size=8, gpus=1, progress_bar=False)
            performance_table.append(
                f"| {variety} | {bleu_value} | {chrf_value} | {model_output.system_score} |"
            )

        # Print and save results for this prompt
        print("\n".join(performance_table[1:]))  # Skip the header for console

        with open("results%s.md" % suffix, "a") as f:
            f.write(f"### {prompt} - {prompt_outputs[prompt]}\n")
            f.write(
                "\n".join(performance_table[1:])
            )  # Skip the title since we include it above with source
            f.write("\n\n")

# gc.collect()
# torch.cuda.empty_cache()

# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
