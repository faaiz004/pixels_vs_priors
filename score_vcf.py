
"""Score every Visual CounterFact results CSV in a folder.

Usage:
    python score_vcf.py --results_dir /content/drive/MyDrive/vcf_results

For each CSV it reports how often the model gave the prior answer
(correct_answer), the counterfactual image answer (incorrect_answer), or
something else, plus accuracy for that condition and output-format rates.
Writes scores_per_file.csv and prints the summary tables.
"""
import argparse
import ast
import glob
import os
import re

import pandas as pd

FILENAME_RE = re.compile(
    r"most_instances_plural_bigger_(?P<task>color|size)_new_MLLM_results_"
    r"most_(?P<most>True|False)_(?P<image>real|counterfact)_line_True_"
    r"qwen2\.5_(?P<size>3b|7b)_(?P<checkpoint>base|dyco|grpo)_"
    r"(?P<mode>normal|dyco)_max(?P<max_tokens>\d+)_(?P<dataset>mini|full)\.csv$"
)

COLOR_ALIASES = {"gray": "grey"}


def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(the|a|an) ", "", text)
    return COLOR_ALIASES.get(text, text)


def parse_list(value):
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return [value]


def extract_answer(text, mode):
    """Return (answer, format_status)."""
    text = "" if pd.isna(text) else str(text).strip()
    if mode == "dyco":
        tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
        if tagged:
            return tagged[-1], "answer_tag"
        if re.search(r"<think>", text, flags=re.I) and not re.search(r"</think>", text, flags=re.I):
            return "", "truncated_in_think"
        if re.search(r"<answer>", text, flags=re.I):
            return re.split(r"<answer>", text, flags=re.I)[-1], "unclosed_answer"
        after_think = re.split(r"</think>", text, flags=re.I)[-1]
        return after_think, "no_answer_tag"
    return text, "direct"


def classify(row, task):
    """Return 'prior', 'image' or 'other' for one prediction."""
    pred = normalize(row["extracted_answer"])
    if not pred:
        return "other"
    prior = [normalize(a) for a in parse_list(row["correct_answer"])]
    image = normalize(row["incorrect_answer"])

    if task == "color":
        if pred in prior:
            return "prior"
        if pred == image:
            return "image"
        return "other"

    # size: the answer should name exactly one of the two objects
    def mentions(name):
        return re.search(rf"\b{re.escape(name)}s?\b", pred) is not None

    hit_prior = any(mentions(p) for p in prior)
    hit_image = mentions(image)
    if hit_prior and not hit_image:
        return "prior"
    if hit_image and not hit_prior:
        return "image"
    return "other"


def score_file(path, info):
    df = pd.read_csv(path)
    extracted = df["generated_text"].apply(lambda t: extract_answer(t, info["mode"]))
    df["extracted_answer"] = [a for a, _ in extracted]
    df["format"] = [f for _, f in extracted]
    df["source"] = df.apply(lambda r: classify(r, info["task"]), axis=1)

    n = len(df)
    pct = lambda mask: 100 * mask.sum() / n
    prior_pct = pct(df["source"] == "prior")
    image_pct = pct(df["source"] == "image")

    # Gold answer: the image for counterfactual + "this", the prior otherwise.
    gold_is_image = info["image"] == "counterfact" and info["most"] == "False"
    accuracy = image_pct if gold_is_image else prior_pct

    row = dict(info, file=os.path.basename(path), n=n,
               accuracy=accuracy, prior_pct=prior_pct, image_pct=image_pct,
               other_pct=100 - prior_pct - image_pct)
    if info["mode"] == "dyco":
        row["answer_tag_pct"] = pct(df["format"] == "answer_tag")
        row["truncated_pct"] = pct(df["format"] == "truncated_in_think")
    return row, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--show_other", type=int, default=0,
                        help="Print this many 'other' examples per file.")
    args = parser.parse_args()

    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "most_instances_*.csv"))):
        m = FILENAME_RE.search(os.path.basename(path))
        if not m or m.group("dataset") != "full":
            skipped.append(os.path.basename(path))
            continue
        row, df = score_file(path, m.groupdict())
        rows.append(row)
        if args.show_other:
            other = df[df["source"] == "other"]
            if len(other):
                print(f"\n--- 'other' answers in {row['file']} ({len(other)})")
                print(other[["object", "correct_answer", "incorrect_answer", "extracted_answer"]]
                      .head(args.show_other).to_string(index=False))

    if skipped:
        print(f"Skipped {len(skipped)} file(s) with unrecognised names or mini runs:")
        for s in skipped:
            print("   ", s)
    if not rows:
        print("No result files found.")
        return

    scores = pd.DataFrame(rows)
    scores["model"] = scores["size"] + "-" + scores["checkpoint"]
    scores["question"] = scores["most"].map({"True": "most", "False": "this"})
    scores["mode"] = scores["mode"].map({"normal": "direct", "dyco": "reasoning"})
    out_path = os.path.join(args.results_dir, "scores_per_file.csv")
    scores.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Accuracy (%) ===")
    print("Gold is the image for counterfact+this, the prior everywhere else.\n")
    acc = scores.pivot_table(index=["task", "model", "mode"],
                             columns=["image", "question"], values="accuracy")
    print(acc.round(1).to_string())

    cf = scores[scores["image"] == "counterfact"]
    if len(cf):
        sa = cf.pivot_table(index=["task", "model", "mode"], columns="question", values="accuracy")
        if {"this", "most"} <= set(sa.columns):
            sa["source_appropriate"] = (sa["this"] + sa["most"]) / 2
            print("\n=== Counterfactual images: accuracy and source-appropriateness (%) ===")
            print("this = follows the image when asked about it; most = keeps the prior when asked in general.\n")
            print(sa.round(1).to_string())

    reasoning = scores[scores["mode"] == "reasoning"]
    if len(reasoning):
        print("\n=== Reasoning format (%) ===")
        print(reasoning.pivot_table(index=["task", "model"], columns=["image", "question"],
                                    values="answer_tag_pct").round(1).to_string())
        print("\nTruncated inside <think> (%):")
        print(reasoning.pivot_table(index=["task", "model"], columns=["image", "question"],
                                    values="truncated_pct").round(1).to_string())

    print("\n=== Unparsed answers ('other', %) ===")
    print(scores.pivot_table(index=["task", "model", "mode"], columns=["image", "question"],
                             values="other_pct").round(1).to_string())
    print(f"\nPer-file scores saved to {out_path}")


if __name__ == "__main__":
    main()