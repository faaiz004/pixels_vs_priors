import re
import pandas as pd
import numpy as np

CSV_PATH = (
    "/Users/faaizumer/Desktop/Fall 2026/ATNLP/Presentation + Lit Review/pixels_vs_priors/"
    "most_instances_plural_bigger_size_new_MLLM_results_"
    "most_True_counterfact_line_True_qwen2.5_base_dyco_full.csv"
)


df = pd.read_csv(CSV_PATH)


def extract_think(text):
    if pd.isna(text):
        return None

    match = re.search(
        r"<think>\s*(.*?)\s*</think>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return None


def count_words(text):
    if text is None:
        return np.nan

    return len(re.findall(r"\b[\w'-]+\b", text))


df["think_text"] = df["generated_text"].apply(extract_think)

def count_words(text):
    if pd.isna(text):
        return np.nan

    return len(re.findall(r"\b[\w'-]+\b", str(text)))

df["reasoning_words"] = df["think_text"].apply(count_words)

valid = df["reasoning_words"].dropna()

print(f"Total examples: {len(df)}")
print(f"Complete <think> blocks: {len(valid)}/{len(df)} = {100 * len(valid)/len(df):.2f}%")

print("\nReasoning length in words:")
print(f"Mean:   {valid.mean():.2f}")
print(f"Median: {valid.median():.2f}")
print(f"Std:    {valid.std():.2f}")
print(f"Min:    {valid.min():.0f}")
print(f"Max:    {valid.max():.0f}")

print("\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"{p:>2}th: {np.percentile(valid, p):.1f} words")