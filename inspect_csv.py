import ast
import os
import re
import pandas as pd


CSV_PATH = (
    "/workspace/pixels_vs_priors/"
    "most_instances_plural_bigger_color_new_MLLM_results_"
    "most_True_counterfact_line_True_qwen2.5_base_dyco_full.csv"
)


COLOR_ALIASES = {
    "gray": "grey",
    "grey": "grey",
}


def get_prompt_mode(csv_path):
    filename = os.path.basename(csv_path)

    match = re.search(
        r"_(normal|dyco)_(mini|full)\.csv$",
        filename
    )

    if not match:
        raise ValueError(
            f"Could not determine prompt mode from filename: {filename}"
        )

    return match.group(1)


PROMPT_MODE = get_prompt_mode(CSV_PATH)


def extract_answer(text):
    text = str(text).strip()

    # Only parse <answer> tags when using the DyCo prompt
    if PROMPT_MODE == "dyco":
        matches = re.findall(
            r"<answer>\s*(.*?)\s*</answer>",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if matches:
            return matches[-1].strip()

    return text


def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return COLOR_ALIASES.get(text, text)


def parse_correct_answers(value):
    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        return [parsed]

    except (ValueError, SyntaxError, TypeError):
        return [value]


def is_correct(row):
    prediction = normalize(row["extracted_answer"])

    correct_answers = [
        normalize(answer)
        for answer in parse_correct_answers(row["correct_answer"])
    ]

    return prediction in correct_answers


def is_visual_override(row):
    prediction = normalize(row["extracted_answer"])
    counterfactual_answer = normalize(row["incorrect_answer"])

    return prediction == counterfactual_answer


df = pd.read_csv(CSV_PATH)

df["extracted_answer"] = df["generated_text"].apply(extract_answer)

df["is_correct"] = df.apply(is_correct, axis=1)
df["is_visual_override"] = df.apply(is_visual_override, axis=1)

total = len(df)

correct = int(df["is_correct"].sum())
visual_override = int(df["is_visual_override"].sum())
other = total - correct - visual_override


print(f"File: {os.path.basename(CSV_PATH)}")
print(f"Detected prompt mode: {PROMPT_MODE}")
print(f"Total examples: {total}")

print(
    f"Correct prior answer: "
    f"{correct}/{total} = {100 * correct / total:.2f}%"
)

print(
    f"Counterfactual visual override: "
    f"{visual_override}/{total} = {100 * visual_override / total:.2f}%"
)

print(
    f"Other: "
    f"{other}/{total} = {100 * other / total:.2f}%"
)


if PROMPT_MODE == "dyco":
    has_answer_tag = df["generated_text"].astype(str).str.contains(
        r"<answer>.*?</answer>",
        case=False,
        regex=True
    )

    num_with_answer_tag = int(has_answer_tag.sum())

    print(
        f"Outputs with <answer> tags: "
        f"{num_with_answer_tag}/{total} = "
        f"{100 * num_with_answer_tag / total:.2f}%"
    )


other_cases = df[
    (~df["is_correct"]) &
    (~df["is_visual_override"])
]

print(f"\nOther cases: {len(other_cases)}")

if len(other_cases) > 0:
    print(
        other_cases[
            [
                "object",
                "correct_answer",
                "incorrect_answer",
                "generated_text",
                "extracted_answer",
            ]
        ].head(20).to_string(index=False)
    )