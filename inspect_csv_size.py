import re
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    "/workspace/pixels_vs_priors/"
    "most_instances_plural_bigger_size_new_MLLM_results_"
    "most_True_counterfact_line_True_qwen2.5_dyco_dyco_full.csv"
)

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# EXTRACT FINAL ANSWER FROM DYCO OUTPUT
# ============================================================

def extract_final_answer(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Proper <answer>...</answer>
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if matches:
        return matches[-1].strip()

    # Opening <answer> without closing </answer>
    match = re.search(
        r"<answer>\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    # If </think> exists, use text following it
    if "</think>" in text.lower():
        parts = re.split(
            r"</think>",
            text,
            flags=re.IGNORECASE,
        )

        tail = parts[-1].strip()

        if tail:
            return tail

    # Otherwise use entire generated response
    return text


# ============================================================
# NORMALIZATION
# Only used to interpret the LLM JUDGE'S returned answer.
# It is NOT used to classify the original model response.
# ============================================================

def normalize_judge_output(text):
    text = str(text).strip().lower()

    text = text.strip(
        "\"'` \n\t.,!?;:"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# LOAD 7B JUDGE
# ============================================================

print(f"Loading judge: {JUDGE_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL
)

judge = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)

judge.eval()

print("Judge loaded.\n")


# ============================================================
# LLM-ONLY EVALUATOR
# ============================================================

def classify_with_llm(
    correct_answer,
    incorrect_answer,
    response,
):
    """
    correct_answer   = prior/world-knowledge answer
    incorrect_answer = counterfactual visual answer

    The LLM judge interprets the response.
    No deterministic parsing of the response is performed.
    """

    prompt = f"""
The original question asked:

Which is larger usually, {correct_answer} or {incorrect_answer}?

The model answered:

{response}

Which object did the model's answer choose as larger?

Reply with exactly one of:
{correct_answer}
{incorrect_answer}
OTHER

Do not answer the original question yourself. Only identify what the model's answer chose.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are evaluating another model's response. "
                "Follow the user's instructions exactly."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(judge.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = judge.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    raw_result = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    # ========================================================
    # The LLM has already made the classification.
    # We ONLY translate its returned object name into our labels.
    # ========================================================

    result_norm = normalize_judge_output(
        raw_result
    )

    correct_norm = normalize_judge_output(
        correct_answer
    )

    incorrect_norm = normalize_judge_output(
        incorrect_answer
    )

    if result_norm == correct_norm:
        return "prior", raw_result

    if result_norm == incorrect_norm:
        return "visual", raw_result

    if result_norm == "other":
        return "other", raw_result

    # Judge violated requested output format
    return "other", raw_result


# ============================================================
# LOAD CSV
# ============================================================

print(
    f"Loading CSV:\n{CSV_PATH}\n"
)

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# EXTRACT MODEL'S FINAL RESPONSE
# ============================================================

df["extracted_answer"] = (
    df["generated_text"].apply(
        extract_final_answer
    )
)


# ============================================================
# EVALUATE EVERY EXAMPLE WITH 7B LLM
# ============================================================

classifications = []
judge_outputs = []


for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Evaluating",
    unit="example",
):

    classification, judge_output = (
        classify_with_llm(
            row["correct_answer"],
            row["incorrect_answer"],
            row["extracted_answer"],
        )
    )

    classifications.append(
        classification
    )

    judge_outputs.append(
        judge_output
    )


df["classification"] = (
    classifications
)

df["judge_output"] = (
    judge_outputs
)


# ============================================================
# RESULTS
# ============================================================

total = len(df)

prior = int(
    (
        df["classification"]
        == "prior"
    ).sum()
)

visual = int(
    (
        df["classification"]
        == "visual"
    ).sum()
)

other = int(
    (
        df["classification"]
        == "other"
    ).sum()
)


print(
    "\n================ RESULTS ================\n"
)

print(
    f"Total examples: {total}"
)

print(
    f"Correct prior answer: "
    f"{prior}/{total} = "
    f"{100 * prior / total:.2f}%"
)

print(
    f"Counterfactual visual override: "
    f"{visual}/{total} = "
    f"{100 * visual / total:.2f}%"
)

print(
    f"Other: "
    f"{other}/{total} = "
    f"{100 * other / total:.2f}%"
)


# ============================================================
# PRINT ALL OTHER CASES
# ============================================================

other_df = df[
    df["classification"]
    == "other"
]


print(
    "\n================ OTHER CASES =============\n"
)


for idx, row in other_df.iterrows():

    print(
        f"ROW {idx + 1}"
    )

    print(
        f"PRIOR candidate:  "
        f"{row['correct_answer']}"
    )

    print(
        f"VISUAL candidate: "
        f"{row['incorrect_answer']}"
    )

    print(
        f"Model response:   "
        f"{row['extracted_answer']}"
    )

    print(
        f"Judge output:     "
        f"{row['judge_output']}"
    )

    print(
        "-" * 60
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = CSV_PATH.replace(
    ".csv",
    "_llm_judged.csv",
)


df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print(
    f"\nSaved judged results to:\n"
    f"{OUTPUT_PATH}"
)