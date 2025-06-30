import pandas as pd
import json
import re
import os

# Paths
excel_path = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels/patient_characteristics.xlsx"
output_dir = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels"

class_output_path = os.path.join(output_dir, "class_prompts.json")
label_output_path = os.path.join(output_dir, "label_prompts.json")

# ---------------------------
# Step 1: Create Class Prompts from the Second Sheet
# ---------------------------
df_class = pd.read_excel(excel_path, sheet_name=1)  # second sheet

# Ensure required columns for class prompts
required_class_cols = ["Engels", "Diagnose code"]
for col in required_class_cols:
    if col not in df_class.columns:
        raise ValueError(
            f"Required column '{col}' not found in the second sheet for class prompts."
        )


def normalize_class_name(name: str) -> str:
    replacements = {"&": "and", "(": "", ")": "", "/": "or", "+": "and"}
    for old, new in replacements.items():
        name = name.replace(old, new)

    name = re.sub(r"\s+", " ", name).strip()
    return name


class_to_prompt = {}

for _, row in df_class.iterrows():
    engels = row["Engels"]
    diagnose_code = row["Diagnose code"]  # numeric class
    normalized = normalize_class_name(str(engels))
    if normalized:
        normalized = normalized[0].lower() + normalized[1:]
    else:
        normalized = "lesion"

    a_or_an = "an" if normalized[0] in "aeiou" else "a"
    prompt = f"Histopathological evaluation of this skin specimen reveals {a_or_an} {normalized}."

    diagnose_code_str = str(int(diagnose_code))  # convert to string for JSON keys
    class_to_prompt[diagnose_code_str] = prompt

# Write class prompts
with open(class_output_path, "w", encoding="utf-8") as f:
    json.dump(class_to_prompt, f, indent=4, ensure_ascii=False)
print(f"Class prompts successfully written to {class_output_path}")

# ---------------------------
# Step 2: Create Label Prompts from the First Sheet
# ---------------------------
df_label = pd.read_excel(excel_path, sheet_name=0)  # first sheet

# Ensure required column for label prompts
if "label" not in df_label.columns:
    raise ValueError(
        "Required column 'label' not found in the first sheet for label prompts."
    )

# Extract unique labels
unique_labels = df_label["label"].dropna().unique()

label_to_prompt = {}
for lbl in unique_labels:
    lbl_str = str(lbl).upper()
    if lbl_str == "TRUE":
        label_to_prompt["1"] = "Histological analysis reveals a complex skin specimen."
    elif lbl_str == "FALSE":
        label_to_prompt["2"] = "Histological analysis reveals a simple skin specimen."
    else:
        raise ValueError(
            f"Unexpected label value '{lbl_str}' found. Expected 'TRUE' or 'FALSE'."
        )

# Write label prompts
with open(label_output_path, "w", encoding="utf-8") as f:
    json.dump(label_to_prompt, f, indent=4, ensure_ascii=False)
print(f"Label prompts successfully written to {label_output_path}")
