import json
import re
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn.functional import softmax

# === Config ===
model_path = "saved/T5_flo_masking_10"
test_file = "datasets/florilege_T5_diffusion_test.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATION = 5
CONFIDENCE_THRESHOLD = 0.9
CATEGORIES = ["Taxon", "Phenotype", "Molecule", "Habitat", "Use"]

# === Model loading ===
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()

# === Data loading ===
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)
step1_data = [item for item in test_data if item.get("step") == 1]

# === Functions ===
def extract_chunks(text):
    parts = re.split(r'(<extra_id_\d+>)', text)
    tag_chunks = {}
    for i in range(1, len(parts) - 1, 2):
        tag = parts[i].strip()
        content = parts[i + 1].strip()
        if content.lower() == "no entity":
            tag_chunks[tag] = []
        else:
            tag_chunks[tag] = [e.strip() for e in content.split("@@") if e.strip()]
    return tag_chunks

def fill_predicted_chunks_back(input_text, pred_text):
    chunks = extract_chunks(pred_text)
    for tag, ents in chunks.items():
        input_text = input_text.replace(tag, ' @@ '.join(ents) if ents else 'no entity')
    return input_text

def revise_input_preserve_structure(input_text, pred_chunks, token_probs):
    print("[DEBUG] Revising Recognition Results by field + tag")
    match = re.search(r"(Recognition Results:)(.*)", input_text, re.DOTALL)
    if not match:
        return input_text, []

    rec_prefix = match.group(1)
    rec_text = match.group(2).strip()
    low_conf_entities = []

    for idx, field in enumerate(CATEGORIES):
        pattern = rf"{field}:\s*([^\n:]+)"
        field_match = re.search(pattern, rec_text)
        if not field_match:
            continue

        original_value = field_match.group(1).strip()
        new_value = original_value

        tag_pattern = re.findall(r"<extra_id_\d+>", original_value)
        for tag in tag_pattern:
            ents = pred_chunks.get(tag, [])
            if not ents:
                continue
            entity_text = " @@ ".join(ents)
            first_token = ents[0].split()[0]
            low_conf = any(
                token.startswith("▁" + first_token) and prob < CONFIDENCE_THRESHOLD
                for token, prob in token_probs
            )
            if not low_conf:
                print(f"[DEBUG] Replacing {field} {tag} → {entity_text}")
                new_value = new_value.replace(tag, entity_text)
            else:
                print(f"[DEBUG] Keeping {field} {tag} due to low confidence")
                low_conf_entities.append((field, entity_text))

        rec_text = re.sub(rf"{field}:\s*[^\n:]+", f"{field}: {new_value}", rec_text)

    updated_input = input_text.split("Recognition Results:")[0] + "Recognition Results: " + rec_text
    return updated_input, low_conf_entities

def extract_field_entities_from_recognition(text):
    match = re.search(r"Recognition Results:(.*)", text, re.DOTALL)
    if not match:
        return {}
    rec_section = match.group(1).strip()
    pattern = rf"({'|'.join(CATEGORIES)}):\s*([^:]+?)(?=(?: {'|'.join(CATEGORIES)}:|$))"
    fields = re.findall(pattern, rec_section, re.DOTALL)
    result = {}
    for field, value in fields:
        if value.strip().lower() == "no entity":
            result[field.strip()] = set()
        else:
            entities = [e.strip() for e in value.split("@@") if e.strip().lower() != "no entity"]
            result[field.strip()] = set(entities)
    return result

def map_target_chunks_to_fields(target_text):
    chunk_dict = extract_chunks(target_text)
    field_chunks = {}
    for idx, field in enumerate(CATEGORIES):
        tag = f"<extra_id_{idx}>"
        ents = chunk_dict.get(tag, [])
        field_chunks[field] = set(ents)
    return field_chunks

# === Main ===
tp = fp = fn = fp_unk = fn_unk = 0

for idx, sample in enumerate(step1_data):
    print("=" * 60)
    print(f"Sample {idx + 1}")
    input_text = sample["input_text"]
    target_text = sample["target_text"]
    pred_text = ""
    input_text_before_prediction = input_text

    for it in range(ITERATION):
        print(f"\niteration {it + 1}")
        print(f"input_text for next iteration:\n{input_text}")
        input_text_before_prediction = input_text

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=256,
                return_dict_in_generate=True,
                output_scores=True
            )
        generated_ids = outputs.sequences[0]
        scores = outputs.scores

        tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        token_probs = []
        for i, (token_id, logit) in enumerate(zip(generated_ids[1:], scores)):
            probs = softmax(logit, dim=-1)
            prob = probs[0, token_id].item()
            token_probs.append((tokens[i + 1], prob))

        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        pred_text = pred_text.replace("</s>", "").replace("<pad>", "")
        pred_chunks = extract_chunks(pred_text)

        print(f"pred_text: {pred_text}")

        new_input_text, low_conf_entities = revise_input_preserve_structure(input_text, pred_chunks, token_probs)
        print(f"new_input_text:\n{new_input_text}")

        if low_conf_entities:
            print(f"Low-confidence entities (below {CONFIDENCE_THRESHOLD}):")
            for field, ent in low_conf_entities:
                print(f"  - {field}: {ent}")

        if new_input_text.strip() == input_text.strip():
            print("No change detected. Early stopping.")
            break
        else:
            input_text = new_input_text

    
    filled_input_text = fill_predicted_chunks_back(input_text_before_prediction, pred_text)
    final_pred = extract_field_entities_from_recognition(filled_input_text)
    final_true = map_target_chunks_to_fields(target_text)

    print(f"\nFinal filled input_text (for eval):\n{filled_input_text}")
    print(f"\nFinal Extraction (after final replacement):")
    print(f"\n---Ground Truth:")
    for field in CATEGORIES:
        print(f"{field}: {final_true.get(field, set())}")
    print(f"\n---Final Prediction:")
    for field in CATEGORIES:
        print(f"{field}: {final_pred.get(field, set())}")

    for field in set(final_pred.keys()) | set(final_true.keys()):
        pred_set = final_pred.get(field, set())
        true_set = final_true.get(field, set())

        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        fp_unk += sum("<unk>" in x for x in (pred_set - true_set))
        fn_unk += sum("<unk>" in x for x in (true_set - pred_set))

# === Final Evaluation ===
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("\nFinal Evaluation:")
print(f"TP: {tp} | FP: {fp} | FN: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

