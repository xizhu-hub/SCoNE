import json
import re
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn.functional import softmax

# === Config ===
model_path = "saved/T5_flo_masking_s"
test_file = "datasets/florilege_T5_diffusion_test.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATION = 1
CONFIDENCE_THRESHOLD = 0.9
CATEGORIES = ["Taxon", "Phenotype", "Molecule", "Habitat", "Use"]

# === Model load ===
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()

# === Data load ===
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)
step1_data = [item for item in test_data if item.get("step") == 1]

# === Functions ===
def extract_chunks(text):
    tag_chunks = {}
    matches = re.findall(r'(<extra_id_\d+>)([^<]*)', text)
    for tag, content in matches:
        content = content.strip()
        if content.lower() == "no entity":
            tag_chunks[tag] = []
        else:
            tag_chunks[tag] = [e.strip() for e in content.split("@@") if e.strip()]
    return tag_chunks

def extract_extra_ids(text):
    return re.findall(r"<extra_id_\d+>", text)

def renumber_extra_ids(text):
    tags = sorted(set(re.findall(r"<extra_id_\d+>", text)), key=lambda x: int(x.split("_")[-1].replace(">", "")))
    tag_map = {tag: f"<extra_id_{i}>" for i, tag in enumerate(tags)}
    for old, new in tag_map.items():
        text = text.replace(old, new)
    return text, tag_map

def remap_pred_text(pred_text, tag_map):
    reverse_map = {v: k for k, v in tag_map.items()}
    for new, old in reverse_map.items():
        pred_text = pred_text.replace(new, old)
    return pred_text

def revise_input_simple(input_text, pred_chunks, token_probs, starting_id=0):
    new_input = input_text
    used_tags = extract_extra_ids(input_text)
    token_low_conf = {tok for tok, prob in token_probs if prob < CONFIDENCE_THRESHOLD}
    low_conf_entities = []
    new_id = starting_id

    for tag in used_tags:
        ents = pred_chunks.get(tag, [])
        new_values = []

        for ent in ents:
            tokens = tokenizer.tokenize(ent)
            if any(tok in token_low_conf for tok in tokens):
                tag1 = f"<extra_id_{new_id}>"
                tag2 = f"<extra_id_{new_id + 1}>"
                new_values.append(f"{tag1} @@ {tag2}")
                low_conf_entities.append((tag, ent))
                new_id += 2
            else:
                new_values.append(ent)

        replacement = " @@ ".join(new_values) if new_values else "no entity"
        new_input = new_input.replace(tag, replacement)

    return new_input, new_id, low_conf_entities

def fill_back_predictions(input_text, pred_text):
    chunks = extract_chunks(pred_text)
    for tag, ents in chunks.items():
        input_text = input_text.replace(tag, ' @@ '.join(ents) if ents else "no entity")
    return input_text


def extract_final_entities(text):
    match = re.search(r"Recognition Results:(.*)", text, re.DOTALL)
    if not match:
        return {}
    rec_text = match.group(1).strip()

    result = {}
    for cat in CATEGORIES:
        pattern = rf"{cat}:\s*([^:]+?)(?=(?: {'|'.join(CATEGORIES)}:|$))"
        m = re.search(pattern, rec_text, re.DOTALL)
        if m:
            val = m.group(1).strip()
            if val.lower() == "no entity":
                result[cat] = set()
            else:
                result[cat] = set(e.strip() for e in val.split("@@") if e.strip())
        else:
            result[cat] = set()
    return result

def extract_target(text):
    chunks = extract_chunks(text)
    return {CATEGORIES[i]: set(chunks.get(f"<extra_id_{i}>", [])) for i in range(len(CATEGORIES))}

# === Measure ===
tp = fp = fn = fp_unk = fn_unk = 0

# === Main  ===
for idx, sample in enumerate(step1_data):
    print("=" * 60)
    print(f"Sample {idx + 1}")
    input_text = sample["input_text"]
    target_text = sample["target_text"]
    pred_text = ""
    next_extra_id = 10
    input_text_before_prediction = input_text

    for it in range(ITERATION):
        print(f"\n Iration {it + 1}")
        print(f"input_text for this iteration:\n{input_text}")
        input_text_before_prediction = input_text  

        renumbered_input, tag_map = renumber_extra_ids(input_text)
        input_ids = tokenizer(renumbered_input, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=256, return_dict_in_generate=True, output_scores=True)

        generated_ids = outputs.sequences[0]
        scores = outputs.scores
        tokens = tokenizer.convert_ids_to_tokens(generated_ids)

        token_probs = [
            (tokens[i + 1], softmax(logit, dim=-1)[0, token_id].item())
            for i, (token_id, logit) in enumerate(zip(generated_ids[1:], scores))
        ]

        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=False).replace("</s>", "").replace("<pad>", "").strip()
        pred_text = remap_pred_text(pred_text, tag_map)
        pred_chunks = extract_chunks(pred_text)
        print(f"pred_text: {pred_text}")

        new_input_text, next_extra_id, low_conf = revise_input_simple(input_text, pred_chunks, token_probs, next_extra_id)
        print(f"new_input_text:\n{new_input_text}")
        if low_conf:
            print("Low-confidence entities:")
            for tag, ent in low_conf:
                print(f"  - {tag}: {ent}")

        if new_input_text.strip() == input_text.strip():
            print("No change detected. Early stopping.")
            break
        else:
            input_text = new_input_text

    #Final Prediction
    final_filled = fill_back_predictions(input_text_before_prediction, pred_text)
    final_pred = extract_final_entities(final_filled)
    final_true = extract_target(target_text)

    print(f"\nFinal filled input_text (for eval):\n{final_filled}")
    print("\nFinal Extraction:")
    for cat in CATEGORIES:
        print(f"{cat}: pred={final_pred.get(cat, set())} | true={final_true.get(cat, set())}")

    for cat in CATEGORIES:
        p = final_pred.get(cat, set())
        t = final_true.get(cat, set())
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
        fp_unk += sum("<unk>" in x for x in (p - t))
        fn_unk += sum("<unk>" in x for x in (t - p))

# === Results ===
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("\n Final Evaluation:")
print(f"TP: {tp} | FP: {fp} | FN: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

