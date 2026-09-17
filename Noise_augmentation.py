import json
import random
from copy import deepcopy

random.seed(42)

# File path
input_file = "processed_florilege_test_T5.json"
output_file = "florilege_T5_diffusion_test.json"

# Load data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Setup entity types
entity_types = ["Taxon", "Phenotype", "Molecule", "Habitat", "Use"]

def parse_ground_truth(gt_text):
    
    entity_dict = {}
    for etype in entity_types:
        prefix = etype + ":"
        start = gt_text.find(prefix)
        if start == -1:
            continue
        
        end = min([gt_text.find(e + ":", start + 1) for e in entity_types if gt_text.find(e + ":", start + 1) != -1] + [len(gt_text)])
        entity_str = gt_text[start + len(prefix): end].strip()
        if entity_str != "no entity":
            entity_dict[etype] = [e.strip() for e in entity_str.split("@@") if e.strip()]
    return entity_dict

def build_augmented_t5_format(entry):
    base_input = entry["input_text"]
    entity_dict = parse_ground_truth(entry["ground_truth"])
    
    
    if not any(entity_dict.values()):
        return []
    
    all_entities = []
    for etype in entity_dict:
        for i, ent in enumerate(entity_dict[etype]):
            all_entities.append((etype, i, ent))  # (type, position, entiy)

    # randomize masking order
    random.shuffle(all_entities)


    masked_dict = deepcopy(entity_dict)

    samples = []

    # step=0 -> partially noisy

    for step_i in range(len(all_entities)):
        etype, idx, _ = all_entities[step_i]
        masked_dict[etype][idx] = f"<extra_id_{step_i}>"

        recognition_result = " ".join([
            f"{t}: {' @@ '.join(masked_dict[t])}" if t in masked_dict else f"{t}: no entity"
            for t in entity_types
        ])
        input_text = base_input + " Recognition Results: " + recognition_result

        output_text = " ".join([
            f"<extra_id_{j}> {all_entities[j][2]}" for j in range(step_i + 1)
        ])

        samples.append({
            "step": 0,
            "input_text": input_text,
            "target_text": output_text
        })
    
    # step=1 -> Fully noisy
    final_input_text = base_input + " Recognition Results: " + " ".join(
        [f"{etype}: <extra_id_{i}>" for i, etype in enumerate(entity_types)]
    )

    final_output_text = " ".join([
        f"<extra_id_{i}> {' @@ '.join(entity_dict.get(etype, [])) or 'no entity'}"
        for i, etype in enumerate(entity_types)
    ])

    samples.append({
        "step": 1,
        "input_text": final_input_text,
        "target_text": final_output_text
    })

    return samples

#Add to final data pairs
final_data = []
for entry in data:
    final_data.extend(build_augmented_t5_format(entry))

# Save data
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"Run noise augmentation successfully, {len(final_data)} samples are saved to：{output_file}")

