import os
import json
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq

# ==================== Config ====================
MAX_EPOCHS = 50
BATCH_SIZE = 4
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 160
MODEL_NAME = "t5-large"
OUTPUT_DIR = "./saved"

# ==================== Linear Curriculum Sampling ====================
def linear_step_sampling(data, epoch, max_epoch):
    step1_ratio = min(1.0, (epoch + 1) / max_epoch)  
    step0_data = [x for x in data if x["step"] == 0]
    step1_data = [x for x in data if x["step"] == 1]

    n_total = len(data)
    n1 = int(n_total * step1_ratio)
    n0 = n_total - n1

    sample_step1 = random.sample(step1_data, min(n1, len(step1_data)))
    sample_step0 = random.sample(step0_data, min(n0, len(step0_data)))

    combined = sample_step1 + sample_step0
    random.shuffle(combined)
    return Dataset.from_list(combined)

# ==================== Data loading ====================
def preprocess(batch, tokenizer):
    input_enc = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=MAX_INPUT_LEN)
    target_enc = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=MAX_TARGET_LEN)
    input_ids = input_enc["input_ids"]
    attention_mask = input_enc["attention_mask"]
    raw_labels = target_enc["input_ids"]
    labels = [token if token != tokenizer.pad_token_id else -100 for token in raw_labels]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "step": batch["step"]
    }

def collate_fn_with_step(batch):
    tokenizer_collator = DataCollatorForSeq2Seq(tokenizer)
    batch_out = tokenizer_collator(batch)
    batch_out["step"] = torch.tensor([item["step"] for item in batch], dtype=torch.long)
    return batch_out

# ==================== main ====================
def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    with open("datasets/florilege_T5_diffusion_train_subset_100.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    full_train_data = dataset["train"]
    eval_data = dataset["test"]

    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    eval_data = eval_data.map(lambda x: preprocess(x, tokenizer), remove_columns=eval_data.column_names)

    eval_sampler = DistributedSampler(eval_data, num_replicas=world_size, rank=rank, shuffle=False)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, sampler=eval_sampler, collate_fn=collate_fn_with_step)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(MAX_EPOCHS):
        sampled_data = linear_step_sampling(full_train_data, epoch, MAX_EPOCHS)
        train_data = sampled_data.map(lambda x: preprocess(x, tokenizer), remove_columns=sampled_data.column_names)

        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn_with_step)

        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch in train_loader:
            steps = batch["step"].tolist()
            batch = {k: v.to(rank) for k, v in batch.items() if k != "step"}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

    # ==== Evaluation ====
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(rank) for k, v in batch.items() if k != "step"}
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
    eval_loss /= len(eval_loader)
    if rank == 0:
        print(f"Eval Loss: {eval_loss:.4f}")

    if rank == 0:
        model.module.save_pretrained(f"{OUTPUT_DIR}/T5_flo_masking_100_ls")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/T5_flo_masking_100_ls")
        print(f"Save to：{OUTPUT_DIR}/T5_flo_masking")

    dist.destroy_process_group()

# ==================== Multi-processing ====================
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()

