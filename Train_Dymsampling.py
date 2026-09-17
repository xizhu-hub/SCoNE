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
MAX_EPOCHS = 20
BATCH_SIZE = 4
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 160
MODEL_NAME = "t5-large"
OUTPUT_DIR = "./saved"
WARMUP_EPOCHS = int(0.1 * MAX_EPOCHS)       # warm-up phrase
FINETUNE_EPOCHS = int(0.2 * MAX_EPOCHS)     # fine-tune phrase

# ==================== Dynamic Sampling ====================
class DynamicStepSamplerWrapper:
    def __init__(self, dataset, epoch_thresholds=(WARMUP_EPOCHS, MAX_EPOCHS - FINETUNE_EPOCHS)):
        self.full_data = dataset
        self.epoch_thresholds = epoch_thresholds  # (warmup_end, finetune_start)
        self.loss_track = {"step0": 1.0, "step1": 1.0}
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_losses(self, step0_loss, step1_loss):
        self.loss_track["step0"] = step0_loss
        self.loss_track["step1"] = step1_loss

    def get_filtered_data(self):
        if self.current_epoch < self.epoch_thresholds[0]:
            p_step0 = 0.9
        elif self.current_epoch >= self.epoch_thresholds[1]:
            p_step0 = 0.0
        else:
            l0 = self.loss_track["step0"]
            l1 = self.loss_track["step1"]
            total = l0 + l1
            p_step0 = l0 / total if total > 0 else 0.5

        p_step1 = 1.0 - p_step0
        samples_step0 = [x for x in self.full_data if x["step"] == 0]
        samples_step1 = [x for x in self.full_data if x["step"] == 1]

        n_total = len(self.full_data)
        n0 = int(n_total * p_step0)
        n1 = n_total - n0

        selected = random.sample(samples_step0, min(n0, len(samples_step0))) + \
                   random.sample(samples_step1, min(n1, len(samples_step1)))

        random.shuffle(selected)
        return Dataset.from_list(selected)

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

# ==================== Main ====================
def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    with open("datasets/florilege_T5_diffusion_train_subset_50.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    eval_data = dataset["test"]

    global tokenizer  
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    eval_data = eval_data.map(lambda x: preprocess(x, tokenizer), remove_columns=eval_data.column_names)

    eval_sampler = DistributedSampler(eval_data, num_replicas=world_size, rank=rank, shuffle=False)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, sampler=eval_sampler, collate_fn=collate_fn_with_step)

    dynamic_sampler = DynamicStepSamplerWrapper(dataset=dataset["train"],
                                                epoch_thresholds=(WARMUP_EPOCHS, MAX_EPOCHS - FINETUNE_EPOCHS))

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(MAX_EPOCHS):
        dynamic_sampler.set_epoch(epoch)
        raw_filtered_data = dynamic_sampler.get_filtered_data()
        train_data = raw_filtered_data.map(lambda x: preprocess(x, tokenizer), remove_columns=raw_filtered_data.column_names)

        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn_with_step)

        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        step0_loss, step1_loss = 0.0, 0.0
        step0_count, step1_count = 0, 0

        for batch in train_loader:
            steps = batch["step"].tolist()
            batch = {k: v.to(rank) for k, v in batch.items() if k != "step"}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            for step_type in steps:
                if step_type == 0:
                    step0_loss += loss.item()
                    step0_count += 1
                else:
                    step1_loss += loss.item()
                    step1_count += 1

        dynamic_sampler.set_losses(
            step0_loss / max(1, step0_count),
            step1_loss / max(1, step1_count)
        )

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

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
        model.module.save_pretrained(f"{OUTPUT_DIR}/T5_flo_masking_50_s")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/T5_flo_masking_50_s")
        print(f"Saved to：{OUTPUT_DIR}/T5_flo_masking_1500_s")

    dist.destroy_process_group()

# ==================== Multi Processing ====================
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()

