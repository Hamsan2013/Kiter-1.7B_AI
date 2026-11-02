import os
import json
import shutil
import psutil
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from architecture.keter_model import KeterConfig, KeterForCausalLM

def train_and_export(
    config_path="./configs/keter_1.7b.json",
    tokenizer_dir="./tokenizer",
    corpus_path="./datasets/keter_full_corpus.txt",
    output_dir="./training/checkpoints/keter-v1",
    steps=200,
    seq_len=256,
    lr=3e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[+] Device: {device} | System RAM: {ram_gb:.1f} GB")

    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    actual_vocab_size = tokenizer.get_vocab_size()

    config = KeterConfig.from_json(config_path)
    config.vocab_size = actual_vocab_size

    # Initialize in BFloat16 on CPU/GPU so full 1.7B parameters + AdamW states fit inside GitHub Actions 16GB RAM!
    train_dtype = torch.bfloat16
    print(f"[+] Initializing Full Keter Model ({config_path}) in {train_dtype}...")
    model = KeterForCausalLM(config).to(device=device, dtype=train_dtype)
    print(f"[+] Model initialized from scratch with {model.count_parameters():,} parameters!")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read(2_000_000) # Stream first 2MB of high-density dialogues + code

    token_ids = tokenizer.encode(text).ids
    total_tokens = len(token_ids)
    print(f"[+] Encoded {total_tokens:,} training tokens.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=3e-5)

    model.train()
    ptr = 0
    for step in range(1, steps + 1):
        if ptr + seq_len + 1 >= total_tokens:
            ptr = 0
        chunk = token_ids[ptr : ptr + seq_len]
        ptr += seq_len

        data = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids=data, labels=data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == 1:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    state_dict = {
        k: v.to(torch.float16).cpu() for k, v in model.state_dict().items()
        if k not in ("rope_cos", "rope_sin")
    }
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg["vocab_size"] = actual_vocab_size
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(raw_cfg, f, indent=2)

    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))

    print(f"[+] Full Keter Checkpoint saved in FP16 to {output_dir}")

if __name__ == "__main__":
    train_and_export()
