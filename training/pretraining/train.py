  import os
import json
import shutil
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from architecture.keter_model import KeterConfig, KeterForCausalLM

def train_and_export(
    config_path="./configs/keter_prototype.json",
    tokenizer_dir="./tokenizer",
    corpus_path="./datasets/sample_seed.txt",
    output_dir="./training/checkpoints/keter-v1",
    steps=60,
    lr=5e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running on device: {device}")

    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    actual_vocab_size = tokenizer.get_vocab_size()

    config = KeterConfig.from_json(config_path)
    config.vocab_size = actual_vocab_size

    model = KeterForCausalLM(config).to(device)
    print(f"[+] Initialized Keter model from scratch with {model.count_parameters():,} parameters.")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text).ids
    data = torch.tensor(token_ids[:512], dtype=torch.long, device=device).unsqueeze(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        _, loss = model(input_ids=data, labels=data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == 1:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    state_dict = {
        k: v.half().cpu() for k, v in model.state_dict().items()
        if k not in ("rope_cos", "rope_sin")
    }
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # Write updated config.json with exact vocab_size
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg["vocab_size"] = actual_vocab_size
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(raw_cfg, f, indent=2)

    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))

    print(f"[+] Checkpoint saved in FP16 to {output_dir}")

if __name__ == "__main__":
    train_and_export()
