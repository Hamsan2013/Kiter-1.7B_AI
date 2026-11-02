import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from architecture.keter_model import KeterConfig, KeterForCausalLM


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def save_training_state(path: Path, model, optimizer, scheduler, step: int, ptr: int):
    """Write a resumable state atomically. The previous run's remote artifact remains safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        path.unlink()
    state = {
        "step": step,
        "ptr": ptr,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temp)
    os.replace(temp, path)
    print(f"[checkpoint] Saved resumable state at step {step}", flush=True)


def restore_training_state(path: Path, model, optimizer, scheduler, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    for opt_state in optimizer.state.values():
        for key, value in opt_state.items():
            if torch.is_tensor(value):
                opt_state[key] = value.to(device)
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    step = int(state["step"])
    ptr = int(state.get("ptr", 0))
    del state
    print(f"[resume] Restored model, optimizer, scheduler and RNG at step {step}", flush=True)
    return step, ptr


def export_fp16(model, config_path: Path, tokenizer_dir: Path, output_dir: Path, actual_vocab_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().to(dtype=torch.float16, device="cpu").contiguous()
        for key, value in model.state_dict().items()
        if key not in ("rope_cos", "rope_sin")
    }
    save_file(state_dict, str(output_dir / "model.safetensors"))
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    raw_cfg["vocab_size"] = actual_vocab_size
    (output_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    print(f"[+] Final FP16 model exported to {output_dir}", flush=True)


def train_and_export(args):
    root = ROOT
    config_path = root / args.config
    tokenizer_dir = root / args.tokenizer_dir
    corpus_path = root / args.corpus
    output_dir = root / args.output_dir
    resume_dir = root / args.resume_dir
    checkpoint_path = resume_dir / "checkpoint.pt"
    manifest_path = resume_dir / "manifest.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    print(f"[+] Training device: {device}", flush=True)

    tokenizer_file = tokenizer_dir / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    actual_vocab_size = tokenizer.get_vocab_size()
    config = KeterConfig.from_json(str(config_path))
    config.vocab_size = actual_vocab_size

    train_dtype = torch.bfloat16
    print(f"[+] Initializing Keter ({config_path.name}) in {train_dtype}...", flush=True)
    model = KeterForCausalLM(config).to(device=device, dtype=train_dtype)
    print(f"[+] Model parameters: {model.count_parameters():,}", flush=True)

    text = corpus_path.read_text(encoding="utf-8")[:2_000_000]
    token_ids = tokenizer.encode(text).ids
    total_tokens = len(token_ids)
    if total_tokens < 2:
        raise RuntimeError("The training corpus produced fewer than two tokens.")
    print(f"[+] Encoded {total_tokens:,} tokens.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=3e-5
    )

    start_step, ptr = 0, 0
    if checkpoint_path.exists():
        start_step, ptr = restore_training_state(
            checkpoint_path, model, optimizer, scheduler, device
        )

    model.train()
    started = time.monotonic()
    last_saved_step = start_step

    for step in range(start_step + 1, args.steps + 1):
        if ptr + args.seq_len + 1 >= total_tokens:
            ptr = 0
        chunk = token_ids[ptr : ptr + args.seq_len]
        ptr += args.seq_len
        if len(chunk) < 2:
            raise RuntimeError("Not enough tokens for a training sequence.")

        data = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids=data, labels=data)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 10 == 0:
            print(
                f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )

        # Keep local checkpoints frequent; the workflow uploads the latest one at chunk end.
        if step == 1 or step % args.checkpoint_every == 0:
            save_training_state(checkpoint_path, model, optimizer, scheduler, step, ptr)
            last_saved_step = step

        if time.monotonic() - started >= args.max_seconds:
            if last_saved_step != step:
                save_training_state(checkpoint_path, model, optimizer, scheduler, step, ptr)
            atomic_json(manifest_path, {
                "complete": False,
                "step": step,
                "total_steps": args.steps,
                "continuation_count": args.continuation_count,
                "reason": "time_slice_finished",
            })
            with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a", encoding="utf-8") as out:
                out.write("complete=false\n")
            print(f"[paused] Safe checkpoint at step {step}; next Actions run will resume.", flush=True)
            return

    # Always write the final state before exporting, so export can be retried independently.
    save_training_state(checkpoint_path, model, optimizer, scheduler, args.steps, ptr)
    export_fp16(model, config_path, tokenizer_dir, output_dir, actual_vocab_size)
    atomic_json(manifest_path, {
        "complete": True,
        "step": args.steps,
        "total_steps": args.steps,
        "continuation_count": args.continuation_count,
        "reason": "training_complete",
    })
    with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a", encoding="utf-8") as out:
        out.write("complete=true\n")
    print("[+] Training completed; final FP16 weights are ready for GGUF conversion.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Keter with resumable checkpoints.")
    parser.add_argument("--config", default="configs/keter_1.7b.json")
    parser.add_argument("--tokenizer-dir", default="tokenizer")
    parser.add_argument("--corpus", default="datasets/keter_full_corpus.txt")
    parser.add_argument("--output-dir", default="training/checkpoints/keter-v1")
    parser.add_argument("--resume-dir", default="training/resume_bundle")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=12600)
    parser.add_argument("--continuation-count", type=int, default=0)
    train_and_export(parser.parse_args())
