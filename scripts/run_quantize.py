import hashlib
import json
import os
import subprocess
from pathlib import Path

MODEL_DIR = Path("training/checkpoints/keter-v1")
TOKENIZER_HASH = "5f4772b6b9d5533a7c3aa232743347e6df977878335a0cc6a776555e694ee1a9"
LLAMA_CPP = Path("llama.cpp")
GGUF_DIR = Path("gguf")


def run_cmd(cmd):
    print("[RUNNING]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def validate_assets():
    required = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "model.safetensors",
        MODEL_DIR / "tokenizer.json",
        MODEL_DIR / "tokenizer_config.json",
        Path("datasets/calibration.txt"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The downloaded training artifact is incomplete; missing: " + ", ".join(missing)
        )

    with (MODEL_DIR / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "llama":
        raise ValueError(f"Expected model_type='llama', got {config.get('model_type')!r}")

    with (MODEL_DIR / "tokenizer.json").open(encoding="utf-8") as f:
        tokenizer = json.load(f)
    pre = tokenizer.get("pre_tokenizer", {})
    if pre.get("type") != "ByteLevel":
        raise ValueError(
            "Keter tokenizer is not the expected ByteLevel BPE tokenizer; "
            f"found pre_tokenizer={pre!r}"
        )
    if tokenizer.get("model", {}).get("type") != "BPE":
        raise ValueError("Keter tokenizer.json does not contain a BPE model")

    print("[OK] FP16 model, config, ByteLevel BPE tokenizer and calibration data found.", flush=True)


def register_keter_pretokenizer():
    base_file = LLAMA_CPP / "conversion" / "base.py"
    if not base_file.is_file():
        raise FileNotFoundError(f"llama.cpp conversion module not found: {base_file}")

    source = base_file.read_text(encoding="utf-8")
    marker = "        res = None\n"
    registration = (
        marker
        + f'\n        if chkhsh == "{TOKENIZER_HASH}":\n'
        + '            # Keter uses the standard Hugging Face ByteLevel BPE pre-tokenizer.\n'
        + '            res = "gpt-2"\n'
    )
    if f'chkhsh == "{TOKENIZER_HASH}"' not in source:
        if marker not in source:
            raise RuntimeError("Could not safely locate get_vocab_base_pre() registration point.")
        source = source.replace(marker, registration, 1)
        base_file.write_text(source, encoding="utf-8")
    print(f"[OK] Registered Keter tokenizer hash in {base_file}.", flush=True)


def main():
    validate_assets()
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    register_keter_pretokenizer()

    converter = LLAMA_CPP / "convert_hf_to_gguf.py"
    run_cmd([
        "python", converter, MODEL_DIR,
        "--outfile", GGUF_DIR / "keter-1.7b-f16.gguf",
        "--outtype", "f16",
    ])

    run_cmd([
        LLAMA_CPP / "build" / "bin" / "llama-imatrix",
        "-m", GGUF_DIR / "keter-1.7b-f16.gguf",
        "-f", "datasets/calibration.txt",
        "-o", GGUF_DIR / "keter.imatrix",
        "--chunks", "2",
    ])

    for qtype in ("IQ1_M", "IQ1_S", "Q4_K_M"):
        output = GGUF_DIR / f"keter-1.7b-{qtype}.gguf"
        run_cmd([
            LLAMA_CPP / "build" / "bin" / "llama-quantize",
            "--imatrix", GGUF_DIR / "keter.imatrix",
            GGUF_DIR / "keter-1.7b-f16.gguf",
            output,
            qtype,
        ])
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Quantization did not produce a valid output: {output}")
        print(f"[OK] {output} ({output.stat().st_size:,} bytes)", flush=True)

    # The large FP16 intermediate and imatrix are not part of the deliverable bundle.
    (GGUF_DIR / "keter-1.7b-f16.gguf").unlink(missing_ok=True)
    (GGUF_DIR / "keter.imatrix").unlink(missing_ok=True)
    print("[SUCCESS] All requested Keter GGUF variants are ready.", flush=True)


if __name__ == "__main__":
    main()
