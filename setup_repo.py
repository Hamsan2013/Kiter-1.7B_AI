  # setup_repo.py
import os

REPO_NAME = "Keter-1.7B"

DIRECTORIES = [
    "configs",
    "architecture",
    "tokenizer",
    "datasets",
    "training/pretraining",
    "training/evaluation",
    "training/checkpoints",
    "quantization",
    "conversion",
    "gguf",
    "benchmarks",
    "model_card",
    "scripts",
]

GITIGNORE = """# Exclude large checkpoints, datasets, and GGUF binaries from Git source control
training/checkpoints/
gguf/*.gguf
datasets/*.bin
datasets/*.jsonl
*.safetensors
*.pt
*.imatrix
__pycache__/
.venv/
"""

README_MD = """# Keter 1.7B
**Your intelligence. Your device. Your control.**

* **Developer:** Hamsan A.
* **Organization:** AD Developers
* **Model Type:** Autoregressive decoder-only language model (trained from randomly initialized weights)
* **Parameters:** ~1.7 Billion
* **Primary Distribution:** GGUF (FP16 Master, 4-bit, 2-bit, and 1-bit-class IQ1_S / IQ1_M / TQ1_0 targets)
"""

def init_repo():
    os.makedirs(REPO_NAME, exist_ok=True)
    for folder in DIRECTORIES:
        path = os.path.join(REPO_NAME, folder)
        os.makedirs(path, exist_ok=True)
        gitkeep = os.path.join(path, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "w").close()

    with open(os.path.join(REPO_NAME, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(GITIGNORE)

    with open(os.path.join(REPO_NAME, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_MD)

    for doc in ["LICENSE", "MODEL_CARD.md"]:
        doc_path = os.path.join(REPO_NAME, doc)
        if not os.path.exists(doc_path):
            open(doc_path, "w").close()

    print(f"[+] Repository structure for Hamsan2013/{REPO_NAME} initialized successfully.")

if __name__ == "__main__":
    init_repo()
