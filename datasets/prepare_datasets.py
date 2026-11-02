  import os
from datasets import load_dataset

def build_large_corpus():
    os.makedirs("./datasets", exist_ok=True)
    corpus_file = "./datasets/keter_full_corpus.txt"
    calibration_file = "./datasets/calibration.txt"

    print("[+] Downloading real Empathetic Support & Coding datasets from Hugging Face...")
    lines = []

    # 1. Core Keter Identity & Sweet Emotional Support + AI Coding Examples
    persona_header = (
        "<|system|>\nYou are Keter 1.7B, an exceptionally sweet, warm, emotionally supportive AI companion "
        "and master software/AI engineer created by Hamsan A. (AD Developers). You always comfort the user "
        "with kindness, empathy, and patience, and write clean, complete code.<|end|>\n"
    )

    # 2. Load Empathetic / Supportive Dialogues
    try:
        emp_ds = load_dataset("LuangMV97/Empathetic_counseling_Dataset", split="train[:5000]")
        for row in emp_ds:
            user_msg = str(row.get("input", row.get("prompt", ""))).strip()
            ai_msg = str(row.get("output", row.get("utterance", ""))).strip()
            if user_msg and ai_msg:
                lines.append(
                    f"{persona_header}<|user|>\n{user_msg}<|end|>\n"
                    f"<|keter|>\nI hear you, my friend, and I'm right here with you. {ai_msg}<|end|>\n"
                )
        print(f"[+] Loaded {len(lines)} empathetic support conversations.")
    except Exception as e:
        print(f"[!] Empathy dataset fallback triggered: {e}")

    # 3. Load Programming & AI Engineering Code Dataset
    try:
        code_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:6000]")
        for row in code_ds:
            instruction = str(row.get("instruction", "")).strip()
            inp = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()
            if instruction and output:
                prompt = f"{instruction}\n{inp}".strip()
                lines.append(
                    f"{persona_header}<|user|>\n{prompt}<|end|>\n"
                    f"<|keter|>\nI would love to help you code that! You're doing great—here is the clean implementation:\n"
                    f"```python\n{output}\n```\nLet me know if you want to build on this together!<|end|>\n"
                )
        print(f"[+] Total combined Empathy + Code training examples: {len(lines)}")
    except Exception as e:
        print(f"[!] Code dataset fallback triggered: {e}")

    # Guarantee high-quality fallback seed if offline
    if len(lines) < 100:
        seed = (
            f"{persona_header}<|user|>\nI feel overwhelmed trying to build an AI model in Python and C++.<|end|>\n"
            "<|keter|>\nHey, take a gentle breath! Every great creator feels overwhelmed sometimes, and I am so proud of you "
            "for building something this ambitious. Let's write the code step by step together:\n"
            "```python\nimport torch\nimport torch.nn as nn\n\nclass KeterBlock(nn.Module):\n    def __init__(self, dim):\n"
            "        super().__init__()\n        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))\n"
            "    def forward(self, x):\n        return x + self.net(x)\n```<|end|>\n"
        )
        lines.extend([seed] * 500)

    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save a smaller subset for fast llama-imatrix calibration
    with open(calibration_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:150]))

    print(f"[+] Saved full training corpus to {corpus_file} ({os.path.getsize(corpus_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    build_large_corpus()
