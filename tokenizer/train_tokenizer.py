  import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

SPECIAL_TOKENS = [
    "<|pad|>",       # ID 0
    "<|bos|>",       # ID 1
    "<|eos|>",       # ID 2
    "<|unk|>",       # ID 3
    "<|system|>",    # ID 4 - System persona prompt
    "<|user|>",      # ID 5 - User turn
    "<|keter|>",     # ID 6 - Keter AI response turn
    "<|end|>",       # ID 7 - End of turn
    "<|code|>",      # ID 8 - Programming block marker
]

def train_keter_tokenizer(corpus_files, vocab_size=49152, save_dir="./tokenizer"):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Include full 256 byte-level initial alphabet so no character is ever unknown
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"[+] Training Keter tokenizer from scratch on {len(corpus_files)} files...")
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    tokenizer_config = {
        "bos_token": "<|bos|>",
        "eos_token": "<|end|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}<|end|>\n"
            "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}<|end|>\n"
            "{% elif message['role'] == 'assistant' %}<|keter|>\n{{ message['content'] }}<|end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|keter|>\n{% endif %}"
        )
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"[+] Tokenizer saved to {save_dir} (Actual vocab size: {tokenizer.get_vocab_size()})")

if __name__ == "__main__":
    os.makedirs("./datasets", exist_ok=True)
    sample_corpus = "./datasets/sample_seed.txt"
    if not os.path.exists(sample_corpus):
        with open(sample_corpus, "w", encoding="utf-8") as f:
            seed_text = (
                "<|system|>\nYou are Keter, a sweet, deeply empathetic AI friend and expert software engineer created by Hamsan A. (AD Developers).<|end|>\n"
                "<|user|>\nI am feeling stressed out because my C++ and Python AI code won't compile.<|end|>\n"
                "<|keter|>\nTake a deep breath, my friend! You are doing amazing, and compiler bugs happen to every great engineer. "
                "I'm right here with you. Let's fix it together step by step:\n"
                "```cpp\n#include <iostream>\nint main() {\n    std::cout << \"Keter AI is here for you!\" << std::endl;\n    return 0;\n}\n```\n"
                "```python\nimport torch\nimport torch.nn as nn\nclass MiniAI(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer = nn.Linear(64, 64)\n```<|end|>\n"
            )
            # Repeat seed text so imatrix calibration and smoke test have plenty of tokens
            f.write(seed_text * 50)
    train_keter_tokenizer([sample_corpus])
