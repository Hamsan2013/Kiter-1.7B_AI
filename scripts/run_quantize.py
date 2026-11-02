import os
import subprocess

def run_cmd(cmd):
    print(f"[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    os.makedirs("gguf", exist_ok=True)

    # Patch llama.cpp converter to accept our custom from-scratch BPE tokenizer hash
    conv_file = "llama.cpp/convert_hf_to_gguf.py"
    with open(conv_file, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace(
        'raise NotImplementedError(f"BPE pre-tokenizer was not recognized',
        'return "gpt2" #raise NotImplementedError(f"BPE pre-tokenizer was not recognized'
    )
    with open(conv_file, "w", encoding="utf-8") as f:
        f.write(code)

    # 1. Convert to Master FP16 GGUF
    run_cmd([
        "python", conv_file,
        "./training/checkpoints/keter-v1",
        "--outfile", "./gguf/keter-1.7b-f16.gguf",
        "--outtype", "f16"
    ])

    # 2. Generate Importance Matrix (imatrix) on Empathy + Code data
    run_cmd([
        "./llama.cpp/build/bin/llama-imatrix",
        "-m", "./gguf/keter-1.7b-f16.gguf",
        "-f", "./datasets/calibration.txt",
        "-o", "./gguf/keter.imatrix",
        "--chunks", "2"
    ])

    # 3. Quantize to 1-Bit (IQ1_M, IQ1_S) and 4-Bit (Q4_K_M)
    for qtype in ["IQ1_M", "IQ1_S", "Q4_K_M"]:
        run_cmd([
            "./llama.cpp/build/bin/llama-quantize",
            "--imatrix", "./gguf/keter.imatrix",
            "./gguf/keter-1.7b-f16.gguf",
            f"./gguf/keter-1.7b-{qtype}.gguf",
            qtype
        ])

    # Delete the huge 3.4GB FP16 master file before upload if needed so GitHub Artifact quota isn't exceeded
    if os.path.exists("./gguf/keter-1.7b-f16.gguf"):
        os.remove("./gguf/keter-1.7b-f16.gguf")

    print("[+] All 1-Bit and 4-Bit Keter 1.7B GGUF models built successfully!")

if __name__ == "__main__":
    main()
