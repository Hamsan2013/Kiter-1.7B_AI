# Kiter-1.7B AI

My first from-scratch language-model project.

## Training and automatic continuation

The `Train Full Keter 1.7B GGUF` workflow trains in bounded time slices instead of trying to use the entire six-hour GitHub-hosted job limit.

- The trainer saves a resumable checkpoint containing model weights, optimizer state, scheduler state, step number, and random-number-generator states.
- Each training slice runs for up to 3.5 hours, then uploads its checkpoint and the exact tokenizer/dataset assets as an Actions artifact.
- The `Continue Keter Training` workflow checks the artifact and dispatches the next training run when training is incomplete.
- When all configured training steps finish, the workflow exports FP16 weights, converts them to GGUF, and creates IQ1_M, IQ1_S, and Q4_K_M files.
- Checkpoint artifacts are retained for 7 days; final GGUF artifacts are retained for 30 days.

### Important limitations

- Automatic continuation protects against the normal six-hour job limit by ending each slice early. A sudden runner loss before an artifact upload can still lose work since the previous uploaded checkpoint.
- The workflow uses GitHub-hosted CPU runners unless the repository is configured for another runner. A 1.46B-parameter model may be extremely slow or exceed available memory on a standard CPU runner. Check the first slice's step timing and memory before assuming the model can finish.
- The current training configuration is a 200-step prototype. Completing those steps and producing a GGUF file does not by itself establish that the model is well-trained or useful.
- If an Actions artifact is missing or expired, the workflow deliberately stops rather than silently restarting from random weights.

To start a fresh run, use **Actions → Train Full Keter 1.7B GGUF → Run workflow** and leave `resume_run_id` empty. Do not start a second fresh run while an automatic continuation chain is active.
