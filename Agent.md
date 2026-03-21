## Project goal

Migrate MiniMind training stack to Qwen-based stack for memory summarization.

## Must do
- Replace MiniMind backbone with Qwen HF model.
- Keep training flow: SFT -> RLAIF -> long-context extension.
- Preserve existing dataset format where possible, but adapt chat template to Qwen.
- Run tests after changes.

## Must not do
- Do not change unrelated evaluation scripts unless required.
- Do not remove current CLI entrypoints unless replaced.

## Acceptance criteria
- SFT training runs on Qwen.
- RLAIF scripts run with Qwen wrapper.
- Long-context config works with Qwen config path.
- Repo passes smoke test.

## Commands
- install: ...
- train_sft: ...
- test: ...