import argparse
import json
from pathlib import Path

from eval_sft_jsonl import evaluate_checkpoint


def score_key(metrics, primary_metric):
    return metrics.get(primary_metric, float("-inf"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple SFT checkpoints on a validation jsonl set")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoint .pth files")
    parser.add_argument("--data_path", required=True, help="Validation jsonl path")
    parser.add_argument("--checkpoint_type", choices=["full", "lora", "qlora"], default="full", help="Checkpoint type")
    parser.add_argument("--output_mode", choices=["plain", "structured"], default="plain", help="Target output mode")
    parser.add_argument("--output_dir", default="", help="Directory to save aggregated results")
    parser.add_argument("--pattern", default="*.pth", help="Glob pattern for checkpoints")
    parser.add_argument("--primary_metric", default="", help="Metric used to pick best checkpoint")
    parser.add_argument("--max_checkpoints", type=int, default=0, help="Limit number of checkpoints; 0 means all")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit evaluation samples; 0 means all")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Max prompt token length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument("--device", default="", help="Evaluation device override")

    parser.add_argument("--model_source", choices=["qwen", "minimind"], default="qwen", help="Model source")
    parser.add_argument("--hf_model_path", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model path for base weights/tokenizer")
    parser.add_argument("--ckpt_tag", default="qwen15", help="Checkpoint tag compatibility")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank when checkpoint_type=lora")
    parser.add_argument("--hidden_size", type=int, default=512, help="MiniMind hidden size compatibility")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="MiniMind layer count compatibility")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0, help="MiniMind MoE compatibility")
    parser.add_argument("--rope_scaling_type", default="none", help="Optional rope scaling type")
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0, help="Optional rope scaling factor")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = sorted(
        path for path in checkpoint_dir.glob(args.pattern)
        if path.is_file() and not path.name.endswith("_resume.pth")
    )
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[:args.max_checkpoints]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} with pattern {args.pattern}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = args.primary_metric
    if not primary_metric:
        primary_metric = "required_field_complete_rate" if args.output_mode == "structured" else "rouge_l"

    results = []
    for checkpoint_path in checkpoints:
        eval_args = argparse.Namespace(
            checkpoint_path=str(checkpoint_path),
            checkpoint_type=args.checkpoint_type,
            data_path=args.data_path,
            output_mode=args.output_mode,
            output_dir=str(output_dir),
            max_samples=args.max_samples,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            device=args.device or "cuda",
            model_source=args.model_source,
            hf_model_path=args.hf_model_path,
            ckpt_tag=args.ckpt_tag,
            lora_rank=args.lora_rank,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe,
            rope_scaling_type=args.rope_scaling_type,
            rope_scaling_factor=args.rope_scaling_factor,
        )
        if not args.device:
            import torch
            eval_args.device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = evaluate_checkpoint(eval_args)
        results.append(metrics)

    best_metrics = max(results, key=lambda item: score_key(item, primary_metric))
    summary = {
        "primary_metric": primary_metric,
        "best_checkpoint": best_metrics["checkpoint_path"],
        "best_score": best_metrics.get(primary_metric),
        "results": results,
    }

    summary_path = output_dir / "checkpoint_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    headers = ["checkpoint", "rouge_1", "rouge_l", "avg_output_length"]
    if args.output_mode == "structured":
        headers.extend(["json_parse_rate", "required_field_complete_rate"])

    print("\t".join(headers))
    for item in results:
        row = [
            Path(item["checkpoint_path"]).name,
            f'{item.get("rouge_1", 0.0):.4f}',
            f'{item.get("rouge_l", 0.0):.4f}',
            f'{item.get("avg_output_length", 0.0):.2f}',
        ]
        if args.output_mode == "structured":
            row.extend([
                f'{item.get("json_parse_rate", 0.0):.4f}',
                f'{item.get("required_field_complete_rate", 0.0):.4f}',
            ])
        print("\t".join(row))

    print(f"Best checkpoint by {primary_metric}: {best_metrics['checkpoint_path']} ({best_metrics.get(primary_metric):.4f})")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
