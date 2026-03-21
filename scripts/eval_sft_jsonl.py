import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model


REQUIRED_STRUCTURED_FIELDS = ["summary", "keywords", "statement_type", "tense", "sentiment"]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize_text(text):
    text = str(text).strip()
    if not text:
        return []
    if jieba is not None:
        tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
        if tokens:
            return tokens
    return list(text)


def ngram_counts(tokens, n):
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def rouge_n_f1(pred, ref, n=1):
    pred_tokens = tokenize_text(pred)
    ref_tokens = tokenize_text(ref)
    pred_counts = ngram_counts(pred_tokens, n)
    ref_counts = ngram_counts(ref_tokens, n)
    overlap = 0
    for gram, pred_count in pred_counts.items():
        overlap += min(pred_count, ref_counts.get(gram, 0))

    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(x, y):
    if not x or not y:
        return 0
    dp = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        prev = 0
        for j in range(1, len(y) + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(pred, ref):
    pred_tokens = tokenize_text(pred)
    ref_tokens = tokenize_text(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def maybe_parse_json(text):
    try:
        value = json.loads(text)
        return value, True
    except Exception:
        return None, False


def field_is_filled(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def normalize_summary_text(raw_text, output_mode):
    if output_mode == "plain":
        return str(raw_text).strip()

    parsed, ok = maybe_parse_json(raw_text)
    if ok and isinstance(parsed, dict):
        return str(parsed.get("summary", "") or "").strip()
    return ""


def build_prompt_and_target(sample):
    conversations = sample["conversations"]
    if not conversations:
        raise ValueError("Sample conversations is empty")
    if conversations[-1].get("role") != "assistant":
        raise ValueError("Last conversation turn must be assistant")
    prompt_messages = conversations[:-1]
    target = str(conversations[-1].get("content", "") or "").strip()
    return prompt_messages, target


def build_generation_text(tokenizer, prompt_messages):
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_eval_model(args):
    if args.model_source == "minimind":
        lm_config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        )
    else:
        lm_config = type("QwenCompatConfig", (), {"use_moe": False, "hidden_size": args.hidden_size})()

    model, tokenizer = init_model(
        lm_config,
        from_weight="none",
        device=args.device,
        model_source=args.model_source,
        hf_model_path=args.hf_model_path,
        ckpt_tag=args.ckpt_tag,
        rope_scaling_type=None if args.rope_scaling_type == "none" else args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
    )
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer


def evaluate_checkpoint(args):
    model, tokenizer = load_eval_model(args)
    samples = read_jsonl(args.data_path)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = Path(args.checkpoint_path).stem
    predictions_path = output_dir / f"{checkpoint_stem}_predictions.jsonl"
    metrics_path = output_dir / f"{checkpoint_stem}_metrics.json"

    rouge1_scores = []
    rougel_scores = []
    output_lengths = []
    json_parse_hits = 0
    field_complete_hits = 0
    rows = []

    for idx, sample in enumerate(samples):
        prompt_messages, target = build_prompt_and_target(sample)
        prompt_text = build_generation_text(tokenizer, prompt_messages)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_input_length).to(args.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        pred_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        output_lengths.append(len(tokenize_text(pred_text)))

        pred_summary = normalize_summary_text(pred_text, args.output_mode)
        ref_summary = normalize_summary_text(target, args.output_mode)
        rouge1_scores.append(rouge_n_f1(pred_summary, ref_summary, n=1))
        rougel_scores.append(rouge_l_f1(pred_summary, ref_summary))

        parsed_ok = None
        complete_ok = None
        parsed_pred = None
        if args.output_mode == "structured":
            parsed_pred, parsed_ok = maybe_parse_json(pred_text)
            parsed_ok = parsed_ok and isinstance(parsed_pred, dict)
            json_parse_hits += int(parsed_ok)
            if parsed_ok:
                complete_ok = all(field_is_filled(parsed_pred.get(field)) for field in REQUIRED_STRUCTURED_FIELDS)
                field_complete_hits += int(complete_ok)
            else:
                complete_ok = False

        row = {
            "index": idx,
            "prompt_messages": prompt_messages,
            "reference": target,
            "prediction": pred_text,
            "rouge_1": rouge1_scores[-1],
            "rouge_l": rougel_scores[-1],
        }
        if args.output_mode == "structured":
            row["json_parse_ok"] = bool(parsed_ok)
            row["required_fields_complete"] = bool(complete_ok)
        rows.append(row)

    metrics = {
        "checkpoint_path": str(args.checkpoint_path),
        "data_path": str(args.data_path),
        "num_samples": len(samples),
        "output_mode": args.output_mode,
        "rouge_1": sum(rouge1_scores) / max(len(rouge1_scores), 1),
        "rouge_l": sum(rougel_scores) / max(len(rougel_scores), 1),
        "avg_output_length": sum(output_lengths) / max(len(output_lengths), 1),
    }
    if args.output_mode == "structured":
        total = max(len(samples), 1)
        metrics["json_parse_rate"] = json_parse_hits / total
        metrics["required_field_complete_rate"] = field_complete_hits / total

    with predictions_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoint on jsonl validation set")
    parser.add_argument("--checkpoint_path", required=True, help="Checkpoint .pth path")
    parser.add_argument("--data_path", required=True, help="Validation jsonl path")
    parser.add_argument("--output_mode", choices=["plain", "structured"], default="plain", help="Target output mode")
    parser.add_argument("--output_dir", default="", help="Directory to save predictions and metrics")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit evaluation samples; 0 means all")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Max prompt token length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Evaluation device")

    parser.add_argument("--model_source", choices=["qwen", "minimind"], default="qwen", help="Model source")
    parser.add_argument("--hf_model_path", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model path for base weights/tokenizer")
    parser.add_argument("--ckpt_tag", default="qwen15", help="Checkpoint tag; only used for config compatibility")
    parser.add_argument("--hidden_size", type=int, default=512, help="MiniMind hidden size compatibility")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="MiniMind layer count compatibility")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0, help="MiniMind MoE compatibility")
    parser.add_argument("--rope_scaling_type", default="none", help="Optional rope scaling type")
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0, help="Optional rope scaling factor")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
