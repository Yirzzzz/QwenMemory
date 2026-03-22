import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer


THRESHOLDS = [512, 1024, 1536, 2048]


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_rendered_text(tokenizer, sample):
    conversations = sample["conversations"]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
    )


def compute_token_length(tokenizer, sample):
    text = build_rendered_text(tokenizer, sample)
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(input_ids)


def open_bucket_files(output_dir):
    handles = {}
    for threshold in THRESHOLDS:
        path = output_dir / f"sft_le_{threshold}.jsonl"
        handles[f"le_{threshold}"] = path.open("w", encoding="utf-8")
    gt_path = output_dir / "sft_gt_2048.jsonl"
    handles["gt_2048"] = gt_path.open("w", encoding="utf-8")
    return handles


def close_bucket_files(handles):
    for handle in handles.values():
        handle.close()


def main():
    parser = argparse.ArgumentParser(description="Split jsonl dataset into token-length buckets")
    parser.add_argument("--input_path", required=True, help="Input jsonl path")
    parser.add_argument("--tokenizer_path", required=True, help="Local tokenizer path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    bucket_counts = {f"le_{threshold}": 0 for threshold in THRESHOLDS}
    bucket_counts["gt_2048"] = 0
    lengths = []
    total_count = 0
    skipped_count = 0

    handles = open_bucket_files(output_dir)
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if "conversations" not in sample:
                        raise KeyError("Missing conversations field")
                    length = compute_token_length(tokenizer, sample)
                except Exception as exc:
                    skipped_count += 1
                    print(f"Warning: skipped line {line_no}: {exc}", file=sys.stderr)
                    continue

                lengths.append(length)
                total_count += 1

                if length > 2048:
                    handles["gt_2048"].write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                    bucket_counts["gt_2048"] += 1
                    continue

                for threshold in THRESHOLDS:
                    if length <= threshold:
                        handles[f"le_{threshold}"].write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                        bucket_counts[f"le_{threshold}"] += 1
    finally:
        close_bucket_files(handles)

    sorted_lengths = sorted(lengths)
    avg_length = sum(lengths) / total_count if total_count else 0

    stats = {
        "input_path": str(input_path),
        "tokenizer_path": str(args.tokenizer_path),
        "total_samples": total_count,
        "skipped_samples": skipped_count,
        "average_length": avg_length,
        "median_length": percentile(sorted_lengths, 0.5),
        "p90_length": percentile(sorted_lengths, 0.9),
        "p95_length": percentile(sorted_lengths, 0.95),
        "p99_length": percentile(sorted_lengths, 0.99),
        "max_length": max(sorted_lengths) if sorted_lengths else 0,
        "bucket_counts": {},
    }

    for bucket_name, count in bucket_counts.items():
        stats["bucket_counts"][bucket_name] = {
            "count": count,
            "ratio": (count / total_count) if total_count else 0,
        }

    stats_path = output_dir / "length_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Stats saved to {stats_path}")
    for threshold in THRESHOLDS:
        print(f"Bucket file: {output_dir / f'sft_le_{threshold}.jsonl'}")
    print(f"Bucket file: {output_dir / 'sft_gt_2048.jsonl'}")


if __name__ == "__main__":
    main()
