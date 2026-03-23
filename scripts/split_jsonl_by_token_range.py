import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer


RANGES = [
    ("sft_512_1024.jsonl", 513, 1024),
    ("sft_1024_1536.jsonl", 1025, 1536),
    ("sft_1536_2048.jsonl", 1537, 2048),
    ("sft_gt_2048.jsonl", 2049, None),
]


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


def compute_token_length(tokenizer, sample):
    text = tokenizer.apply_chat_template(
        sample["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(input_ids)


def build_bucket_name(length):
    for filename, lower, upper in RANGES:
        if upper is None and length >= lower:
            return filename
        if upper is not None and lower <= length <= upper:
            return filename
    return None


def main():
    parser = argparse.ArgumentParser(description="Split jsonl dataset into token-length ranges")
    parser.add_argument("--input_path", required=True, help="Input jsonl path")
    parser.add_argument("--tokenizer_path", required=True, help="Local tokenizer path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    handles = {
        filename: (output_dir / filename).open("w", encoding="utf-8")
        for filename, _, _ in RANGES
    }

    lengths = []
    skipped_count = 0
    total_count = 0
    bucket_counts = {filename: 0 for filename, _, _ in RANGES}
    unmatched_count = 0

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

                total_count += 1
                lengths.append(length)
                bucket_name = build_bucket_name(length)
                if bucket_name is None:
                    unmatched_count += 1
                    continue

                handles[bucket_name].write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                bucket_counts[bucket_name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    sorted_lengths = sorted(lengths)
    stats = {
        "input_path": str(input_path),
        "tokenizer_path": str(args.tokenizer_path),
        "total_samples": total_count,
        "skipped_samples": skipped_count,
        "unmatched_samples": unmatched_count,
        "average_length": (sum(lengths) / total_count) if total_count else 0,
        "median_length": percentile(sorted_lengths, 0.5),
        "p90_length": percentile(sorted_lengths, 0.9),
        "p95_length": percentile(sorted_lengths, 0.95),
        "p99_length": percentile(sorted_lengths, 0.99),
        "max_length": max(sorted_lengths) if sorted_lengths else 0,
        "bucket_counts": {},
    }

    for filename, _, _ in RANGES:
        count = bucket_counts[filename]
        stats["bucket_counts"][filename] = {
            "count": count,
            "ratio": (count / total_count) if total_count else 0,
        }

    stats_path = output_dir / "length_range_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Stats saved to {stats_path}")
    for filename, _, _ in RANGES:
        print(f"Bucket file: {output_dir / filename}")


if __name__ == "__main__":
    main()
