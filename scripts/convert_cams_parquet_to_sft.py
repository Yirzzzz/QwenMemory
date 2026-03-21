import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def stringify_turn(turn):
    if isinstance(turn, str):
        return turn.strip()

    if isinstance(turn, dict):
        speaker = (
            turn.get("speaker")
            or turn.get("role")
            or turn.get("name")
            or turn.get("participant")
            or ""
        )
        content = (
            turn.get("text")
            or turn.get("content")
            or turn.get("utterance")
            or turn.get("sentence")
            or ""
        )
        content = str(content).strip()
        if speaker and content:
            return f"{speaker}: {content}"
        return content

    return str(turn).strip()


def normalize_dialogue(value):
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        turns = [stringify_turn(turn) for turn in value]
        turns = [turn for turn in turns if turn]
        return "\n".join(turns).strip()

    if isinstance(value, dict):
        return stringify_turn(value)

    return str(value).strip()


def build_user_content(dialogue, instruction, input_prefix):
    parts = [instruction.strip()]
    if input_prefix:
        parts.append(input_prefix.strip())
    parts.append(dialogue.strip())
    return "\n\n".join(part for part in parts if part)


def resolve_target_col(args):
    if args.target_col:
        return args.target_col

    summary_map = {
        "short": "short_summary",
        "medium": "medium_summary",
        "long": "long_summary",
    }
    return summary_map[args.summary_level]


def build_structured_summary(row, summary_col):
    payload = {
        "summary": str(row.get(summary_col, "") or "").strip(),
        "keywords": row.get("keywords", []),
        "statement_type": str(row.get("statement_type", "") or "").strip(),
        "tense": str(row.get("tense", "") or "").strip(),
        "sentiment": str(row.get("sentiment", "") or "").strip(),
    }

    if isinstance(payload["keywords"], str):
        payload["keywords"] = [part.strip() for part in payload["keywords"].split(",") if part.strip()]
    elif not isinstance(payload["keywords"], list):
        payload["keywords"] = [str(payload["keywords"]).strip()] if str(payload["keywords"]).strip() else []

    return json.dumps(payload, ensure_ascii=False)


def build_record(dialogue, assistant_content, system_prompt, instruction, input_prefix):
    conversations = []
    if system_prompt.strip():
        conversations.append({"role": "system", "content": system_prompt.strip()})
    conversations.extend(
        [
            {
                "role": "user",
                "content": build_user_content(dialogue, instruction, input_prefix),
            },
            {"role": "assistant", "content": assistant_content},
        ]
    )
    return {"conversations": conversations}


def main():
    parser = argparse.ArgumentParser(description="Convert CAMS parquet data to MiniMind/Qwen SFT jsonl format")
    parser.add_argument("--input_path", required=True, help="Input parquet file or directory")
    parser.add_argument("--output_path", required=True, help="Output jsonl path")
    parser.add_argument("--source_col", default="text", help="Source conversation column name")
    parser.add_argument("--target_col", default="", help="Target summary column name; overrides --summary_level")
    parser.add_argument("--summary_level", choices=["short", "medium", "long"], default="short", help="Summary level for CAMS fields")
    parser.add_argument("--output_mode", choices=["plain", "structured"], default="plain", help="Assistant output mode")
    parser.add_argument("--append", action="store_true", help="Append to output jsonl instead of overwriting")
    parser.add_argument("--split", action="store_true", help="Split output into train/val/test files")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split shuffle")
    parser.add_argument("--system_prompt", default="", help="Optional system prompt")
    parser.add_argument("--instruction", default="请阅读下面的会话内容，并生成简洁准确的中文摘要。", help="Instruction prepended to user input")
    parser.add_argument("--input_prefix", default="会话内容：", help="Prefix before normalized dialogue")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_col = resolve_target_col(args)

    if input_path.is_dir():
        parquet_files = sorted(input_path.glob("*.parquet"))
    else:
        parquet_files = [input_path]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {input_path}")

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    records = []
    written = 0
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        rows = table.to_pylist()

        for row in rows:
            if args.source_col not in row:
                raise KeyError(
                    f"Missing required columns. Available columns: {list(row.keys())}"
                )

            if args.output_mode == "plain" and target_col not in row:
                raise KeyError(
                    f"Missing target column '{target_col}'. Available columns: {list(row.keys())}"
                )

            dialogue = normalize_dialogue(row[args.source_col])
            if args.output_mode == "plain":
                assistant_content = str(row[target_col]).strip()
            else:
                assistant_content = build_structured_summary(row, target_col)

            if not dialogue or not assistant_content:
                continue

            records.append(
                build_record(
                    dialogue,
                    assistant_content,
                    args.system_prompt,
                    args.instruction,
                    args.input_prefix,
                )
            )

    if args.split:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        total = len(records)
        train_end = int(total * args.train_ratio)
        val_end = train_end + int(total * args.val_ratio)
        split_map = {
            "train": records[:train_end],
            "val": records[train_end:val_end],
            "test": records[val_end:],
        }
        stem = output_path.stem
        suffix = output_path.suffix or ".jsonl"
        for split_name, split_records in split_map.items():
            split_path = output_path.with_name(f"{stem}_{split_name}{suffix}")
            with split_path.open("w", encoding="utf-8") as f:
                for record in split_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += len(split_records)
        print(
            f"Converted {written} samples to "
            f"{output_path.with_name(f'{stem}_train{suffix}')}, "
            f"{output_path.with_name(f'{stem}_val{suffix}')}, "
            f"{output_path.with_name(f'{stem}_test{suffix}')}"
        )
        return

    output_mode = "a" if args.append else "w"
    with output_path.open(output_mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Converted {written} samples to {output_path}")


if __name__ == "__main__":
    main()
