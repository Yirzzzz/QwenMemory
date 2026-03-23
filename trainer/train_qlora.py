import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from datasets import Dataset as HFDataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    is_main_process,
    setup_seed,
)

warnings.filterwarnings('ignore')


def build_adapter_dir():
    return Path(args.save_dir) / f"{args.adapter_name}_{args.ckpt_tag}"


def build_snapshot_dir(epoch, step):
    return Path(args.save_dir) / f"{args.adapter_name}_{args.ckpt_tag}_epoch{epoch + 1}_step{step}"


def inspect_trainable_params(model, max_names=100):
    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(numel for _, numel in trainable)
    Logger(f"Actual Trainable Params: {trainable_params / 1e6:.3f}M")
    Logger(f"Trainable Parameter Tensors: {len(trainable)}")
    if trainable:
        preview = ", ".join(name for name, _ in trainable[:max_names])
        Logger(f"Trainable Param Names (first {min(len(trainable), max_names)}): {preview}")


def maybe_wrap_ddp(model, local_rank):
    if not dist.is_initialized():
        return model
    return DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_adapter(model, output_dir):
    raw_model = unwrap_model(model)
    raw_model.save_pretrained(output_dir)


def load_quantized_model():
    compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=args.double_quant == 1,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map={"": args.device},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def attach_qlora_adapter(model):
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    adapter_dir = build_adapter_dir()
    if args.resume_adapter and adapter_dir.exists():
        Logger(f"Loading existing adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    return model


def build_train_dataset(tokenizer):
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if args.max_samples > 0:
        subset_indices = list(range(min(args.max_samples, len(train_ds))))
        train_ds.samples = train_ds.samples.select(subset_indices)
    return train_ds


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    progress = None
    if is_main_process() and tqdm is not None:
        progress = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)
    iterable = progress if progress is not None else loader

    for step, (input_ids, labels) in enumerate(iterable, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids=input_ids, labels=labels)
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })
            if progress is not None:
                progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

        if (step % args.save_steps == 0 or step == iters - 1) and is_main_process():
            model.eval()
            latest_dir = build_adapter_dir()
            snapshot_dir = build_snapshot_dir(epoch, step)
            latest_dir.mkdir(parents=True, exist_ok=True)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            save_adapter(model, latest_dir)
            save_adapter(model, snapshot_dir)
            model.train()

        del input_ids, labels, res, loss

        if args.max_steps > 0 and step >= args.max_steps:
            Logger(f"Reached max_steps={args.max_steps}, stopping current epoch early")
            break

    if progress is not None:
        progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen QLoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/qlora", help="Adapter 保存目录")
    parser.add_argument("--adapter_name", type=str, default="qlora_adapter", help="Adapter 名称前缀")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="QLoRA 计算精度")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_steps", type=int, default=4000, help="按 step 保存 adapter")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练最大长度")
    parser.add_argument("--data_path", type=str, required=True, help="QLoRA 训练数据路径")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Qwen HF 模型路径或本地目录")
    parser.add_argument("--ckpt_tag", type=str, default="qwen15", help="保存 tag")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="逗号分隔的 target modules")
    parser.add_argument("--quant_type", type=str, default="nf4", choices=["nf4", "fp4"], help="4bit quant type")
    parser.add_argument("--double_quant", type=int, default=1, choices=[0, 1], help="是否启用 double quant")
    parser.add_argument("--resume_adapter", type=int, default=0, choices=[0, 1], help="是否从 latest adapter 续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb/swanlab")
    parser.add_argument("--wandb_project", type=str, default="Qwen-QLoRA", help="wandb 项目名")
    parser.add_argument("--max_steps", type=int, default=0, help="Smoke test 最大 step；0 表示完整训练")
    parser.add_argument("--max_samples", type=int, default=0, help="仅取前 N 条样本做 smoke test；0 表示全部")
    args = parser.parse_args()

    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be >= 1")
    if args.save_steps < 1:
        raise ValueError("--save_steps must be >= 1")

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=compute_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        run_name = f"QLoRA-{args.adapter_name}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=run_name)

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise ImportError("QLoRA requires bitsandbytes. Please install bitsandbytes first.") from exc

    model, tokenizer = load_quantized_model()
    model = attach_qlora_adapter(model)
    inspect_trainable_params(model)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    train_ds = build_train_dataset(tokenizer)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    model = maybe_wrap_ddp(model, local_rank)

    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader), 0, wandb)
        if args.max_steps > 0:
            Logger(f"Smoke test mode reached max_steps={args.max_steps}, stopping after epoch {epoch + 1}")
            break

    if dist.is_initialized():
        dist.destroy_process_group()
