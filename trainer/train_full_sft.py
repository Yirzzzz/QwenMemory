import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
    build_ckpt_path,
)

warnings.filterwarnings('ignore')


def build_snapshot_path(epoch, step):
    base_path = build_ckpt_path(args.save_dir, args.save_weight, lm_config=lm_config, ckpt_tag=args.ckpt_tag)
    stem, ext = os.path.splitext(base_path)
    return f"{stem}_epoch{epoch + 1}_step{step}{ext}"


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
            res = model(input_ids, labels=labels)
            aux_loss = getattr(res, 'aux_loss', None)
            if aux_loss is None:
                aux_loss = torch.tensor(0.0, device=args.device)
            loss = (res.loss + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })
            if progress is not None:
                progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

        if (step % args.save_steps == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = build_ckpt_path(args.save_dir, args.save_weight, lm_config=lm_config, ckpt_tag=args.ckpt_tag)
            snapshot_ckp = build_snapshot_path(epoch, step)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            cpu_state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(cpu_state_dict, ckp)
            torch.save(cpu_state_dict, snapshot_ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler,
                ckpt_tag=args.ckpt_tag,
            )
            model.train()
            del state_dict, cpu_state_dict

        del input_ids, labels, res, loss

    if progress is not None:
        progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen/MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重前缀")
    parser.add_argument('--ckpt_tag', default='qwen', type=str, help="checkpoint tag（Qwen推荐固定）")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--save_steps", type=int, default=0, help="按step保存快照；0时回退到save_interval")
    parser.add_argument('--hidden_size', default=512, type=int, help="仅用于MiniMind/兼容命名")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="仅用于MiniMind")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练最大长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE（仅MiniMind）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="从本地权重继续训练（none表示仅用HF初始化）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动恢复断点")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="Qwen-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")

    parser.add_argument('--model_source', type=str, default='qwen', choices=['qwen', 'minimind'], help='训练模型来源')
    parser.add_argument('--hf_model_path', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Qwen HF模型路径或本地目录')
    parser.add_argument('--rope_scaling_type', type=str, default='none', help='Qwen rope_scaling type，例如 yarn/linear')
    parser.add_argument('--rope_scaling_factor', type=float, default=1.0, help='Qwen rope_scaling factor，>1时生效')

    args = parser.parse_args()
    args.save_steps = args.save_steps or args.save_interval

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_source == 'minimind':
        lm_config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        )
    else:
        lm_config = type("QwenCompatConfig", (), {"use_moe": False, "hidden_size": args.hidden_size})()

    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints', ckpt_tag=args.ckpt_tag) if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"SFT-{args.model_source}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    rope_type = None if args.rope_scaling_type == 'none' else args.rope_scaling_type
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device,
        model_source=args.model_source,
        hf_model_path=args.hf_model_path,
        ckpt_tag=args.ckpt_tag,
        rope_scaling_type=rope_type,
        rope_scaling_factor=args.rope_scaling_factor,
    )

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        if args.model_source == 'minimind':
            model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}] 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
