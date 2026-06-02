from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Tuple

import torch


# ============================================================
# 让脚本可以直接通过 python scripts/train.py 运行
# 不再需要 PYTHONPATH=.
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from tasksegment.configs import (
    DataConfig,
    DEFAULT_DENSE_GUIDANCE_LOSS_CFG,
    DEFAULT_EMA_ALPHA_CFG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_FG_RETRIEVAL_ALPHA,
    DEFAULT_TASK_IMPORTANCE_CFG,
    DEFAULT_TEXT_RETRIEVAL_WEIGHT,
    DEFAULT_TRAIN_LOSS_WEIGHT_CFG,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.models import TaskSegmentModel
from tasksegment.text import load_text_bank
from tasksegment.training import train


def _parse_image_size(values) -> Tuple[int, int]:
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError("--image-size 只接受 1 个值 H 或 2 个值 H W，例如 --image-size 512 或 --image-size 512 512")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TaskSegmentV3")

    # -------------------------
    # 路径参数
    # -------------------------
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据集根目录，下面应包含 thyroid、TN3K、BUSI_WHU、BUS-BRA 等子目录",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default="./text_features",
        help="文本特征目录",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="USTS_guidance_reg.pth",
        help="模型保存路径",
    )

    # -------------------------
    # 显式实验配置
    # -------------------------
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[512, 512],
        metavar=("H", "W"),
        help="输入图像尺寸。支持 --image-size 512 或 --image-size 512 512",
    )
    ftext_group = parser.add_mutually_exclusive_group()
    ftext_group.add_argument(
        "--use-ftext",
        dest="use_ftext",
        action="store_true",
        default=bool(DEFAULT_MODEL_CONFIG["use_ftext"]),
        help="启用文本 token / Ftext",
    )
    ftext_group.add_argument(
        "--no-use-ftext",
        dest="use_ftext",
        action="store_false",
        help="禁用文本 token / Ftext",
    )
    parser.add_argument(
        "--text-dim",
        type=int,
        default=int(DEFAULT_MODEL_CONFIG["text_dim"]),
        help="文本特征维度；会与已加载 text_features 做 assert",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(DEFAULT_MODEL_CONFIG["hidden_dim"]),
        help="模型 hidden_dim",
    )
    parser.add_argument(
        "--num-query-tokens",
        type=int,
        default=int(DEFAULT_MODEL_CONFIG["num_query_tokens"]),
        help="task encoder query token 数量",
    )
    parser.add_argument(
        "--guidance-dropout-prob",
        type=float,
        default=float(DEFAULT_MODEL_CONFIG.get("guidance_dropout_prob", 0.15)),
        help="训练时随机关闭 dense guidance 的概率；推理不生效",
    )
    parser.add_argument(
        "--guidance-scale-min",
        type=float,
        default=float(DEFAULT_MODEL_CONFIG.get("guidance_scale_min", 0.50)),
        help="训练时 dense guidance 随机缩放下界；推理不生效",
    )
    parser.add_argument(
        "--guidance-scale-max",
        type=float,
        default=float(DEFAULT_MODEL_CONFIG.get("guidance_scale_max", 1.00)),
        help="训练时 dense guidance 随机缩放上界；推理不生效",
    )
    parser.add_argument(
        "--guidance-clamp-max",
        type=float,
        default=float(DEFAULT_MODEL_CONFIG.get("guidance_clamp_max", 0.80)),
        help="dense guidance 用于特征调制前的最大值截断",
    )
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="把 Dataset 读取后的 tensor 缓存在内存中，减少重复 IO/resize 开销",
    )
    clahe_group = parser.add_mutually_exclusive_group()
    clahe_group.add_argument(
        "--cpu-clahe",
        dest="cpu_clahe",
        action="store_true",
        help="在 Dataset CPU 阶段随机应用 CLAHE，避免 GPU loop 内 .cpu().numpy()",
    )
    clahe_group.add_argument(
        "--no-cpu-clahe",
        dest="cpu_clahe",
        action="store_false",
        help="关闭 CPU CLAHE",
    )
    parser.set_defaults(cpu_clahe=True)
    parser.add_argument(
        "--train-query-positive-ratio",
        type=float,
        default=0.7,
        help="训练 episode query 中正样本目标比例",
    )

    # -------------------------
    # 训练参数
    # -------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="episode batch size",
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=1500,
        help="每个 epoch 的 episode 数量",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early stopping patience",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="warmup 轮数",
    )

    # -------------------------
    # few-shot / retrieval 参数
    # -------------------------
    parser.add_argument(
        "--max-support-per-domain",
        type=int,
        default=128,
        help="每个 domain 最多缓存多少个 support 样本",
    )
    parser.add_argument(
        "--val-ensemble-k",
        type=int,
        default=8,
        help="验证时 ensemble 的 support 数量",
    )
    parser.add_argument(
        "--ema-bank-momentum",
        type=float,
        default=0.99,
        help="EMA memory bank momentum",
    )
    # 🔥 新增了联合检索训练的控制开关
    parser.add_argument(
        "--retrieval-train",
        dest="retrieval_train_enabled",
        action="store_true",
        default=True,
        help="开启 Retrieval-Aware Training (默认开启)",
    )
    parser.add_argument(
        "--no-retrieval-train",
        dest="retrieval_train_enabled",
        action="store_false",
        help="关闭 Retrieval-Aware Training",
    )

    # -------------------------
    # loss 权重
    # -------------------------
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=0.5,
        help="边界加权 CE loss 权重",
    )
    parser.add_argument(
        "--raw-aux-weight",
        type=float,
        default=DEFAULT_TRAIN_LOSS_WEIGHT_CFG["raw_aux"],
        help="retrieval-aware training 中 raw segmentation auxiliary loss 权重",
    )
    parser.add_argument(
        "--token-consistency-weight",
        type=float,
        default=DEFAULT_TRAIN_LOSS_WEIGHT_CFG["token_consistency"],
        help="raw task token 与 retrieval task token 的一致性 loss 权重",
    )
    parser.add_argument(
        "--raw-dense-aux-weight",
        type=float,
        default=DEFAULT_TRAIN_LOSS_WEIGHT_CFG["raw_dense_aux"],
        help="raw dense guidance auxiliary loss 权重",
    )
    parser.add_argument(
        "--dense-main-weight",
        type=float,
        default=DEFAULT_TRAIN_LOSS_WEIGHT_CFG["dense_main"],
        help="主 dense guidance supervision loss 权重",
    )
    parser.add_argument(
        "--task-importance-dice-weight",
        type=float,
        default=DEFAULT_TASK_IMPORTANCE_CFG["dice_weight"],
        help="EMA task importance 中 per-sample Dice 的权重",
    )
    parser.add_argument(
        "--task-importance-confidence-weight",
        type=float,
        default=DEFAULT_TASK_IMPORTANCE_CFG["confidence_weight"],
        help="EMA task importance 中 prediction confidence 的权重",
    )
    parser.add_argument(
        "--task-importance-support-size-weight",
        type=float,
        default=DEFAULT_TASK_IMPORTANCE_CFG["support_size_weight"],
        help="EMA task importance 中 support mask size score 的权重",
    )
    parser.add_argument(
        "--task-importance-support-size-norm",
        type=float,
        default=DEFAULT_TASK_IMPORTANCE_CFG["support_size_norm"],
        help="support mask 面积归一化阈值",
    )
    parser.add_argument(
        "--task-importance-min",
        type=float,
        default=DEFAULT_TASK_IMPORTANCE_CFG["min_importance"],
        help="task importance 的下限",
    )

    # -------------------------
    # retrieval 权重
    # -------------------------
    parser.add_argument(
        "--fg-retrieval-alpha",
        type=float,
        default=DEFAULT_FG_RETRIEVAL_ALPHA,
        help="Foreground descriptor similarity weight within retrieval scoring (控制检索时前景特征与全局特征的权重)",
    )
    parser.add_argument(
        "--text-retrieval-weight",
        type=float,
        default=DEFAULT_TEXT_RETRIEVAL_WEIGHT,
        help="文本检索相似度权重",
    )

    # -------------------------
    # 设备与随机种子
    # -------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="是否启用完全确定性训练，开启后速度可能变慢",
    )

    return parser.parse_args()


def check_paths(args):
    data_root = Path(args.data_root)
    text_dir = Path(args.text_dir)

    if not data_root.exists():
        raise FileNotFoundError(
            f"数据目录不存在: {data_root}\n"
            f"请确认你的数据是否放在 {data_root.resolve()} 下，"
            f"或者通过 --data-root 指定正确路径。"
        )

    if not text_dir.exists():
        print(
            f"[警告] 文本特征目录不存在: {text_dir}\n"
            f"如果你启用了文本特征，请先运行:\n"
            f"  python scripts/build_text_features.py --output-dir {text_dir}\n"
        )


def main():
    args = parse_args()
    check_paths(args)

    image_size = _parse_image_size(args.image_size)
    set_seed(args.seed, deterministic=args.deterministic)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 指定了 cuda，但当前环境不可用，将自动切换到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("TaskSegmentV3 Training")
    print("=" * 80)
    print(f"项目根目录: {ROOT}")
    print(f"数据目录: {Path(args.data_root).resolve()}")
    print(f"文本特征目录: {Path(args.text_dir).resolve()}")
    print(f"模型保存路径: {Path(args.save_path).resolve()}")
    print(f"设备: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Episodes per epoch: {args.episodes_per_epoch}")
    print(f"Image size: {image_size}")
    print(f"Use Ftext: {args.use_ftext}")
    print("Use dense guidance: True")
    print(f"Text dim: {args.text_dim}")
    print(f"Retrieval Train Enabled: {args.retrieval_train_enabled}")
    print(f"Dataset cache: {args.cache_dataset}")
    print(f"CPU CLAHE: {args.cpu_clahe}")
    print("=" * 80)

    # ============================================================
    # 1. 构建数据集
    # ============================================================
    data_cfg = DataConfig(data_root=args.data_root, image_size=image_size)

    train_dataset, val_dataset, _ = get_datasets(
        data_cfg.roots(),
        image_size=data_cfg.image_size,
        seed=args.seed,
        train_query_positive_ratio=args.train_query_positive_ratio,
        prefer_empty_train_queries=True,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=args.cpu_clahe,
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # ============================================================
    # 2. 加载文本特征
    # ============================================================
    text_bank = load_text_bank(args.text_dir, device, expected_text_dim=args.text_dim)
    print(f"成功加载文本特征的域: {list(text_bank.keys())}")

    if len(text_bank) == 0:
        print(
            "[警告] 没有加载到任何文本特征。\n"
            "如果模型配置 use_ftext=True，建议先生成 text_features。"
        )

    # ============================================================
    # 3. 构建模型
    # ============================================================
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update(
        {
            "use_ftext": bool(args.use_ftext),
            "text_dim": int(args.text_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_query_tokens": int(args.num_query_tokens),
            "use_dense_text_guidance": True,
            "guidance_dropout_prob": float(args.guidance_dropout_prob),
            "guidance_scale_min": float(args.guidance_scale_min),
            "guidance_scale_max": float(args.guidance_scale_max),
            "guidance_clamp_max": float(args.guidance_clamp_max),
        }
    )

    model = TaskSegmentModel(
        **model_config
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("模型配置:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

    train_loss_weight_cfg = dict(DEFAULT_TRAIN_LOSS_WEIGHT_CFG)
    train_loss_weight_cfg.update(
        {
            "raw_aux": float(args.raw_aux_weight),
            "token_consistency": float(args.token_consistency_weight),
            "raw_dense_aux": float(args.raw_dense_aux_weight),
            "dense_main": float(args.dense_main_weight),
        }
    )

    dense_guidance_loss_cfg = dict(DEFAULT_DENSE_GUIDANCE_LOSS_CFG)

    print("训练 loss 权重:")
    for k, v in train_loss_weight_cfg.items():
        print(f"  {k}: {v}")
    print(f"Dense guidance loss cfg: {dense_guidance_loss_cfg}")

    task_importance_cfg = dict(DEFAULT_TASK_IMPORTANCE_CFG)
    task_importance_cfg.update(
        {
            "dice_weight": float(args.task_importance_dice_weight),
            "confidence_weight": float(args.task_importance_confidence_weight),
            "support_size_weight": float(args.task_importance_support_size_weight),
            "support_size_norm": float(args.task_importance_support_size_norm),
            "min_importance": float(args.task_importance_min),
        }
    )

    # ============================================================
    # 4. 开始训练
    # ============================================================
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        text_bank=text_bank,

        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        episodes_per_epoch=args.episodes_per_epoch,
        grad_accum_steps=args.grad_accum_steps,
        warmup_epochs=args.warmup_epochs,
        retrieval_train_enabled=args.retrieval_train_enabled, # 🔥 将命令行参数传给训练函数

        max_support_per_domain=args.max_support_per_domain,
        val_ensemble_k=args.val_ensemble_k,

        save_path=args.save_path,

        ema_bank_momentum=args.ema_bank_momentum,
        ema_alpha_cfg=dict(DEFAULT_EMA_ALPHA_CFG),

        boundary_loss_weight=args.boundary_loss_weight,

        text_retrieval_weight=args.text_retrieval_weight,
        fg_retrieval_alpha=args.fg_retrieval_alpha,

        model_config=model_config,
        dense_guidance_loss_cfg=dense_guidance_loss_cfg,
        train_loss_weight_cfg=train_loss_weight_cfg,
        task_importance_cfg=task_importance_cfg,
    )


if __name__ == "__main__":
    main()