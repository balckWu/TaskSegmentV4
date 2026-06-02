from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch


# ============================================================
# 让脚本可以直接通过 python scripts/predict.py 运行
# 不再需要 PYTHONPATH=.
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from tasksegment.configs import (
    DataConfig,
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_EMA_ALPHA_CFG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_POSTPROCESS_CFG,
    DEFAULT_FG_RETRIEVAL_ALPHA,
    DEFAULT_TEXT_RETRIEVAL_WEIGHT,
)
from tasksegment.data import get_datasets, set_seed
from tasksegment.inference import predict_all_organs
from tasksegment.models import TaskSegmentModel
from tasksegment.text import load_text_bank
from tasksegment.training import EMATaskMemoryBank


def _parse_image_size(values) -> Tuple[int, int]:
    if values is None:
        raise ValueError("image size values cannot be None")
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise ValueError("--image-size 只接受 1 个值 H 或 2 个值 H W，例如 --image-size 512 或 --image-size 512 512")


def _parse_int_list(value: str) -> List[int]:
    if value is None or value.strip() == "":
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _validate_checkpoint_schema(checkpoint: object, ckpt_path: Path) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint schema 错误: {ckpt_path} 不是 dict。")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint schema 错误: {ckpt_path} 缺少 model_state_dict。")
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with TaskSegmentV3")

    parser.add_argument(
        "--model-path",
        type=str,
        default="USTS_guidance_reg.pth",
        help="训练好的模型权重路径。只应加载自己训练或可信来源的 checkpoint。",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据集根目录",
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default="./text_features",
        help="文本特征目录",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./vis_dense",
        help="预测可视化保存目录",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        metavar=("H", "W"),
        help="输入图像尺寸。默认优先读取 checkpoint train_config['image_size']，否则使用 512 512。",
    )
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="把 Dataset 读取后的 tensor 缓存在内存中，减少重复 IO/resize 开销",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="推理 batch size，当前建议保持 1",
    )
    parser.add_argument(
        "--max-support-per-domain",
        type=int,
        default=128,
        help="每个 domain 最多使用多少 support 样本构建 support bank",
    )
    parser.add_argument(
        "--ensemble-k",
        type=int,
        default=8,
        help="每个 query 检索多少个 support token 做 ensemble",
    )
    parser.add_argument(
        "--num-vis-per-organ",
        type=int,
        default=5,
        help="每个器官保存多少张可视化图",
    )
    parser.add_argument(
        "--vis-data-idx",
        type=str,
        default="",
        help="额外强制可视化指定 test dataset data_idx，逗号分隔，例如 700,659,754,409,771",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="推理时覆盖 dense guidance 强度。0 表示只计算 guidance 可视化但不让它调制特征；建议扫 0, 0.25, 0.5, 0.75, 1.0。",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="全域统一 threshold；不传则使用 config/checkpoint 中的 per-domain threshold",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="关闭预测后处理。默认与验证一致：threshold + postprocess",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help="后处理最小连通域面积；不传则使用 config/checkpoint 默认值",
    )
    parser.add_argument(
        "--closing-kernel",
        type=int,
        default=None,
        help="后处理闭运算 kernel；不传则使用 config/checkpoint 默认值",
    )
    
    # 🔥 新增这一段：保留最大连通区域的开关
    parser.add_argument(
        "--keep-largest",
        action="store_true",
        help="后处理时是否只保留最大连通区域，彻底过滤孤立假阳性碎片",
    )

    parser.add_argument(
        "--fg-retrieval-alpha",
        "--retrieval-alpha",
        dest="fg_retrieval_alpha",
        type=float,
        default=DEFAULT_FG_RETRIEVAL_ALPHA,
        help="前景 descriptor 在 retrieval score 中的权重",
    )
    parser.add_argument(
        "--text-retrieval-weight",
        type=float,
        default=DEFAULT_TEXT_RETRIEVAL_WEIGHT,
        help="文本检索权重",
    )
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

    return parser.parse_args()


def load_checkpoint(model_path: str, device: torch.device):
    ckpt_path = Path(model_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件: {ckpt_path.resolve()}\n"
            f"请通过 --model-path 指定正确的 .pth 文件。"
        )

    checkpoint = _validate_checkpoint_schema(_safe_torch_load(str(ckpt_path), map_location=device), ckpt_path)

    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update(checkpoint.get("model_config", {}))
    model_config.pop("decoder_num_classes", None)

    model = TaskSegmentModel(**model_config).to(device)

    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "[info] checkpoint 以 non-strict 方式加载："
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    model.eval()

    ema_bank = EMATaskMemoryBank()
    if "ema_memory_bank" in checkpoint:
        ema_bank.load_state_dict(checkpoint["ema_memory_bank"])

    train_config = checkpoint.get("train_config", {})

    return model, ema_bank, model_config, train_config


def main():
    args = parse_args()
    vis_data_indices = _parse_int_list(args.vis_data_idx)

    set_seed(args.seed, deterministic=False)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 指定了 cuda，但当前环境不可用，将自动切换到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ============================================================
    # 1. 加载模型
    # ============================================================
    model, ema_bank, model_config, train_config = load_checkpoint(
        args.model_path,
        device,
    )

    if args.image_size is not None:
        image_size = _parse_image_size(args.image_size)
    elif "image_size" in train_config:
        image_size = tuple(int(x) for x in train_config["image_size"])
    else:
        image_size = (512, 512)

    domain_thresholds = dict(train_config.get("domain_thresholds", DEFAULT_DOMAIN_THRESHOLDS))
    if args.threshold is not None:
        domain_thresholds = {k: float(args.threshold) for k in domain_thresholds.keys()}

    postprocess_cfg = dict(train_config.get("postprocess_cfg", DEFAULT_POSTPROCESS_CFG))
    if args.min_area is not None:
        postprocess_cfg["min_area"] = int(args.min_area)
    if args.closing_kernel is not None:
        postprocess_cfg["closing_kernel"] = int(args.closing_kernel)
        
    # 🔥 新增这两行：将命令行的开关状态写入后处理配置
    if args.keep_largest:
        postprocess_cfg["keep_largest"] = True

    if args.guidance_scale is not None:
        model.decoder.dense_guidance_strength = float(args.guidance_scale)
    effective_guidance_scale = float(getattr(model.decoder, "dense_guidance_strength", 0.0))

    print("=" * 80)
    print("TaskSegmentV3 Predict")
    print("=" * 80)
    print(f"项目根目录: {ROOT}")
    print(f"模型路径: {Path(args.model_path).resolve()}")
    print(f"数据目录: {Path(args.data_root).resolve()}")
    print(f"文本特征目录: {Path(args.text_dir).resolve()}")
    print(f"保存目录: {Path(args.save_dir).resolve()}")
    print(f"设备: {device}")
    print(f"Image size: {image_size}")
    print(f"Use postprocess: {not args.no_postprocess}")
    print(f"Domain thresholds: {domain_thresholds}")
    print(f"Postprocess cfg: {postprocess_cfg}")
    print(f"Target vis data_idx: {vis_data_indices}")
    print(f"Dense guidance scale: {effective_guidance_scale}")
    print("=" * 80)

    print("模型配置:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    # ============================================================
    # 2. 加载数据
    # ============================================================
    data_cfg = DataConfig(data_root=args.data_root, image_size=image_size)

    train_dataset, _, test_dataset = get_datasets(
        data_cfg.roots(),
        image_size=data_cfg.image_size,
        seed=args.seed,
        train_query_positive_ratio=0.7,
        prefer_empty_train_queries=True,
        cache_in_memory=args.cache_dataset,
        use_cpu_clahe=False,
    )

    print(f"Support 数据集样本数: {len(train_dataset)}")
    print(f"Test 数据集样本数: {len(test_dataset)}")

    # ============================================================
    # 3. 加载文本特征
    # ============================================================
    text_bank = load_text_bank(
        args.text_dir,
        device,
        expected_text_dim=int(model_config.get("text_dim", DEFAULT_MODEL_CONFIG["text_dim"])),
    )
    print(f"成功加载文本特征的域: {list(text_bank.keys())}")

    # ============================================================
    # 4. 推理
    # ============================================================
    predict_all_organs(
        model=model,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        device=device,
        text_bank=text_bank,
        ema_bank=ema_bank,
        save_dir=args.save_dir,

        max_support_per_domain=args.max_support_per_domain,
        ensemble_k=args.ensemble_k,

        retrieval_alpha=args.fg_retrieval_alpha,
        text_retrieval_weight=args.text_retrieval_weight,
        ema_alpha_cfg=dict(DEFAULT_EMA_ALPHA_CFG),

        domain_thresholds=domain_thresholds,
        postprocess_cfg=postprocess_cfg,
        use_postprocess=not args.no_postprocess,
        num_vis_per_organ=args.num_vis_per_organ,
        vis_data_indices=vis_data_indices,
        guidance_scale=effective_guidance_scale,
    )


if __name__ == "__main__":
    main()