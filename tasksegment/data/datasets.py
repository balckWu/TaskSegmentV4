from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resize_pil(img: Image.Image, image_size: Tuple[int, int], interpolation=Image.BILINEAR) -> Image.Image:
    # image_size is kept as (H, W), while PIL expects (W, H).
    h, w = int(image_size[0]), int(image_size[1])
    return img.resize((w, h), interpolation)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[None, ...]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


class PerImageZScoreNormalize:
    """Per-image normalization to reduce cross-domain intensity gaps."""

    def __init__(self, clamp_value: float = 5.0, eps: float = 1e-6):
        self.clamp_value = float(clamp_value)
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std().clamp_min(self.eps)
        tensor = (tensor - mean) / std
        tensor = tensor.clamp(-self.clamp_value, self.clamp_value) / self.clamp_value
        return tensor


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def _apply_clahe_pil(img: Image.Image, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Image.Image:
    """Apply CLAHE on CPU before tensor/GPU conversion."""
    img_np = np.array(img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    img_np = clahe.apply(img_np)
    return Image.fromarray(img_np, mode="L")


def _default_clahe_cfg_by_domain() -> Dict[str, Dict[str, float]]:
    return {
        "thyroid": {"prob": 0.18, "clip_limit": 1.4, "tile_grid_size": 8},
        "TN3K": {"prob": 0.18, "clip_limit": 1.4, "tile_grid_size": 8},
        "BUSI_WHU": {"prob": 0.30, "clip_limit": 1.8, "tile_grid_size": 8},
        "BUS-BRA": {"prob": 0.30, "clip_limit": 1.8, "tile_grid_size": 8},
    }


class MultiOrganDataset(Dataset):
    """多域二分类前景分割数据集。"""

    def __init__(
        self,
        data_roots: Dict[str, str],
        image_transform=None,
        split: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        seed: int = 42,
        query_positive_ratio: float = 0.7,
        prefer_empty_queries: bool = True,
        cache_in_memory: bool = False,
        use_cpu_clahe: bool = False,
        cpu_clahe_cfg_by_domain: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.image_transform = image_transform
        self.samples: List[dict] = []
        self.organ_to_indices: Dict[str, List[int]] = {}
        self.organ_to_positive_indices: Dict[str, List[int]] = {}
        self.organ_to_empty_indices: Dict[str, List[int]] = {}
        self.split = split
        self.image_size = image_size
        self.seed = seed
        self.query_positive_ratio = float(min(max(query_positive_ratio, 0.0), 1.0))
        self.prefer_empty_queries = bool(prefer_empty_queries)
        self.cache_in_memory = bool(cache_in_memory)
        self.use_cpu_clahe = bool(use_cpu_clahe)
        self.cpu_clahe_cfg_by_domain = cpu_clahe_cfg_by_domain or _default_clahe_cfg_by_domain()
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, str]] = {}

        if self.cache_in_memory and self.use_cpu_clahe:
            print(
                "[提示] 当前数据集同时开启了 cache_in_memory 和 use_cpu_clahe。"
                "随机 CPU CLAHE 会在首次读取时被缓存，因此不再每次重采样。"
            )

        print(f"正在初始化多器官数据集 (split: {split})...")

        for organ_name, root_path in data_roots.items():
            if split is not None:
                sub_dirs = [split]
            else:
                sub_dirs = ["train", "val", "test"]

            for split_name in sub_dirs:
                split_path = os.path.join(root_path, split_name)
                if not os.path.exists(split_path):
                    continue

                img_dir = os.path.join(split_path, "images")
                mask_dir = os.path.join(split_path, "masks")
                if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                    continue

                img_files = sorted(os.listdir(img_dir))
                valid_count = 0
                positive_count = 0
                missing_mask_count = 0

                for img_name in img_files:
                    img_path = os.path.join(img_dir, img_name)
                    
                    # 提取不带后缀的文件名 (例如 '123.jpg' 或 '123.JPG' 变成 '123')
                    base_name, _ = os.path.splitext(img_name)
                    
                    # 灵活匹配掩码后缀 (支持 .png, .PNG 以及与其他数据集兼容的同名后缀)
                    mask_path_png = os.path.join(mask_dir, f"{base_name}.png")
                    mask_path_PNG_upper = os.path.join(mask_dir, f"{base_name}.PNG")
                    mask_path_exact = os.path.join(mask_dir, img_name)
                    
                    if os.path.exists(mask_path_png):
                        mask_path = mask_path_png
                    elif os.path.exists(mask_path_PNG_upper):
                        mask_path = mask_path_PNG_upper
                    else:
                        mask_path = mask_path_exact

                    if not os.path.exists(mask_path):
                        missing_mask_count += 1
                        continue

                    label = Image.open(mask_path).convert("L")
                    has_foreground = bool((np.array(label) > 0).any())

                    sample = {
                        "image": img_path,
                        "mask": mask_path,
                        "organ": organ_name,
                        "split": split_name,
                        "has_foreground": has_foreground,
                    }

                    idx = len(self.samples)
                    self.samples.append(sample)
                    self.organ_to_indices.setdefault(organ_name, []).append(idx)
                    if has_foreground:
                        self.organ_to_positive_indices.setdefault(organ_name, []).append(idx)
                        positive_count += 1
                    else:
                        self.organ_to_empty_indices.setdefault(organ_name, []).append(idx)

                    valid_count += 1

                empty_mask_count = valid_count - positive_count
                print(
                    f" - 加载 {organ_name}/{split_name}: 总样本 {valid_count} | "
                    f"前景样本 {positive_count} | 空掩码 {empty_mask_count} | 缺失掩码 {missing_mask_count}"
                )

        for organ_name in self.organ_to_indices.keys():
            self.organ_to_positive_indices.setdefault(organ_name, [])
            self.organ_to_empty_indices.setdefault(organ_name, [])

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(
        self,
        pool: List[int],
        num_samples: int,
        replace: Optional[bool] = None,
    ) -> List[int]:
        if num_samples <= 0 or len(pool) == 0:
            return []
        if replace is None:
            replace = len(pool) < num_samples
        if (not replace) and len(pool) < num_samples:
            num_samples = len(pool)
        return np.random.choice(pool, num_samples, replace=replace).tolist()

    def get_organ_samples(self, organ_name: str, num_samples: int, positive_only: bool = False) -> List[int]:
        if positive_only:
            indices = self.organ_to_positive_indices.get(organ_name, [])
        else:
            indices = self.organ_to_indices.get(organ_name, [])

        if len(indices) == 0:
            suffix = "前景样本" if positive_only else "可用样本"
            raise ValueError(f"器官/域 {organ_name} 没有{suffix}！")
        return self._sample_indices(indices, num_samples)

    def sample_episode_indices(
        self,
        organ_name: str,
        num_support: int,
        num_query: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Support 始终使用正样本；Query 采用“正样本 + 全样本/空掩码优先”的混采策略。
        """
        pos_pool = list(self.organ_to_positive_indices.get(organ_name, []))
        all_pool = list(self.organ_to_indices.get(organ_name, []))
        empty_pool = list(self.organ_to_empty_indices.get(organ_name, []))

        if len(pos_pool) == 0:
            raise ValueError(f"器官/域 {organ_name} 没有前景样本！")
        if len(all_pool) == 0:
            raise ValueError(f"器官/域 {organ_name} 没有可用样本！")

        support_indices = self._sample_indices(pos_pool, num_support)
        used_support = set(support_indices)

        target_pos_queries = int(round(num_query * self.query_positive_ratio))
        if num_query > 0 and len(pos_pool) > 0:
            target_pos_queries = max(1, target_pos_queries)
        target_pos_queries = min(num_query, target_pos_queries)

        # ---------- step1: 正样本 query，优先与 support 严格去重 ----------
        pos_candidates = [idx for idx in pos_pool if idx not in used_support]
        if len(pos_candidates) > 0:
            query_indices = self._sample_indices(
                pos_candidates,
                target_pos_queries,
                replace=len(pos_candidates) < target_pos_queries,
            )
        else:
            query_indices = []
            if target_pos_queries > 0:
                # 只有在根本没有非重叠正样本时才回退
                query_indices = self._sample_indices(pos_pool, target_pos_queries, replace=True)

        remaining_needed = max(0, num_query - len(query_indices))

        # ---------- step2: 补齐剩余 query，优先空掩码，再退回全样本 ----------
        if remaining_needed > 0:
            excluded = used_support if len([idx for idx in all_pool if idx not in used_support]) > 0 else set()
            
            fallback_candidates: List[int] = []
            if self.prefer_empty_queries:
                empty_candidates = [idx for idx in empty_pool if idx not in excluded]
                fallback_candidates.extend(empty_candidates)

            all_candidates = [idx for idx in all_pool if idx not in excluded]

            if len(fallback_candidates) > 0:
                seen = set()
                merged = []
                for idx in fallback_candidates + all_candidates:
                    if idx not in seen:
                        merged.append(idx)
                        seen.add(idx)
                fallback_candidates = merged
            else:
                fallback_candidates = all_candidates

            if len(fallback_candidates) == 0:
                fallback_candidates = list(all_pool)

            query_indices.extend(
                self._sample_indices(
                    fallback_candidates,
                    remaining_needed,
                    replace=len(fallback_candidates) < remaining_needed,
                )
            )

        if len(query_indices) == 0:
            query_indices = self._sample_indices(pos_pool, num_query, replace=True)

        return support_indices, query_indices

    def _maybe_apply_cpu_clahe(self, img: Image.Image, organ_name: str) -> Image.Image:
        if not self.use_cpu_clahe:
            return img
        cfg = self.cpu_clahe_cfg_by_domain.get(organ_name, None)
        if cfg is None:
            return img
        prob = float(cfg.get("prob", 0.0))
        if prob <= 0.0 or random.random() >= prob:
            return img
        return _apply_clahe_pil(
            img,
            clip_limit=float(cfg.get("clip_limit", 2.0)),
            tile_grid_size=int(cfg.get("tile_grid_size", 8)),
        )

    @staticmethod
    def _clone_cached_item(item: Tuple[torch.Tensor, torch.Tensor, str]):
        img, label_tensor, organ = item
        return img.clone(), label_tensor.clone(), organ

    def _load_item_uncached(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("L")
        label = Image.open(sample["mask"]).convert("L")

        img = self._maybe_apply_cpu_clahe(img, sample["organ"])

        img = _resize_pil(img, self.image_size, Image.BILINEAR)
        label = _resize_pil(label, self.image_size, Image.NEAREST)

        img = _pil_to_tensor(img)
        if self.image_transform is not None:
            img = self.image_transform(img)

        label_np = (np.array(label) > 0).astype(np.uint8)
        label_tensor = torch.from_numpy(label_np).long()
        return img.contiguous(), label_tensor.contiguous(), sample["organ"]

    def __getitem__(self, idx: int):
        if self.cache_in_memory and idx in self._cache:
            return self._clone_cached_item(self._cache[idx])

        item = self._load_item_uncached(idx)
        if self.cache_in_memory:
            self._cache[idx] = (item[0].clone(), item[1].clone(), item[2])
            return self._clone_cached_item(self._cache[idx])
        return item

    @property
    def label_resize(self):
        return lambda label: _resize_pil(label, self.image_size, Image.NEAREST)


def get_image_transform():
    return PerImageZScoreNormalize(clamp_value=5.0)


def get_datasets(
    data_config: Dict[str, str],
    image_size: Tuple[int, int] = (512, 512),
    seed: int = 42,
    train_query_positive_ratio: float = 0.7,
    prefer_empty_train_queries: bool = True,
    cache_in_memory: bool = False,
    use_cpu_clahe: bool = False,
):
    transform_img = get_image_transform()
    train_dataset = MultiOrganDataset(
        data_config,
        image_transform=transform_img,
        split="train",
        image_size=image_size,
        seed=seed,
        query_positive_ratio=train_query_positive_ratio,
        prefer_empty_queries=prefer_empty_train_queries,
        cache_in_memory=cache_in_memory,
        use_cpu_clahe=use_cpu_clahe,
    )
    val_dataset = MultiOrganDataset(
        data_config,
        image_transform=transform_img,
        split="val",
        image_size=image_size,
        seed=seed,
        cache_in_memory=cache_in_memory,
        use_cpu_clahe=False,
    )
    test_dataset = MultiOrganDataset(
        data_config,
        image_transform=transform_img,
        split="test",
        image_size=image_size,
        seed=seed,
        cache_in_memory=cache_in_memory,
        use_cpu_clahe=False,
    )
    return train_dataset, val_dataset, test_dataset