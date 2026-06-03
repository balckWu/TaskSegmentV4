from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple, Optional

import torch
from transformers import BertModel, BertTokenizer

# export HF_ENDPOINT=https://hf-mirror.com

GROUP_ORDER = ["anatomy", "appearance", "boundary", "exclusion", "task"]
GROUP2ID = {k: i for i, k in enumerate(GROUP_ORDER)}
GROUP_WEIGHTS = {
    "anatomy": 1.20,
    "appearance": 1.00,
    "boundary": 1.25,
    "exclusion": 0.90,
    "task": 1.10,
}

PROMPT_BANK: Dict[str, Dict[str, List[str]]] = {
    "thyroid": {
        "anatomy": [
            "thyroid gland",
            "thyroid nodule",
            "thyroid lesion",
            "focal thyroid mass",
        ],
        "appearance": [
            "hypoechoic lesion",
            "heterogeneous echo pattern",
            "nonuniform internal texture",
            "low contrast foreground",
            "compact lesion region",
        ],
        "boundary": [
            "blurred margin",
            "irregular contour",
            "faint boundary",
            "partially obscured border",
            "closed lesion boundary",
        ],
        "exclusion": [
            "normal thyroid parenchyma",
            "background tissue",
            "ultrasound speckle noise",
            "non lesion region",
        ],
        "task": [
            "segment the thyroid nodule region",
            "focus on lesion pixels instead of the whole thyroid gland",
            "separate the suspicious nodule from surrounding tissue",
        ],
    },

    "TN3K": {
        "anatomy": [
            "thyroid gland",
            "thyroid nodule",
            "thyroid lesion",
        ],
        "appearance": [
            "hypoechoic or isoechoic nodule",
            "heterogeneous texture",
            "possible microcalcification",
            "low contrast lesion",
        ],
        "boundary": [
            "well defined or blurred margin",
            "irregular or regular contour",
            "lesion boundary",
            "compact foreground boundary",
        ],
        "exclusion": [
            "normal gland tissue",
            "background region",
            "ultrasound noise",
        ],
        "task": [
            "segment the thyroid lesion area",
            "highlight lesion pixels in the ultrasound scan",
        ],
    },

    "BUSI_WHU": {
        "anatomy": [
            "breast lesion",
            "breast mass",
            "breast tumor",
        ],
        "appearance": [
            "low echogenicity",
            "heterogeneous internal texture",
            "oval or irregular mass",
            "posterior acoustic shadowing",
        ],
        "boundary": [
            "indistinct margins",
            "irregular boundary",
            "true lesion contour",
            "compact tumor boundary",
        ],
        "exclusion": [
            "normal breast tissue",
            "normal parenchyma",
            "speckle noise",
            "non lesion region",
        ],
        "task": [
            "segment the breast lesion region",
            "separate the breast abnormality from normal tissue",
        ],
    },

    "BUS-BRA": {
        "anatomy": [
            "breast lesion",
            "breast mass",
            "solid breast mass",
        ],
        "appearance": [
            "heterogeneous internal echoes",
            "oval round or irregular lesion",
            "posterior acoustic shadowing or enhancement",
            "varying echogenicity",
        ],
        "boundary": [
            "distinct or indistinct margin",
            "compact lesion boundary",
            "mass contour",
        ],
        "exclusion": [
            "normal glandular tissue",
            "fatty tissue",
            "fibrous tissue",
            "acoustic artifacts",
        ],
        "task": [
            "segment the breast mass region",
            "focus on the compact lesion area",
        ],
    },

}


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled


def _flatten_prompt_dict(prompt_dict: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
    items: List[Tuple[str, str, float]] = []
    for group_name in GROUP_ORDER:
        prompts = prompt_dict.get(group_name, [])
        for prompt in prompts:
            items.append((group_name, prompt, float(GROUP_WEIGHTS[group_name])))
    return items


def _filter_subtokens(
    tokenizer: BertTokenizer,
    input_ids: torch.Tensor,
    token_feats: torch.Tensor,
    valid_positions: List[int],
) -> Tuple[torch.Tensor, List[str]]:
    sub_tokens = tokenizer.convert_ids_to_tokens(input_ids[valid_positions].tolist())

    kept_feats = []
    kept_texts = []

    for feat, tok in zip(token_feats, sub_tokens):
        clean_tok = tok.replace("##", "").strip()
        if clean_tok == "":
            continue
        if all(ch in "-_,.;:()[]{}\\/|?!'\"" for ch in clean_tok):
            continue
        if len(clean_tok) <= 1 and clean_tok.lower() not in {"t", "c"}:
            continue

        kept_feats.append(feat.unsqueeze(0))
        kept_texts.append(clean_tok)

    if len(kept_feats) == 0:
        kept_feats = [token_feats.mean(dim=0, keepdim=True)]
        kept_texts = ["fallback_token"]

    return torch.cat(kept_feats, dim=0), kept_texts


@torch.no_grad()
def encode_prompt_bank(
    model,
    tokenizer,
    prompt_dict: Dict[str, List[str]],
    max_length: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    面向“稀疏文本 -> 密集语义诱导”的文本编码版本：
    1) 结构化 prompt：anatomy / appearance / boundary / exclusion / task
    2) 保留细粒度 token bank，给 cross-attention / dense induction 用
    3) 额外构造 group summary tokens，并放在 text_features 前缀
    4) 向后兼容：仍然输出 text_features，train.py / predict.py 可直接读取
    """
    device = next(model.parameters()).device
    flat_items = _flatten_prompt_dict(prompt_dict)

    token_banks: List[torch.Tensor] = []
    token_group_ids: List[torch.Tensor] = []
    token_prompt_ids: List[torch.Tensor] = []
    token_weights: List[torch.Tensor] = []
    token_texts: List[str] = []

    sentence_embeddings: List[torch.Tensor] = []
    sentence_group_ids: List[int] = []
    sentence_weights: List[float] = []
    flat_prompts: List[str] = []
    token_counts_per_prompt: List[int] = []

    group_sentence_banks: Dict[str, List[torch.Tensor]] = {g: [] for g in GROUP_ORDER}

    hidden_dim: Optional[int] = None

    for prompt_id, (group_name, prompt, group_weight) in enumerate(flat_items):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        last4 = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)  # [1, L, D]

        if hidden_dim is None:
            hidden_dim = int(last4.shape[-1])

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0].bool()

        valid_positions = attention_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        # 去掉 [CLS] / [SEP]
        valid_positions = [i for i in valid_positions if i not in (0, int(attention_mask.sum().item()) - 1)]
        if len(valid_positions) == 0:
            valid_positions = attention_mask.nonzero(as_tuple=False).squeeze(1).tolist()

        raw_token_feats = last4[0, valid_positions, :].detach().cpu()
        kept_feats, kept_texts = _filter_subtokens(
            tokenizer=tokenizer,
            input_ids=input_ids.detach().cpu(),
            token_feats=raw_token_feats,
            valid_positions=valid_positions,
        )

        token_banks.append(kept_feats)
        token_group_ids.append(
            torch.full((kept_feats.shape[0],), GROUP2ID[group_name], dtype=torch.long)
        )
        token_prompt_ids.append(
            torch.full((kept_feats.shape[0],), prompt_id, dtype=torch.long)
        )
        token_weights.append(
            torch.full((kept_feats.shape[0],), float(group_weight), dtype=torch.float)
        )
        token_texts.extend(kept_texts)
        token_counts_per_prompt.append(int(kept_feats.shape[0]))

        sent_emb = masked_mean_pool(last4, inputs["attention_mask"]).squeeze(0).detach().cpu()
        sentence_embeddings.append(sent_emb)
        sentence_group_ids.append(GROUP2ID[group_name])
        sentence_weights.append(float(group_weight))
        flat_prompts.append(prompt)
        group_sentence_banks[group_name].append(sent_emb)

    if hidden_dim is None:
        hidden_dim = int(model.config.hidden_size)

    prompt_embeddings = torch.stack(sentence_embeddings, dim=0)
    global_prompt_summary = prompt_embeddings.mean(dim=0)

    group_summary_tokens = []
    group_summary_weights = []
    group_prompt_counts = []

    for group_name in GROUP_ORDER:
        embs = group_sentence_banks[group_name]
        if len(embs) > 0:
            group_summary = torch.stack(embs, dim=0).mean(dim=0)
        else:
            group_summary = global_prompt_summary.clone()
        group_summary_tokens.append(group_summary.unsqueeze(0))
        group_summary_weights.append(float(GROUP_WEIGHTS[group_name]))
        group_prompt_counts.append(len(embs))

    group_summary_tokens = torch.cat(group_summary_tokens, dim=0)  # [G, D]
    fine_token_features = torch.cat(token_banks, dim=0)            # [N, D]

    # 关键：把 group summary token 放在前缀
    text_features = torch.cat([group_summary_tokens, fine_token_features], dim=0)

    text_mask = torch.ones(text_features.shape[0], dtype=torch.long)
    text_group_summary_count = len(GROUP_ORDER)

    prefix_group_ids = torch.arange(text_group_summary_count, dtype=torch.long)
    prefix_prompt_ids = torch.full((text_group_summary_count,), -1, dtype=torch.long)
    prefix_weights = torch.tensor(group_summary_weights, dtype=torch.float)
    prefix_is_group_summary = torch.ones(text_group_summary_count, dtype=torch.long)

    suffix_is_group_summary = torch.zeros(fine_token_features.shape[0], dtype=torch.long)

    token_group_ids_all = torch.cat([prefix_group_ids, torch.cat(token_group_ids, dim=0)], dim=0)
    token_prompt_ids_all = torch.cat([prefix_prompt_ids, torch.cat(token_prompt_ids, dim=0)], dim=0)
    token_weights_all = torch.cat([prefix_weights, torch.cat(token_weights, dim=0)], dim=0)
    token_is_group_summary = torch.cat([prefix_is_group_summary, suffix_is_group_summary], dim=0)

    token_texts_all = [f"<GROUP:{g}>" for g in GROUP_ORDER] + token_texts

    return {
        # ===== 向后兼容 =====
        "text_features": text_features,                      # [G + N, D]
        "text_mask": text_mask,                              # [G + N]
        "prompt_embeddings": prompt_embeddings,              # [P, D]
        "prompts": flat_prompts,
        "token_counts_per_prompt": token_counts_per_prompt,
        "embedding_type": "structured_prompt_bank_with_group_prefix_v3",

        # ===== 新增元信息 =====
        "group_names": GROUP_ORDER,
        "text_group_summary_count": text_group_summary_count,
        "group_prompt_counts": torch.tensor(group_prompt_counts, dtype=torch.long),
        "group_summary_weights": torch.tensor(group_summary_weights, dtype=torch.float),
        "group_summary_is_prefix": True,

        "token_group_ids": token_group_ids_all,              # [G + N]
        "token_prompt_ids": token_prompt_ids_all,            # [G + N]
        "token_weights": token_weights_all,                  # [G + N]
        "token_is_group_summary": token_is_group_summary,    # [G + N]
        "token_texts": token_texts_all,
        "hidden_dim": hidden_dim,
    }


def generate_and_save_text_embedding(model_path, dataset_name, save_path, extra_prompts=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prompt_dict = {k: list(v) for k, v in PROMPT_BANK[dataset_name].items()}
    if extra_prompts is not None:
        for group_name, group_prompts in extra_prompts.items():
            prompt_dict.setdefault(group_name, [])
            prompt_dict[group_name].extend(group_prompts)

    print(f"1. 加载模型: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    total_prompt_num = sum(len(v) for v in prompt_dict.values())
    print(f"2. 数据集: {dataset_name}")
    print(f"3. Prompt 组: {list(prompt_dict.keys())}")
    print(f"4. Prompt 总数: {total_prompt_num}")

    obj = encode_prompt_bank(
        model=model,
        tokenizer=tokenizer,
        prompt_dict=prompt_dict,
        max_length=32,
    )

    print(f"5. 保存到: {save_path}")
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(obj, save_path)

    print(f"✅ 完成！text_features 形状: {tuple(obj['text_features'].shape)}")
    print(f"✅ prompt_embeddings 形状: {tuple(obj['prompt_embeddings'].shape)}")
    print(f"✅ text_group_summary_count: {obj['text_group_summary_count']}")
    print(f"✅ group_names: {obj['group_names']}")


def build_all(model_path: str, output_dir: str):
    all_datasets = [
        "thyroid",
        "TN3K",
        "BUSI_WHU",
        "BUS-BRA",
    ]
    for dataset_name in all_datasets:
        save_filename = f"text_features_{dataset_name}.pt"
        save_path = os.path.join(output_dir, save_filename)
        generate_and_save_text_embedding(
            model_path=model_path,
            dataset_name=dataset_name,
            save_path=save_path,
            extra_prompts=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate structured medical text features for dense semantic induction."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google-bert/bert-large-uncased",
        help="Path to the BERT model directory or HuggingFace ID.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text_features",
        help="Directory to save the generated .pt files.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="all",
        help="Dataset name: thyroid / TN3K / BUSI_WHU / BUS-BRA / all",
    )
    args = parser.parse_args()

    if args.dataset_name == "all":
        build_all(model_path=args.model_path, output_dir=args.output_dir)
    else:
        if args.dataset_name not in PROMPT_BANK:
            raise ValueError(
                f"Unsupported dataset_name='{args.dataset_name}'. "
                f"Expected one of {list(PROMPT_BANK.keys()) + ['all']}"
            )
        save_filename = f"text_features_{args.dataset_name}.pt"
        save_path = os.path.join(args.output_dir, save_filename)
        generate_and_save_text_embedding(
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            save_path=save_path,
            extra_prompts=None,
        )
