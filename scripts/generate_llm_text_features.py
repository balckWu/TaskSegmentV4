import os
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModel


SUMMARY_KEYS = ("target", "location", "appearance_boundary")
GROUP_SUMMARY_NAMES = (
    "global_target",
    "global_plus_location",
    "global_plus_appearance_boundary",
)
GROUP_SUMMARY_COUNT = len(GROUP_SUMMARY_NAMES)
GLOBAL_ANCHOR_WEIGHT = 0.8
GROUP_VARIATION_WEIGHT = 0.2


def _fallback_summaries(spec):
    """Fallback summaries used when the LLM output is not perfectly parseable."""
    return {
        "target": f"segment the {spec['target']}",
        "location": spec["location"],
        "appearance_boundary": spec["appearance_boundary"],
    }


def _parse_three_summaries(raw_text, spec):
    """
    Parse three LLM summaries before the anchor/variation blend is built.

    Important: these are not three repeated copies of one global CLS token.
    The target, location, and appearance_boundary texts are encoded separately
    by BioBERT first. The final prefix tokens are then constructed as:
      1) global target anchor
      2) global target anchor + small location variation
      3) global target anchor + small appearance/boundary variation
    """
    summaries = _fallback_summaries(spec)
    key_aliases = {
        "target": "target",
        "goal": "target",
        "object": "target",
        "location": "location",
        "anatomy": "location",
        "site": "location",
        "boundary": "appearance_boundary",
        "appearance": "appearance_boundary",
        "appearance_boundary": "appearance_boundary",
        "boundary_appearance": "appearance_boundary",
        "margin": "appearance_boundary",
    }

    for line in raw_text.splitlines():
        line = line.strip().strip("-* ")
        if not line or ":" not in line:
            continue
        left, right = line.split(":", 1)
        left = re.sub(r"[^a-zA-Z_ ]", "", left).strip().lower().replace(" ", "_")
        right = right.strip()
        if not right:
            continue
        if left in key_aliases:
            summaries[key_aliases[left]] = right

    return summaries


def _encode_text(bert_tokenizer, bert_model, text, device):
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state[0].cpu()
        attention_mask = inputs["attention_mask"][0].cpu().bool()
        valid_feats = last_hidden_state[attention_mask]

    cls_feat = valid_feats[0].unsqueeze(0)  # [1, 1024]
    fine_feats = valid_feats[1:-1]          # remove [CLS] and [SEP]
    return cls_feat, fine_feats


def _build_global_anchor_variation_tokens(summary_cls_by_key):
    """Build 3 prefix tokens as global anchor + small group variations.

    This keeps the first token as the stable target prior and makes the other
    two tokens target-aware instead of letting location or appearance act as
    standalone semantic anchors. It is not global_cls.repeat(3, 1).
    """
    global_cls = summary_cls_by_key["target"]
    location_cls = summary_cls_by_key["location"]
    appearance_cls = summary_cls_by_key["appearance_boundary"]

    location_token = (
        GLOBAL_ANCHOR_WEIGHT * global_cls
        + GROUP_VARIATION_WEIGHT * location_cls
    )
    appearance_token = (
        GLOBAL_ANCHOR_WEIGHT * global_cls
        + GROUP_VARIATION_WEIGHT * appearance_cls
    )

    return torch.cat([global_cls, location_token, appearance_token], dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载 Qwen2-VL-7B-Instruct (生成 3 个 summary，并构造全局锚点+分组差异)...")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # 每个 domain 生成 3 个 summary：目标是什么 / 目标在哪里 / 边界或外观是什么样。
    # 注意：不生成 exclusion，不生成 task，也不把一个 CLS 复制 3 遍。
    # 最终前缀 token 会做成 global anchor + small group variation。
    domains = {
        "thyroid": {
            "target": "thyroid nodule",
            "location": "lesion located within the thyroid gland in ultrasound imaging",
            "appearance_boundary": "oval or irregular hypoechoic nodule with variable margin clarity and internal echotexture",
        },
        "TN3K": {
            "target": "thyroid nodule",
            "location": "lesion located within the thyroid gland in ultrasound imaging",
            "appearance_boundary": "solid or mixed thyroid nodule with distinguishable margin, echogenicity, and internal structure",
        },
        "BUSI_WHU": {
            "target": "breast mass",
            "location": "mass located in breast ultrasound tissue surrounded by glandular or fatty tissue",
            "appearance_boundary": "hypoechoic breast lesion with round, oval, lobulated, or irregular boundary and heterogeneous texture",
        },
        "BUS-BRA": {
            "target": "breast mass",
            "location": "mass located in breast ultrasound tissue surrounded by normal breast parenchyma",
            "appearance_boundary": "breast mass with variable margin sharpness, posterior echo pattern, and internal echogenicity",
        },
        "OTU": {
            "target": "ovarian tumor",
            "location": "tumor region located in ovarian ultrasound imaging near adnexal tissue",
            "appearance_boundary": "cystic, solid, or mixed ovarian lesion with visible wall, septation, or irregular contour",
        },
        "prostate": {
            "target": "prostate gland",
            "location": "prostate region located in pelvic or transrectal ultrasound imaging",
            "appearance_boundary": "glandular prostate region with smooth or partially indistinct contour and heterogeneous internal echotexture",
        },
    }

    prompt_template = (
        "You are an expert radiologist preparing text priors for ultrasound image segmentation.\n"
        "Target object: {target}\n"
        "Known anatomical context: {location}\n"
        "Known boundary or appearance context: {appearance_boundary}\n\n"
        "Return exactly three concise English lines and nothing else, using this format:\n"
        "target: what should be segmented\n"
        "location: where the target is anatomically located\n"
        "appearance_boundary: what the boundary, margin, shape, echogenicity, or texture looks like\n\n"
        "Do not mention background exclusion. Do not mention generic task instructions."
    )

    domain_summaries = {}
    raw_llm_texts = {}
    for domain, spec in domains.items():
        print(f"\n正在让 LLM 生成 [{domain}] 的 3 个 summary...")
        messages = [{"role": "user", "content": prompt_template.format(**spec)}]
        text_prompt = qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = qwen_processor(text=[text_prompt], return_tensors="pt").to(device)
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=160)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0].strip()

        raw_llm_texts[domain] = output_text
        domain_summaries[domain] = _parse_three_summaries(output_text, spec)
        print("LLM 原始输出:")
        print(output_text)
        print("解析后的 3 个 summary:")
        for key in SUMMARY_KEYS:
            print(f"  {key}: {domain_summaries[domain][key]}")

    print("\n2. 正在加载 BioBERT-large (分别编码 summary，再构造锚点+差异 token)...")
    del qwen_model
    del qwen_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    bert_model = AutoModel.from_pretrained(
        "dmis-lab/biobert-large-cased-v1.1",
        use_safetensors=True,
    ).to(device)
    bert_model.eval()

    output_dir = "./text_features_llm"
    os.makedirs(output_dir, exist_ok=True)

    print("\n3. 正在保存 global-anchor + variation 文本特征 .pt 文件...")
    for domain, summaries in domain_summaries.items():
        # 关键点：三个 summary 先分别编码，再构造“全局锚点 + 分组差异”。
        # 不是 global_cls.repeat(3, 1)，也不是让 location/appearance 单独作为主锚点。
        summary_cls_by_key = {}
        for key in SUMMARY_KEYS:
            cls_feat, _ = _encode_text(bert_tokenizer, bert_model, summaries[key], device)
            summary_cls_by_key[key] = cls_feat
        group_summary_tokens = _build_global_anchor_variation_tokens(summary_cls_by_key)  # [3, 1024]

        # dense guidance 的 fine tokens 使用三条 summary 合并后的细粒度 token。
        # 这样 prior map 只关注 target/location/appearance，不混入 exclusion/task。
        combined_text = " ".join(summaries[key] for key in SUMMARY_KEYS)
        _, fine_feats = _encode_text(bert_tokenizer, bert_model, combined_text, device)

        final_text_features = torch.cat([group_summary_tokens, fine_feats], dim=0)
        total_tokens = final_text_features.shape[0]

        text_mask = torch.ones(total_tokens, dtype=torch.long)
        token_is_group_summary = torch.cat([
            torch.ones(GROUP_SUMMARY_COUNT, dtype=torch.long),
            torch.zeros(fine_feats.shape[0], dtype=torch.long),
        ], dim=0)

        save_dict = {
            "text_features": final_text_features,
            "text_mask": text_mask,
            "text_group_summary_count": GROUP_SUMMARY_COUNT,
            "token_is_group_summary": token_is_group_summary,
            "group_summary_names": list(GROUP_SUMMARY_NAMES),
            "summary_texts": dict(summaries),
            "summary_source_keys": list(SUMMARY_KEYS),
            "summary_token_strategy": "global_anchor_plus_group_variation",
            "summary_token_formula": {
                "global_target": "target_cls",
                "global_plus_location": "0.8 * target_cls + 0.2 * location_cls",
                "global_plus_appearance_boundary": "0.8 * target_cls + 0.2 * appearance_boundary_cls",
            },
            "raw_text": combined_text,
            "raw_llm_text": raw_llm_texts.get(domain, ""),
            "hidden_dim": 1024,
        }

        save_path = os.path.join(output_dir, f"text_features_{domain}.pt")
        torch.save(save_dict, save_path)
        print(
            f"已保存 [{domain}] -> {save_path} | "
            f"summary_count={GROUP_SUMMARY_COUNT} | shape={tuple(final_text_features.shape)}"
        )

    print("\n✅ 3 个 summary 特征生成完毕：global target / global+location / global+appearance_boundary")


if __name__ == "__main__":
    main()
