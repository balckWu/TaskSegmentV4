import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. 正在加载 Qwen2-VL-7B-Instruct (生成专家级超声文本先验)...")
    # 使用 bfloat16 精度，完美适配 RTX 4090 的算力与显存
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # 🌟 针对当前框架的 6 个域设定特定的病灶目标
    domains = {
        "thyroid": "thyroid nodule",
        "TN3K": "thyroid nodule",
        "BUSI_WHU": "breast mass",
        "BUS-BRA": "breast mass",
        "OTU": "ovarian tumor",       # 新加入的 OTU
        "prostate": "prostate gland"  # 新加入的 前列腺
    }

    prompt_template = (
        "You are an expert radiologist. Describe the general ultrasound imaging "
        "characteristics of a {}. Focus on echogenicity, margin regularity, "
        "and internal structure. Provide the description in 3 concise, professional English sentences."
    )

    domain_texts = {}
    for domain, organ in domains.items():
        print(f"\n正在让 LLM 生成 [{domain}] 的超声特征描述...")
        messages = [{"role": "user", "content": prompt_template.format(organ)}]
        text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = qwen_processor(text=[text_prompt], return_tensors="pt").to(device)
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
        
        # 提取生成的回复文本
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        domain_texts[domain] = output_text.strip()
        print(f"生成的文本: {domain_texts[domain]}")

    print("\n2. 正在加载 BioBERT-large (将文本编码为 1024 维安全格式特征)...")
    # 释放 Qwen 占用的显存，防止两张大模型同时挤爆显存
    del qwen_model
    del qwen_processor
    torch.cuda.empty_cache()

    # 加载原生输出 1024 维的顶级医学文本编码器，并强制只读安全的 safetensors 格式
    bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    bert_model = AutoModel.from_pretrained(
        "dmis-lab/biobert-large-cased-v1.1",
        use_safetensors=True  # 🔥 强制跳过 .bin 文件，彻底解决 PyTorch 安全报错
    ).to(device)
    bert_model.eval()

    output_dir = "./text_features_llm"
    os.makedirs(output_dir, exist_ok=True)

    print("\n3. 正在提取 5 个明确语义维度的 Group Summary 锚点...")
    # ==========================================
    # 兼容性设定与语义锚点构造
    # ==========================================
    GROUP_ORDER = ["anatomy", "appearance", "boundary", "exclusion", "task"]
    GROUP_SUMMARY_COUNT = len(GROUP_ORDER)
    
    group_summary_list = []
    for group_name in GROUP_ORDER:
        g_inputs = bert_tokenizer(group_name, return_tensors="pt").to(device)
        with torch.no_grad():
            g_outputs = bert_model(**g_inputs)
            # 提取每个维度单词专属的 [CLS] 向量，作为独立的 Summary
            group_summary_list.append(g_outputs.last_hidden_state[0, 0, :].cpu())
    
    # 提前将这 5 个具有明确物理意义的向量堆叠: Shape -> [5, 1024]
    group_summary_tokens = torch.stack(group_summary_list, dim=0)

    print("\n4. 正在编码并保存向下兼容的密集语义特征 .pt 文件...")
    for domain, text in domain_texts.items():
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            
            # 获取所有 token 的特征: [1, Seq_len, 1024]
            last_hidden_state = outputs.last_hidden_state[0].cpu() 
            attention_mask = inputs["attention_mask"][0].cpu().bool()
            
            # 过滤掉 padding，仅保留有效 token 的特征
            valid_feats = last_hidden_state[attention_mask] # Shape: [N, 1024]
            
            # 提取句子本身的密集细粒度特征 (去掉首位的 [CLS] 和末位的 [SEP])
            fine_feats = valid_feats[1:-1]                  
            
        # ==========================================
        # 结构组装：桥接 LLM 特征与现有的 Dense Guidance 架构
        # ==========================================
        # 拼接成最终模型期望的格式: 前 5 个是具有区分度的独立语义 Summary, 后面跟着 LLM 生成的细粒度特征
        final_text_features = torch.cat([group_summary_tokens, fine_feats], dim=0) # [5 + 细粒度Token数, 1024]
        total_tokens = final_text_features.shape[0]
        
        # 构造向下兼容的伪造掩码与索引，供 Attention 层使用
        text_mask = torch.ones(total_tokens, dtype=torch.long)
        token_is_group_summary = torch.cat([
            torch.ones(GROUP_SUMMARY_COUNT, dtype=torch.long),
            torch.zeros(fine_feats.shape[0], dtype=torch.long)
        ], dim=0)
        
        # 组装字典，匹配旧版 prompt_encoder 返回的核心 key
        save_dict = {
            "text_features": final_text_features,                # 核心特征矩阵
            "text_mask": text_mask,                              # Attention 掩码
            "text_group_summary_count": GROUP_SUMMARY_COUNT,     # 数量标识，防止 DataLoader 截断报错
            "token_is_group_summary": token_is_group_summary,    # 区分前缀与细粒度特征
            "raw_text": text,                                    # 附加原始 LLM 文本，方便分析
            "hidden_dim": 1024                                   # 声明特征维度
        }
            
        save_path = os.path.join(output_dir, f"text_features_{domain}.pt")
        torch.save(save_dict, save_path)
        print(f"已保存 [{domain}] 特征 -> {save_path} | 特征矩阵 Shape: {final_text_features.shape}")

    print("\n✅ 1024维安全格式密集文本特征 (语义锚点融合版) 生成完毕，且已完全适配 Dense Guidance 架构！")

if __name__ == "__main__":
    main()