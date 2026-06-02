import os
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModel

DOMAINS = {
    "thyroid": "thyroid nodule",
    "TN3K": "thyroid nodule",
    "BUSI_WHU": "breast mass",
    "BUS-BRA": "breast mass",
    "OTU": "ovarian tumor",       
    "prostate": "prostate gland"  
}

def main():
    parser = argparse.ArgumentParser(description="Generate EMSD text features for TaskSegmentV3.")
    parser.add_argument("--output-dir", type=str, default="./text_features_llm", help="Directory to save features.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("1. 正在加载 Qwen2-VL-7B-Instruct (生成专家级连贯超声文本先验)...")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto" if torch.cuda.is_available() else None
    )
    qwen_model.eval()
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # 🌟 EMSD 核心修改：让模型自然输出 1 段话，激发 LLM 自连贯性
    prompt_template = (
        "You are an expert radiologist. Describe the general ultrasound imaging "
        "characteristics of a {}. Focus on echogenicity, margin regularity, "
        "and internal structure. Provide the description in 3 concise, professional English sentences."
    )

    domain_texts = {}
    for domain, organ in DOMAINS.items():
        print(f"\n正在让 LLM 生成 [{domain}] 的超声特征描述...")
        messages = [{"role": "user", "content": prompt_template.format(organ)}]
        text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = qwen_processor(text=[text_prompt], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        domain_texts[domain] = output_text.strip()
        print(f"生成的文本:\n{domain_texts[domain]}")

    print("\n2. 正在加载 BioBERT-large...")
    del qwen_model
    del qwen_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    bert_model = AutoModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1", use_safetensors=True).to(device)
    bert_model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n3. 正在编码并保存 EMSD 单锚点密集语义特征...")
    
    # 🌟 EMSD 核心修改：明确宣告整个网络只有一个全局锚点
    GROUP_SUMMARY_COUNT = 1  

    for domain, text in domain_texts.items():
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs, output_hidden_states=True)
            # 融合最后 4 层，提取最稳健的深层表征
            last4 = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
            
            attention_mask = inputs["attention_mask"][0].bool()
            valid_positions = attention_mask.nonzero(as_tuple=False).squeeze(1)
            
            if valid_positions.numel() >= 3:
                content_positions = valid_positions[1:-1]
            else:
                content_positions = valid_positions
                
            # 🌟 显式解耦：绝对纯净的 [CLS] + 连续的细微自然语言序列
            global_feat = last4[0, 0, :].unsqueeze(0).cpu()       # [1, 1024]
            fine_feats = last4[0, content_positions, :].cpu()     # [N, 1024]
            
        final_text_features = torch.cat([global_feat, fine_feats], dim=0) # [1 + N, 1024]
        total_tokens = final_text_features.shape[0]
        
        text_mask = torch.ones(total_tokens, dtype=torch.long)
        token_is_group_summary = torch.cat([
            torch.ones(GROUP_SUMMARY_COUNT, dtype=torch.long),
            torch.zeros(fine_feats.shape[0], dtype=torch.long)
        ], dim=0)
        
        save_dict = {
            "text_features": final_text_features,               
            "text_mask": text_mask,                              
            "text_group_summary_count": GROUP_SUMMARY_COUNT,     
            "token_is_group_summary": token_is_group_summary,    
            "raw_text": text,                                    
            "hidden_dim": 1024                                   
        }
            
        save_path = os.path.join(args.output_dir, f"text_features_{domain}.pt")
        torch.save(save_dict, save_path)
        print(f"已保存 [{domain}] 特征 -> Shape: {final_text_features.shape}")

    print("\n✅ EMSD 文本特征生成完毕！")

if __name__ == "__main__":
    main()