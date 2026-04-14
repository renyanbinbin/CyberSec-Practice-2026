import os
import sys
import json
import time
import torch
from PIL import Image

sys.path.append("../../") 

from amia import AMIA
from models import LLaVAWrapper
from evaluation.metrics import is_safe_response

# 配置文件路径
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
DATA_INPUT = "data/figstep_samples.jsonl"
RESULT_OUT = "results/defense_results_v3.jsonl"

# 防御参数
N_PATCHES = 16
K_MASK = 3
USE_AMIA = True

def load_data():
    print(f"Loading data from: {DATA_INPUT}")
    all_items = []
    f = open(DATA_INPUT, 'r', encoding='utf-8')
    for line in f:
        all_items.append(json.loads(line))
    f.close()
    print(f"Loaded {len(all_items)} samples.")
    return all_items

def main():
    print("-" * 40)
    print("Start VLM Defense Exp...")
    print(f"Time: {time.ctime()}")
    print("-" * 40)

    # 加载模型
    print(">>> Loading VLM Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" 
    vlm = LLaVAWrapper(model_name_or_path=MODEL_PATH, device=device)

    if USE_AMIA:
        print(f">>> AMIA Enabled: N={N_PATCHES}, K={K_MASK}")
        defender = AMIA(lvlm=vlm, n_patches=N_PATCHES, k_mask=K_MASK)
    else:
        print(">>> Warning: Running Baseline (No Defense)")
        defender = None

    samples = load_data()
    
    if not os.path.exists("results"):
        os.makedirs("results")

    save_list = []
    total_safe = 0
    total_num = len(samples)

    print("\nStart inference...")
    
    # debug_idx = 5  
    for i in range(total_num):
        # if i != debug_idx: continue # 调试
        
        s = samples[i]
        img_path = s["image"]
        prompt = s["text"]

        print(f"[{i+1}/{total_num}] Processing: {img_path}")
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Error loading image, skipping...")
            continue

        if USE_AMIA:
            res = defender.defend(img, prompt)
            ans = res["final_response"]
            reason = res["intention_analysis"]
            # print(f"DEBUG AMIA: {reason}")
        else:
            ans = vlm.generate(img, prompt)
            reason = "Baseline"

        is_safe = is_safe_response(ans)
        if is_safe:
            total_safe += 1
        
        # 打印结果
        print(f"  > Safety: {is_safe}")
        # print(f"  > Response: {ans}")
        print(f"  > Output: {ans[:50]}...") 

        log_entry = {
            "idx": i,
            "img": img_path,
            "query": prompt,
            "output": ans,
            "analysis": reason,
            "safe": is_safe
        }
        save_list.append(log_entry)

        out_f = open(RESULT_OUT, 'a', encoding='utf-8')
        out_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        out_f.close()

    # 统计结果
    print("\n" + "#" * 40)
    print("Summary")
    print("-" * 40)

    dsr = (total_safe / total_num) * 100
    
    print(f"Model: {MODEL_PATH}")
    print(f"Total: {total_num}")
    print(f"Safe: {total_safe}")
    print(f"DSR: {dsr:.2f}%")
    
    print("\nPreview:")
    for r in save_list[:5]: 
        res_str = "PASS" if r["safe"] else "FAIL"
        print(f"{r['idx']} | {res_str}")
    
    print("-" * 40)
    print(f"Done: {time.ctime()}")
    print("#" * 40)

if __name__ == "__main__":
    main()
