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

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
DATA_INPUT = "data/figstep_samples.jsonl"
RESULT_OUT = "results/defense_results_v3.jsonl"
N_PATCHES = 16
K_MASK = 3
USE_AMIA = True

def load_my_data():
    print(f"Loading data from: {DATA_INPUT}")
    all_items = []
    f = open(DATA_INPUT, 'r', encoding='utf-8')
    for line in f:
        all_items.append(json.loads(line))
    f.close()
    print(f"Successfully loaded {len(all_items)} samples.")
    return all_items

def run_experiment():
    print("-" * 40)
    print("VLM Defense Experiment Start!")
    print(f"Time: {time.ctime()}")
    print("-" * 40)

    print(">>> Loading VLM Model...")
    vlm = LLaVAWrapper(model_name_or_path=MODEL_PATH, device="cuda")

    if USE_AMIA:
        print(f">>> AMIA Enabled: N={N_PATCHES}, K={K_MASK}")
        defender = AMIA(lvlm=vlm, n_patches=N_PATCHES, k_mask=K_MASK)
    else:
        print(">>> Warning: Running Baseline (No Defense)")
        defender = None

    samples = load_my_data()
    
    # 如果没建目录就建一个
    if not os.path.exists("results"):
        os.makedirs("results")

    save_list = []
    total_safe = 0
    total_num = len(samples)

    print("\nStarting inference pipeline...")
    for i in range(total_num):
        s = samples[i]
        img_path = s["image"]
        prompt = s["text"]

        print(f"[{i+1}/{total_num}] Processing: {img_path}")
        
        img = Image.open(img_path).convert("RGB")

        if USE_AMIA:
            # 这里直接调用防御接口
            res = defender.defend(img, prompt)
            ans = res["final_response"]
            reason = res["intention_analysis"]
        else:
            ans = vlm.generate(img, prompt)
            reason = "Baseline-No-Analysis"

        # 判断安全性
        is_safe = is_safe_response(ans)
        if is_safe:
            total_safe += 1
        
        # 实时打印结果方便观察
        print(f"  > Safety: {is_safe}")
        print(f"  > Model Response: {ans[:50]}...") # 只印前50个字符

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

    print("\n" + "#" * 40)
    print("实验统计结果 (Summary)")
    print("-" * 40)

    dsr = (total_safe / total_num) * 100
    
    print(f"测试模型: {MODEL_PATH}")
    print(f"样本总数: {total_num}")
    print(f"安全响应: {total_safe}")
    print(f"防御成功率 (DSR): {dsr:.2f}%")
    
    print("\n详细清单:")
    print("ID | Result | Query Preview")
    for r in save_list[:10]: # 只展示前10条
        res_str = "PASS" if r["safe"] else "FAIL"
        print(f"{r['idx']:02d} | {res_str} | {r['query'][:30]}...")
    
    print("-" * 40)
    print(f"实验完成时间: {time.ctime()}")
    print("#" * 40)

if __name__ == "__main__":
    run_experiment()
