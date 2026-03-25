import torch
import json
import gc
import sys
import time
import re
from pathlib import Path
from litgpt import LLM

try:
    from report_engine import (
        build_asset_matrix, 
        detect_faults, 
        build_maintenance_plan, 
        calculate_repair_metrics,
        convert_format_a_to_b 
    )
except ImportError:
    print("✘ ERROR: all required functions imported.")
    sys.exit()

from llm_report import checkpoint_model, checkpoint_model_checker

data_root = Path(r"C:\XXX\data")
weights_path = Path(r"C:\XXX\llm_model_weights.pth")

main_report_file = Path("gcu_asset_report_FINAL.txt")
history_log_file = data_root / "gcu_historical_log.txt"

raw_data_path = data_root / "dt_analytics_RAW.txt"
if raw_data_path.exists():
    print("✔ Txt format Found. Converting Format A to B...")
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        dt_analytics = convert_format_a_to_b(f.read())
else:
    print("Using existing dt_analytics.json...")
    dt_analytics = json.load(open(data_root / "dt_analytics.json", 'r', encoding='utf-8'))

latest_dt = dt_analytics['dt_analytics'][-1]
report_fmt = json.load(open(data_root / "report_format.json", 'r', encoding='utf-8'))
asset_matrix = build_asset_matrix(latest_dt, report_fmt)
faulty_list = detect_faults(asset_matrix)

is_faulty = len(faulty_list) > 0
report_payload = []

if is_faulty:
    hr_data = json.load(open(data_root / "human_res.json", 'r', encoding='utf-8'))
    for task in build_maintenance_plan(latest_dt, faulty_list, hr_data):
        task.update(calculate_repair_metrics(task['asset']))
        report_payload.append(task)
    mode_title = "CRITICAL MAINTENANCE REQUIRED"
else:
    report_payload = asset_matrix 
    mode_title = "SYSTEM HEALTH & PERFORMANCE REPORT"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_safe_model(checkpoint):
    model = LLM.load(checkpoint)
    model.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.model.to(device)
    return model

print(f"Initializing Models for {mode_title} Mode...")
generator = load_safe_model(checkpoint_model)
discriminator = load_safe_model(checkpoint_model_checker)

MAX_RETRIES = 3

for attempt in range(1, MAX_RETRIES + 1):
    print(f"\n[Attempt {attempt}] Generating Report...")

    instruction = (
        "You are a maintenance engineer identifying faults." if is_faulty 
        else "You are a reliability engineer. Systems are healthy. Summarize performance."
    )

    gen_prompt = f"""<|system|>
{instruction}
STRICT RULE: Do not repeat sentences. Return ONLY a valid JSON object.
<|user|>
DATA: {json.dumps(report_payload)}

Required JSON Format:
{{
  "identification": "Status overview",
  "analysis": "Asset performance metrics",
  "conclusion": "Final recommendation"
}}
<|assistant|>
{{"""

    response = generator.generate(
        "{" + gen_prompt, 
        max_new_tokens=450, 
        temperature=0.05, 
        top_p=0.9
    )

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            report_data = json.loads(match.group(0))
            
            full_report_text = (
                f"OFFICIAL GCU ASSET REPORT: {mode_title}\n"
                f"{'='*60}\n"
                f"GENERATED: {time.ctime()}\n\n"
                f"SECTION 1: STATUS OVERVIEW\n{report_data.get('identification')}\n\n"
                f"SECTION 2: PERFORMANCE ANALYSIS\n{report_data.get('analysis')}\n\n"
                f"SECTION 3: RECOMMENDATION\n{report_data.get('conclusion')}\n"
                f"{'-'*60}\n\n"
            )

            with open(main_report_file, "w", encoding="utf-8") as f_live:
                f_live.write(full_report_text)
                f_live.flush()

            
            with open(history_log_file, "a", encoding="utf-8") as f_hist:
                f_hist.write(full_report_text)
                f_hist.flush()
            
            
            audit_prompt = f"Does the text reflect a {mode_title}? Reply PASSED or NO."
            audit_res = discriminator.generate(audit_prompt, max_new_tokens=50)
            
            if "PASSED" in audit_res.upper():
                print(f"✔ Audit Passed. Files updated successfully.")
                break
            else:
                print(f"✘ Audit Failed (Attempt {attempt}). Content: {audit_res}")

        except Exception as e:
            print(f"✘ Extraction error: {e}. Retrying...")


del generator, discriminator
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"Process Finalized. View history in {history_log_file.name}")
