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
    print("✘ ERROR: Update report_engine.py with the new helper function first.")
    sys.exit()

from llm_report import checkpoint_model, checkpoint_model_checker

data_dir = Path(r"C:\XXX\data")
weights_path = Path(r"C:\XXX\llm_model_weights.pth")

main_report_file = Path("gcu_asset_report_FINAL.txt")
log_file = Path("gcu_asset_report_log.txt")

try:
    with open(data_dir / "dt_analytics_RAW.txt", 'r', encoding='utf-8') as f:
        raw_format_a = f.read()
    
    dt_analytics = convert_format_a_to_b(raw_format_a)
    print(f"✔ Converted Format A to B: {len(dt_analytics['dt_analytics'])} records found.")
except FileNotFoundError:
    print("✘ RAW Data file not found. Ensure dt_analytics_RAW.txt exists.")
    sys.exit()

hr_data = json.load(open(data_dir / "human_res.json", 'r', encoding='utf-8'))
report_fmt = json.load(open(data_dir / "report_format.json", 'r', encoding='utf-8'))

latest_dt = dt_analytics['dt_analytics'][-1]
asset_matrix = build_asset_matrix(latest_dt, report_fmt)
faulty_list = detect_faults(asset_matrix)

enriched_plan = []
for task in build_maintenance_plan(latest_dt, faulty_list, hr_data):
    task.update(calculate_repair_metrics(task['asset']))
    enriched_plan.append(task)

device = "cuda" if torch.cuda.is_available() else "cpu"

def safe_load(checkpoint):

    llm = LLM.load(checkpoint)
    llm.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    llm.model.to(device)
    return llm

print("Initializing Models...")
generator = safe_load(checkpoint_model)
discriminator = safe_load(checkpoint_model_checker)

MAX_RETRIES = 3
current_feedback = "None"

for attempt in range(1, MAX_RETRIES + 1):
    print(f"\n[Attempt {attempt}] Generating...")

    gen_prompt = f"""<|system|>
You are a technical engineer. Convert the DATA into a JSON report.
Rule 1: Use 3 paragraphs.
Rule 2: Do not repeat sentences.
Rule 3: Use EXACT costs from DATA.
<|user|>
DATA: {json.dumps(enriched_plan)}
ERROR TO FIX: {current_feedback}
<|assistant|>
{{"""

    response = generator.generate(
        "{" + gen_prompt, 
        max_new_tokens=450, 
        temperature=0.1, 
        repetition_penalty=1.2,
        top_p=0.9
    )

    report_data = None
    match = re.search(r'(\{.*?\})', response, re.DOTALL)
    if match:
        try:
            report_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            report_data = None

    with open(main_report_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"GCU MAINTENANCE REPORT | ATTEMPT {attempt}\n" + "="*50 + "\n\n")
        if report_data:
            f_out.write(f"1. FAULT IDENTIFICATION\n{report_data.get('identification', 'No data')}\n\n")
            f_out.write(f"2. STAFFING\n{report_data.get('actions_and_personnel', 'No data')}\n\n")
            f_out.write(f"3. COSTS & DOWNTIME\n{report_data.get('timeline_and_budget', 'No data')}\n")
        else:
            f_out.write("CRITICAL ERROR: Model output was malformed. System retrying...")
        f_out.flush()

    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n--- ATTEMPT {attempt} RAW ---\n{response}\n")
        f_log.flush()

    if report_data:
        audit_prompt = f"Does the text identify the correct assets and costs? Reply PASSED or list errors."
        current_feedback = discriminator.generate(audit_prompt, max_new_tokens=100).strip()
        
        if "PASSED" in current_feedback.upper():
            print("✔ Success: Quality threshold met.")
            break
        else:
            print(f"✘ Rejected: {current_feedback}")

del generator, discriminator
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Process Finalized. View result in gcu_asset_report_FINAL.txt")
