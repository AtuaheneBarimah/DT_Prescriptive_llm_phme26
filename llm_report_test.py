import torch
import json
import gc
import sys
import time
import re
from pathlib import Path
from litgpt import LLM

try:
    from report_engine import build_asset_matrix, detect_faults, build_maintenance_plan, calculate_repair_metrics
except ImportError:
    print("✘ ERROR: report_engine.py missing.")
    sys.exit()

from llm_report import checkpoint_model, checkpoint_model_checker


data_root = Path(r"C:\XXX\data")
weights_path = Path(r"C:\XXX\llm_model_weights.pth")

main_report_file = Path("gcu_asset_report_FINAL.txt")
log_file = Path("gcu_asset_report_log.txt")

# Load Data
hr_data = json.load(open(data_root / "human_res.json", 'r', encoding='utf-8'))
dt_analytics = json.load(open(data_root / "dt_analytics.json", 'r', encoding='utf-8'))
report_fmt = json.load(open(data_root / "report_format.json", 'r', encoding='utf-8'))

latest_dt = dt_analytics['dt_analytics'][-1]
asset_matrix = build_asset_matrix(latest_dt, report_fmt)
faulty_list = detect_faults(asset_matrix)

enriched_plan = []
for task in build_maintenance_plan(latest_dt, faulty_list, hr_data):
    task.update(calculate_repair_metrics(task['asset']))
    enriched_plan.append(task)


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_safe_model(checkpoint):
    """Loads model with memory precautions to avoid Segmentation Faults."""
    model = LLM.load(checkpoint)

    model.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.model.to(device)
    return model

print("Loading Generator...")
generator = load_safe_model(checkpoint_model)
print("Loading Discriminator...")
discriminator = load_safe_model(checkpoint_model_checker)



def clean_json_response(raw_text):
    """Extracts JSON and stops the 'gibberish' loops."""

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        clean_text = match.group(0)

        clean_text = re.sub(r",\s*([\]}])", r"\1", clean_text)
        try:
            return json.loads(clean_text)
        except:
            return None
    return None

MAX_RETRIES = 3

for attempt in range(1, MAX_RETRIES + 1):
    print(f"\n[Attempt {attempt}] Generating...")


    gen_prompt = f"""<|system|>
You are a technical report writer. Use the DATA provided. 
DO NOT REPEAT WORDS. DO NOT RAMBLE. 
PROVIDE ONLY A JSON OBJECT.
<|user|>
DATA: {json.dumps(enriched_plan)}

Format:
{{
  "identification": "Short paragraph on faults",
  "actions_and_personnel": "Short paragraph on technicians",
  "timeline_and_budget": "Short paragraph on costs"
}}
<|assistant|>
"""
    response = generator.generate(
        gen_prompt, 
        max_new_tokens=400, 
        temperature=0.1,
        top_p=0.9
    )

    report_data = clean_json_response(response)

    with open(main_report_file, "w", encoding="utf-8") as f:
        f.write(f"GCU MAINTENANCE REPORT - ATTEMPT {attempt}\n" + "="*40 + "\n")
        if report_data:
            f.write(f"1. FAULTS: {report_data.get('identification', 'Error parsing section')}\n\n")
            f.write(f"2. STAFF: {report_data.get('actions_and_personnel', 'Error parsing section')}\n\n")
            f.write(f"3. COSTS: {report_data.get('timeline_and_budget', 'Error parsing section')}\n")
        else:

            f.write("ERROR: System generated invalid format. Retrying...")
        f.flush()

    with open(log_file, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n--- ATTEMPT {attempt} LOG ---\n{response}\n")
        f_log.flush()


    if report_data:
        audit_prompt = f"Does this report mention {enriched_plan[0]['asset']} and its cost? Reply PASSED or NO."
        audit_res = discriminator.generate(audit_prompt, max_new_tokens=50)
        if "PASSED" in audit_res.upper():
            print("✔ Threshold met.")
            break

del generator, discriminator
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Process finished.")
