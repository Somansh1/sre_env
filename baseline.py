import os
import json
import time
import re
from sre_env import SREEnv, SREAction

def create_prompt(obs):
    sys_prompt = """You are an elite Site Reliability Engineer.
Your goal is to restore 100% of the microservices to SLA (<200ms latency, <1% errors).
You can take 3 actions:
- {"action": "restart", "service_id": "api_gateway"} (restarts a service, costs 1 budget, 1-step downtime but fixes some faults)
- {"action": "scale_up", "service_id": "auth_service"} (doubles capacity, lowers latency, costs money)
- {"action": "wait", "service_id": null} (wait 1 step to observe)

Respond STRICTLY in JSON:
{
  "thought": "your diagnosis",
  "action": "restart|scale_up|wait",
  "service_id": "name of service or null"
}"""

    user_prompt = f"""Current observation:
Metrics: {json.dumps(obs.metrics, indent=2)}
SLA Status: {json.dumps(obs.sla_status, indent=2)}
Action History: {obs.action_history}

What is your next action?"""

    return sys_prompt, user_prompt

def run_baseline():
    """Run baseline inference over the SRE Autopilot using a LOCAL Qwen model via transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Please install torch and transformers: pip install torch transformers")
        return

    env = SREEnv(base_url="http://localhost:8000")
    
    # We follow the OpenEnv notebook5 pattern: local LLM, no API keys
    model_name = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    
    print(f"Loading local model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print("Model loaded successfully.\\n")

    for tier in ["easy", "medium", "hard"]:
        print(f"=== Testing {tier.upper()} Tier ===")
        with env.sync() as sync_env:
            result = sync_env.reset(task_tier=tier)
            obs = result.observation

            while not obs.done:
                sys_prompt, user_prompt = create_prompt(obs)
                
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                
                model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
                generated_ids = model.generate(**model_inputs, max_new_tokens=128, temperature=0.1, do_sample=False)
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

                try:
                    # Clean up the output to find JSON
                    clean_text = generated_text.strip()
                    match = re.search(r"\{.*?\}", clean_text.replace('\n', ''))
                    if match:
                        data = json.loads(match.group(0))
                    else:
                        data = json.loads(clean_text)
                        
                    act_str = data.get("action", "wait")
                    srv = data.get("service_id", None)
                    thought = data.get("thought", "No thought provided.")
                    action = SREAction(action=act_str, service_id=srv)
                    
                except Exception as e:
                    print(f"Failed LLM inference parsing: {e}. Output was: {generated_text}. Falling back to wait.")
                    action = SREAction(action="wait")
                    thought = "Fallback"

                print(f"[{obs.step}] LLM Thought: {thought}")
                print(f"[{obs.step}] Action: {action.action} on {action.service_id or 'none'} | Reward: {obs.reward}")
                
                result = sync_env.step(action)
                obs = result.observation
                time.sleep(0.1)

            print(f"Finished {tier.upper()} Task. Final Reward: {obs.reward}\n")
        
    env.close()

if __name__ == "__main__":
    print("SRE Autopilot Local Qwen LLM Baseline Agent (No API Key)")
    run_baseline()
