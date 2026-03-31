"""
inference.py — Submission-required inference script for SRE Autopilot.

Uses the OpenAI Client to call an LLM via the following environment variables:
  - API_BASE_URL : The API endpoint for the LLM (e.g., https://api.openai.com/v1)
  - MODEL_NAME   : The model identifier (e.g., gpt-4o-mini)
  - HF_TOKEN     : Your Hugging Face / API key (used as the OpenAI API key)

Runs all 3 task tiers (easy, medium, hard), prints scores in 0.0–1.0 range.
"""

import os
import sys
import json
import re
import time
import requests

from openai import OpenAI

# ──────────────────────────────────────────────────────────
# Environment Variables (required)
# ──────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    print("ERROR: The following environment variables must be set:")
    print("  API_BASE_URL  — The API endpoint for the LLM")
    print("  MODEL_NAME    — The model identifier to use")
    print("  HF_TOKEN      — Your Hugging Face / API key")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# OpenAI Client Setup
# ──────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ──────────────────────────────────────────────────────────
# SRE Environment Server URL
# ──────────────────────────────────────────────────────────
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

SYSTEM_PROMPT = """You are an elite Site Reliability Engineer (SRE) managing a production microservices system.

Your goal is to restore ALL services to SLA compliance (latency < 200ms, error rate < 1%).

The system has these microservices:
- api_gateway (depends on: auth_service, product_service, order_service)
- auth_service (depends on: user_db)
- product_service (depends on: inventory_db)
- order_service (depends on: payment_gateway, inventory_db)
- user_db (leaf)
- inventory_db (leaf)
- payment_gateway (leaf)

Available actions (respond STRICTLY in JSON):
{
  "thought": "your brief diagnosis",
  "action": "restart|scale_up|wait",
  "service_id": "service_name or null"
}

Action details:
- restart: Restarts a service. Costs 1 budget, causes 1-step downtime, but can fix faults.
- scale_up: Doubles capacity of a service. Costs money, takes 3 steps to take effect.
- wait: Observe for 1 step without acting.

Strategy tips:
- Look at error_rate and latency_p99 to find the worst service.
- High memory_pct on a leaf DB often means it's the root cause.
- Restarting a leaf DB can fix cascading failures upstream.
- Don't waste restarts on services that are only failing due to downstream issues."""


def call_llm(observation: dict) -> dict:
    """Call the LLM via OpenAI client and parse the action response."""
    user_prompt = f"""Current system state (step {observation.get('step', '?')}):

Metrics:
{json.dumps(observation.get('metrics', {}), indent=2)}

SLA Status:
{json.dumps(observation.get('sla_status', {}), indent=2)}

Dependency Graph:
{json.dumps(observation.get('dependency_graph', {}), indent=2)}

Recent Actions: {observation.get('action_history', [])}

Reward Hint: {observation.get('reward_hint', 0.0)}

What is your next action? Respond STRICTLY in JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()

        # Try to find a JSON object
        obj_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if obj_match:
            data = json.loads(obj_match.group(0))
        else:
            data = json.loads(content)

        action = data.get("action", "wait")
        service_id = data.get("service_id", None)
        thought = data.get("thought", "")

        # Validate action
        if action not in ("restart", "scale_up", "wait"):
            action = "wait"
            service_id = None

        return {"action": action, "service_id": service_id, "thought": thought}

    except Exception as e:
        print(f"  [LLM Error] {e}. Falling back to wait.")
        return {"action": "wait", "service_id": None, "thought": "LLM fallback"}


def env_reset(tier: str) -> dict:
    """Reset the SRE environment for a given task tier."""
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_tier": tier})
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """Take a step in the SRE environment."""
    payload = {"action": {"action": action["action"], "service_id": action.get("service_id")}}
    resp = requests.post(f"{ENV_BASE_URL}/step", json=payload)
    resp.raise_for_status()
    return resp.json()


def run_grader(tier: str) -> dict:
    """Call the grader endpoint for a given tier."""
    resp = requests.get(f"{ENV_BASE_URL}/grader", params={"tier": tier})
    resp.raise_for_status()
    return resp.json()


def run_episode(tier: str, max_steps: int = 30) -> dict:
    """Run a full inference episode for one task tier."""
    print(f"\n{'='*60}")
    print(f"  TASK: {tier.upper()}")
    print(f"{'='*60}")

    # Reset environment
    result = env_reset(tier)
    observation = result.get("observation", result)

    total_reward = 0.0
    steps_taken = 0

    for step_num in range(max_steps):
        done = observation.get("done", False)
        if done:
            break

        # Get LLM action
        llm_result = call_llm(observation)
        thought = llm_result.get("thought", "")
        action_str = llm_result["action"]
        service = llm_result.get("service_id", "none")

        print(f"  [{step_num:2d}] {action_str:10s} -> {service or 'none':20s} | {thought}")

        # Take step
        step_result = env_step(llm_result)
        observation = step_result.get("observation", step_result)
        reward = step_result.get("reward", observation.get("reward_hint", 0.0))
        total_reward += reward
        steps_taken = step_num + 1

    print(f"  --- Episode complete: {steps_taken} steps, total reward: {total_reward:.3f}")

    # Get final SLA status
    final_sla = observation.get("sla_status", {})
    sla_ok = sum(1 for v in final_sla.values() if v)
    sla_total = len(final_sla) if final_sla else 0
    print(f"  --- Final SLA: {sla_ok}/{sla_total} services healthy")

    return {
        "tier": tier,
        "steps_taken": steps_taken,
        "total_reward": total_reward,
        "final_sla": final_sla,
    }


def main():
    print("=" * 60)
    print("  SRE Autopilot — Inference Script")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  API Base: {API_BASE_URL}")
    print(f"  Env URL:  {ENV_BASE_URL}")
    print("=" * 60)

    tiers = ["easy", "medium", "hard"]
    results = {}

    for tier in tiers:
        episode_result = run_episode(tier)
        results[tier] = episode_result

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Tier':<10} {'Steps':<8} {'Reward':<10} {'SLA'}")
    print(f"  {'-'*45}")
    for tier in tiers:
        r = results[tier]
        sla = r["final_sla"]
        sla_ok = sum(1 for v in sla.values() if v)
        sla_total = len(sla) if sla else 0
        print(f"  {tier:<10} {r['steps_taken']:<8} {r['total_reward']:<10.3f} {sla_ok}/{sla_total}")

    # ── Run graders and print scores ─────────────────────
    print(f"\n{'='*60}")
    print("  GRADER SCORES (0.0 – 1.0)")
    print(f"{'='*60}")
    for tier in tiers:
        try:
            grader_result = run_grader(tier)
            score = grader_result.get("score", "N/A")
            print(f"  {tier:<10} score = {score}")
        except Exception as e:
            print(f"  {tier:<10} grader error: {e}")

    print(f"\n{'='*60}")
    print("  Inference complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
