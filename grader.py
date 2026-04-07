"""
Programmatic graders for SRE Autopilot tasks.

Each grader takes an episode result and returns a score between 0.0 and 1.0.

Scoring criteria per task tier:
  - Easy:   Did the agent restore all services to SLA within budget?
  - Medium: Did the agent identify the root cause DB and fix it (not just symptoms)?
  - Hard:   Did the agent learn the flapping pattern and apply the right fix at the right time?

All graders are deterministic given the same episode history.
"""
from typing import Dict, List, Any


def grade_episode(
    tier: str,
    final_sla_status: Dict[str, bool],
    total_steps: int,
    max_steps: int,
    remaining_restarts: int,
    initial_restarts: int,
    total_cost: float,
    failure_mode: str,
    total_reward: float,
) -> float:
    """
    Grade a completed episode and return a score in [0.0, 1.0].
    
    This is the unified grader called by the /grader endpoint.
    It dispatches to tier-specific logic internally.
    """
    
    # --- Component 1: SLA Compliance (0.0 – 0.5) ---
    # What fraction of services ended within SLA?
    if final_sla_status:
        services_in_sla = sum(1 for v in final_sla_status.values() if v)
        total_services = len(final_sla_status)
        sla_score = services_in_sla / total_services if total_services > 0 else 0.0
    else:
        sla_score = 0.0
    
    sla_component = sla_score * 0.5  # max 0.5

    # --- Component 2: Efficiency (0.0 – 0.25) ---
    # How quickly did the agent resolve? (fewer steps = better)
    speed_ratio = 1.0 - (total_steps / max_steps) if max_steps > 0 else 0.0
    speed_ratio = max(0.0, speed_ratio)
    
    # How budget-efficient? (fewer restarts used = better)
    budget_ratio = remaining_restarts / initial_restarts if initial_restarts > 0 else 1.0
    
    # Cost efficiency (lower cost = better, cap at 10.0 for normalization)
    cost_ratio = max(0.0, 1.0 - (total_cost / 10.0))
    
    efficiency_score = (speed_ratio * 0.4 + budget_ratio * 0.3 + cost_ratio * 0.3)
    efficiency_component = efficiency_score * 0.25  # max 0.25

    # --- Component 3: Tier-Specific Bonus (0.0 – 0.25) ---
    tier_component = 0.0
    
    if tier == "easy":
        # Easy: Full credit if all SLAs are met. Simple.
        tier_component = 0.25 if sla_score == 1.0 else sla_score * 0.15
        
    elif tier == "medium":
        # Medium: Bonus for fixing root cause (not just symptoms)
        # If failure mode changed to "healthy", agent found the root cause
        if failure_mode == "healthy":
            tier_component = 0.25
        elif sla_score >= 0.8:
            tier_component = 0.15  # Partial credit for symptom management
        else:
            tier_component = sla_score * 0.1
            
    elif tier == "hard":
        # Hard: The flapping network is nearly impossible to fully solve
        # Any SLA compliance above 50% is impressive
        if sla_score == 1.0:
            tier_component = 0.25
        elif sla_score >= 0.7:
            tier_component = 0.20
        elif sla_score >= 0.5:
            tier_component = 0.12
        else:
            tier_component = sla_score * 0.08
    
    final_score = round(min(1.0, sla_component + efficiency_component + tier_component), 4)
    # Restrict score heavily to avoid any float rounding issues in the pipeline causing 1.0 or 0.0
    final_score = max(0.01, min(0.99, final_score))
    return float(final_score)


# --- Task Definitions ---

TASKS = [
    {
        "id": "easy",
        "name": "Single Service OOM Recovery",
        "description": (
            "A single service is experiencing an Out-of-Memory crash. "
            "The agent must identify the failing service and restart it to restore SLA. "
            "Tests basic pattern recognition and action execution."
        ),
        "difficulty": "easy",
        "max_steps": 30,
        "success_threshold": 0.7,
    },
    {
        "id": "medium",
        "name": "Cascading Database Failure",
        "description": (
            "A database connection leak is causing cascading latency degradation "
            "across dependent services. Fixing leaf nodes (symptoms) doesn't help; "
            "the agent must diagnose and fix the root-cause database first. "
            "Tests dependency reasoning under partial observability."
        ),
        "difficulty": "medium",
        "max_steps": 30,
        "success_threshold": 0.5,
    },
    {
        "id": "hard",
        "name": "Intermittent Flapping Network",
        "description": (
            "The payment gateway is intermittently flapping: failure appears, "
            "auto-recovers, then reappears. Red herring metrics exist on other services. "
            "The correct fix only works within a narrow timing window. "
            "Requires learning the failure fingerprint over multiple episodes; "
            "pure reasoning fails."
        ),
        "difficulty": "hard",
        "max_steps": 30,
        "success_threshold": 0.3,
    },
]

ACTION_SCHEMA = {
    "action": {
        "type": "string",
        "enum": ["restart", "scale_up", "circuit_break", "rollback", "wait"],
        "description": "The remediation action to take.",
    },
    "service_id": {
        "type": "string",
        "nullable": True,
        "enum": [
            "api_gateway", "auth_service", "product_service",
            "order_service", "user_db", "inventory_db", "payment_gateway",
        ],
        "description": "Target microservice. Required for restart/scale_up, null for wait.",
    },
    "value": {
        "type": "integer",
        "nullable": True,
        "description": "Optional parameter (e.g., number of instances for scale_up).",
    },
}
