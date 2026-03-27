from typing import Dict, List, Optional
from openenv.core.env_server.interfaces import Action, Observation, State
from pydantic import BaseModel, Field

class SREAction(Action):
    """
    Action for SRE Autopilot.
    action: "restart", "scale_up", "rollback", "circuit_break", "reroute", "wait"
    service_id: The target microservice (e.g., "frontend", "auth", "db")
    value: Optional int parameter (e.g., instances to scale to, or target service for reroute)
    """
    action: str
    service_id: Optional[str] = None
    value: Optional[int] = None


class SREReward(BaseModel):
    """
    Structured reward breakdown for SRE Autopilot.
    """
    sla_compliance: float = Field(description="Reward from services staying within SLA (primary signal)")
    cost_penalty: float = Field(description="Penalty from scaling operations and infrastructure costs")
    downtime_penalty: float = Field(description="Penalty from service restarts causing temporary downtime")
    resolution_bonus: float = Field(description="Bonus for full system health at episode end")
    total: float = Field(description="Summed total reward for this step")

class SREObservation(Observation):
    """
    Observation for SRE Autopilot.
    """
    metrics: Dict[str, Dict[str, float]]  # e.g., {"frontend": {"latency": 120.5, "error_rate": 0.05, "cpu": 45.0}}
    dependency_graph: Dict[str, List[str]]  # e.g., {"frontend": ["auth", "docs"]}
    action_history: List[str]
    step: int
    sla_status: Dict[str, bool]
    reward_hint: float = 0.0
    reward_breakdown: Optional[SREReward] = None

class SREState(State):
    """
    True hidden state for SRE Autopilot.
    """
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    last_action: str = "reset"
    
    # Latent Failure State
    failure_mode: str = "healthy"
    root_cause_service: Optional[str] = None
    
    # Service internals (not shown directly to agent, observation adds noise)
    true_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Budgets
    remaining_restarts: int = 10
    total_cost: float = 0.0
