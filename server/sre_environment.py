import random
import uuid
import math
from typing import Dict, List, Optional
try:
    from models import SREAction, SREObservation, SREState, SREReward
except (ImportError, ValueError):
    try:
        from sre_env.models import SREAction, SREObservation, SREState, SREReward
    except (ImportError, ValueError):
        from ..models import SREAction, SREObservation, SREState, SREReward
from openenv.core.env_server.interfaces import Environment

# Fixed topology
DEPENDENCY_GRAPH = {
    "api_gateway": ["auth_service", "product_service", "order_service"],
    "auth_service": ["user_db"],
    "product_service": ["inventory_db"],
    "order_service": ["payment_gateway", "inventory_db"],
    "user_db": [],
    "inventory_db": [],
    "payment_gateway": []
}

SERVICES = list(DEPENDENCY_GRAPH.keys())

# SLA Thresholds
TARGET_LATENCY = 200.0
TARGET_ERROR_RATE = 0.01

class SREEnvironment(Environment):
    """
    SRE Autopilot Simulation Environment.
    Models microservices with latent failures and cascading latency/errors.
    """
    def __init__(self, seed: int = 42, max_steps: int = 30):
        super().__init__()
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self._state: SREState | None = None
        self.action_history: List[str] = []

    def reset(self, task_tier: str = "random") -> SREObservation:
        self.action_history = []
        
        # Pick failure mode based on tier
        if task_tier == "easy":
            modes = ["single_oom"]
        elif task_tier == "medium":
            modes = ["db_connection_leak"]
        elif task_tier == "hard":
            modes = ["flapping_network"]
        else:
            modes = ["single_oom", "db_connection_leak", "flapping_network", "healthy"]
            
        failure_mode = self.rng.choice(modes)
        
        # Assign root cause
        if failure_mode == "single_oom":
            root_cause = self.rng.choice(["api_gateway", "product_service"])
        elif failure_mode == "db_connection_leak":
            root_cause = self.rng.choice(["inventory_db", "user_db"])
        elif failure_mode == "flapping_network":
            root_cause = "payment_gateway"
        else:
            root_cause = None

        # Base nominal metrics
        true_metrics = {}
        for s in SERVICES:
            true_metrics[s] = {
                "base_latency": self.rng.uniform(10.0, 30.0),
                "base_error": 0.0,
                "capacity": 1.0,
                "leak_counter": 0.0,
                "restarting_timer": 0,
                "scaling_timer": 0,
                "flapping_state": 0.0
            }

        self._state = SREState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            last_action="reset",
            failure_mode=failure_mode,
            root_cause_service=root_cause,
            true_metrics=true_metrics,
            remaining_restarts=5,
            total_cost=0.0
        )
        self.action_history.append("reset")
        return self._make_observation()

    def step(self, action: SREAction) -> SREObservation:
        st = self._state
        if st is None:
            return self.reset()

        reward = 0.0
        st.step_count += 1
        st.last_action = action.action
        
        action_str = f"{action.action}({action.service_id or ''})"
        self.action_history.append(action_str)
        if len(self.action_history) > 5:
            self.action_history.pop(0)

        # 1. Process Actions
        if action.action == "restart":
            if action.service_id in SERVICES:
                if st.remaining_restarts > 0:
                    st.remaining_restarts -= 1
                    # Restart induces 1 step downtime but resets leak
                    st.true_metrics[action.service_id]["restarting_timer"] = 1
                    st.true_metrics[action.service_id]["leak_counter"] = 0.0
                    # 80% chance to fix single OOM
                    if st.failure_mode == "single_oom" and action.service_id == st.root_cause_service:
                        if self.rng.random() < 0.8:
                            st.failure_mode = "healthy"
                    # Small penalty for downtime causing restarts
                    reward -= 0.1
                else:
                    reward -= 0.1  # Out of budget
            else:
                reward -= 0.1 # Invalid service
                
        elif action.action == "scale_up":
            if action.service_id in SERVICES:
                st.total_cost += 1.0
                reward -= 0.05  # Cost penalty
                st.true_metrics[action.service_id]["scaling_timer"] = 3
            else:
                reward -= 0.1
                
        elif action.action == "circuit_break":
            if action.service_id in SERVICES:
                # Stops propagation from this service but sets its own error as 100%
                st.true_metrics[action.service_id]["circuit_open_timer"] = 3
                reward -= 0.2 # Penalty for deliberately breaking traffic
            else:
                reward -= 0.1

        elif action.action == "rollback":
            if action.service_id in SERVICES:
                # Fixes failure instantly but very high cost
                st.failure_mode = "healthy"
                st.total_cost += 5.0
                reward -= 0.5 
            else:
                reward -= 0.1
                
        elif action.action == "wait":
            pass
        else:
            reward -= 0.1  # Invalid action

        # 2. Advance simulation state
        for s in SERVICES:
            tm = st.true_metrics[s]
            if tm["restarting_timer"] > 0:
                tm["restarting_timer"] -= 1
            if tm["scaling_timer"] > 0:
                tm["scaling_timer"] -= 1
                if tm["scaling_timer"] == 0:
                    tm["capacity"] += 0.5  # Capacity unlocked
            if tm.get("circuit_open_timer", 0) > 0:
                tm["circuit_open_timer"] -= 1

        # Advance latency failures
        if st.failure_mode == "db_connection_leak" and st.root_cause_service:
            # Leak continuously grows if not restarted
            tm = st.true_metrics[st.root_cause_service]
            if tm["restarting_timer"] == 0:
                tm["leak_counter"] += 1.0
                
        if st.failure_mode == "single_oom" and st.root_cause_service:
            tm = st.true_metrics[st.root_cause_service]
            if tm["restarting_timer"] == 0:
                tm["leak_counter"] += 1.0  # Acts as memory pressure that crashes repeatedly
                if tm["leak_counter"] > 3:
                    tm["base_error"] = 1.0  # OOM Crash

        if st.failure_mode == "flapping_network" and st.root_cause_service:
            tm = st.true_metrics[st.root_cause_service]
            if tm["restarting_timer"] == 0:
                # 30% chance to flip state
                if self.rng.random() < 0.3:
                    tm["flapping_state"] = 1.0 if tm["flapping_state"] == 0.0 else 0.0

        # 3. Calculate propagated metrics (Bottom-Up)
        # Sort topological from leaves to root
        topo_sorted = ["user_db", "inventory_db", "payment_gateway", 
                       "auth_service", "product_service", "order_service", "api_gateway"]
        
        calculated_metrics = {}
        for s in SERVICES:
            calculated_metrics[s] = {"latency_p99": 0.0, "error_rate": 0.0}

        for s in topo_sorted:
            tm = st.true_metrics[s]
            lat = tm["base_latency"] / tm["capacity"]
            err = tm["base_error"]

            # Apply immediate failure states
            if tm["restarting_timer"] > 0:
                err = 1.0
                lat = 0.0
            else:
                if st.failure_mode == "db_connection_leak" and s == st.root_cause_service:
                    lat += (tm["leak_counter"] * 50.0) / tm["capacity"]
                elif st.failure_mode == "flapping_network" and s == st.root_cause_service:
                    err = 1.0 if tm["flapping_state"] == 1.0 else 0.0
                elif st.failure_mode == "single_oom" and s == st.root_cause_service:
                    if tm["leak_counter"] > 3:
                        err = 1.0
                
                if tm.get("circuit_open_timer", 0) > 0:
                    err = 1.0 # Circuit is open, 100% error locally
                        
            # Aggregate dependencies
            max_dep_lat = 0.0
            for dep in DEPENDENCY_GRAPH[s]:
                # If circuit is open, we don't propagate downstream errors/latency
                if tm.get("circuit_open_timer", 0) == 0:
                    if calculated_metrics[dep]["error_rate"] > 0:
                        err = max(err, calculated_metrics[dep]["error_rate"])
                    max_dep_lat = max(max_dep_lat, calculated_metrics[dep]["latency_p99"])

            calculated_metrics[s]["latency_p99"] = lat + max_dep_lat
            calculated_metrics[s]["error_rate"] = min(1.0, err)

        # 4. Reward function (SLA compliance & Final Grade)
        sla_count = 0
        sla_status = {}
        for s in SERVICES:
            in_sla = calculated_metrics[s]["latency_p99"] < TARGET_LATENCY and calculated_metrics[s]["error_rate"] <= TARGET_ERROR_RATE
            sla_status[s] = in_sla
            if in_sla:
                sla_count += 1
                
        frac_in_sla = sla_count / len(SERVICES)

        # Base reward = 0 for all non-terminal steps. 
        # The pipeline evaluator validates that the overall reward score is strictly (0, 1), not per-step.
        reward = 0.0

        done = st.step_count >= self.max_steps
        
        # Calculate terminal reward (score) using the true 0.01 - 0.99 grader
        if done:
            try:
                from grader import grade_episode
            except (ImportError, ValueError):
                try:
                    from sre_env.grader import grade_episode
                except (ImportError, ValueError):
                    try:
                        from ..grader import grade_episode
                    except Exception:
                        grade_episode = None
            
            if grade_episode:
                total_cost = (self.initial_restarts - st.remaining_restarts) * 1.0
                reward = grade_episode(
                    tier=self.task_tier,
                    final_sla_status=sla_status,
                    total_steps=st.step_count,
                    max_steps=self.max_steps,
                    remaining_restarts=st.remaining_restarts,
                    initial_restarts=self.initial_restarts,
                    total_cost=total_cost,
                    failure_mode=st.failure_mode,
                    total_reward=0.0
                )
            else:
                # Fallback purely bound heuristic
                fallback_score = frac_in_sla * 0.5 + 0.25 if frac_in_sla == 1.0 else frac_in_sla * 0.5
                reward = max(0.01, min(0.99, fallback_score))

        obs = self._make_observation(calculated_metrics, sla_status, float(reward))
        return obs

    def _make_observation(self, calc_metrics=None, sla_status=None, reward_hint=0.0) -> SREObservation:
        st = self._state
        
        # Structured reward breakdown
        
        sla_comp = 0.0
        cost_pen = 0.0
        down_pen = 0.0
        res_bonus = 0.0
        
        if sla_status:
            sla_comp = sum(1 for v in sla_status.values() if v) / len(SERVICES)
        
        if st:
            cost_pen = (st.total_cost * 0.05) / st.step_count if st.step_count > 0 else 0.0
            # Downtime penalty is -0.1 per restart in step() logic, we'll just hint at it
        
        done = False
        if st:
            done = st.step_count >= self.max_steps
            if done and sla_comp == 1.0:
                res_bonus = 5.0

        reward_breakdown = SREReward(
            sla_compliance=sla_comp,
            cost_penalty=cost_pen,
            downtime_penalty=0.0, # Handled per-step
            resolution_bonus=res_bonus,
            total=reward_hint
        )

        if calc_metrics is None:
            # Default healthy observation before stepping
            calc_metrics = {s: {"latency_p99": 20.0, "error_rate": 0.0} for s in SERVICES}
            sla_status = {s: True for s in SERVICES}

        # Apply Observability Noise
        noisy_metrics = {}
        for s in SERVICES:
            noisy_metrics[s] = {
                "latency_p99": max(0.0, self.rng.gauss(calc_metrics[s]["latency_p99"], 5.0)),
                "error_rate": max(0.0, min(1.0, self.rng.gauss(calc_metrics[s]["error_rate"], 0.02))),
                "cpu_pct": max(0.0, min(100.0, self.rng.gauss(30.0 + (calc_metrics[s]["latency_p99"] * 0.1), 5.0))),
                "memory_pct": max(0.0, min(100.0, self.rng.gauss(40.0, 2.0)))
            }
            # Simulate db leak memory pressure
            if st.failure_mode == "db_connection_leak" and s == st.root_cause_service:
                noisy_metrics[s]["memory_pct"] = min(100.0, 40.0 + st.true_metrics[s]["leak_counter"] * 10.0)
            elif st.failure_mode == "single_oom" and s == st.root_cause_service:
                noisy_metrics[s]["memory_pct"] = min(100.0, 50.0 + st.true_metrics[s]["leak_counter"] * 20.0)

            # Randomly drop metrics (Missing Data) 5% of the time
            if self.rng.random() < 0.05:
                noisy_metrics[s]["latency_p99"] = -1.0 # Indicate missing/timeout

        return SREObservation(
            metrics=noisy_metrics,
            dependency_graph=DEPENDENCY_GRAPH,
            action_history=self.action_history,
            step=st.step_count,
            sla_status=sla_status,
            reward_hint=reward_hint,
            reward=reward_hint,
            done=done,
            reward_breakdown=reward_breakdown
        )

    @property
    def state(self) -> SREState:
        if self._state is None:
            self.reset()
        return self._state
