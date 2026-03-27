"""
FastAPI application for the SRE Autopilot Environment.

Exposes the standard OpenEnv HTTP interface (step/reset/state)
plus hackathon-required endpoints: /tasks, /grader, /baseline.
"""
import os
import json
import time
import traceback
from typing import Optional

from fastapi import Query
from openenv.core.env_server.http_server import create_app
from ..models import SREAction, SREObservation
from .sre_environment import SREEnvironment
from ..grader import grade_episode, TASKS, ACTION_SCHEMA


def create_sre_environment():
    return SREEnvironment()

app = create_app(
    create_sre_environment,
    SREAction,
    SREObservation,
    env_name="sre_env",
)


# ──────────────────────────────────────────────────────────
# Additional Endpoints Required by Hackathon Submission
# ──────────────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """Return the list of available tasks and the action schema."""
    return {
        "tasks": TASKS,
        "action_schema": ACTION_SCHEMA,
    }


@app.get("/grader")
def get_grader(tier: str = Query("easy", description="Task tier: easy, medium, hard")):
    """
    Run one full episode with a simple heuristic agent, then grade it.
    Returns the grader score (0.0–1.0) for the specified task tier.
    """
    env = SREEnvironment(seed=int(time.time()) % 100000)
    obs = env.reset(task_tier=tier)

    # Run a deterministic heuristic agent for reproducibility
    for _ in range(env.max_steps):
        if obs.done:
            break

        # Simple heuristic: restart the worst-performing service
        worst_service = None
        max_err = 0.0
        for svc, metrics in obs.metrics.items():
            score = metrics.get("error_rate", 0.0) + (metrics.get("latency_p99", 0.0) / 1000.0)
            if score > max_err:
                max_err = score
                worst_service = svc

        if max_err > 0.1 and worst_service:
            action = SREAction(action="restart", service_id=worst_service)
        else:
            action = SREAction(action="wait")

        obs = env.step(action)

    # Now grade the completed episode
    st = env.state
    score = grade_episode(
        tier=tier,
        final_sla_status=obs.sla_status,
        total_steps=st.step_count,
        max_steps=env.max_steps,
        remaining_restarts=st.remaining_restarts,
        initial_restarts=5,
        total_cost=st.total_cost,
        failure_mode=st.failure_mode,
        total_reward=st.total_reward,
    )
    return {
        "tier": tier,
        "score": score,
        "steps_taken": st.step_count,
        "final_sla": obs.sla_status,
        "failure_mode": st.failure_mode,
    }


@app.get("/baseline")
def run_baseline():
    """
    Run the baseline heuristic agent on all 3 tasks and return scores.
    This endpoint is called by the automated judging pipeline.
    """
    results = {}
    for tier in ["easy", "medium", "hard"]:
        env = SREEnvironment(seed=42)  # Fixed seed for reproducibility
        obs = env.reset(task_tier=tier)

        for _ in range(env.max_steps):
            if obs.done:
                break

            # Deterministic heuristic baseline
            worst_service = None
            max_err = 0.0
            for svc, metrics in obs.metrics.items():
                score = metrics.get("error_rate", 0.0) + (metrics.get("latency_p99", 0.0) / 1000.0)
                if score > max_err:
                    max_err = score
                    worst_service = svc

            if max_err > 0.1 and worst_service:
                action = SREAction(action="restart", service_id=worst_service)
            else:
                action = SREAction(action="wait")

            obs = env.step(action)

        st = env.state
        score = grade_episode(
            tier=tier,
            final_sla_status=obs.sla_status,
            total_steps=st.step_count,
            max_steps=env.max_steps,
            remaining_restarts=st.remaining_restarts,
            initial_restarts=5,
            total_cost=st.total_cost,
            failure_mode=st.failure_mode,
            total_reward=st.total_reward,
        )
        results[tier] = {
            "score": score,
            "steps_taken": st.step_count,
            "final_sla": obs.sla_status,
        }

    return {
        "baseline_agent": "deterministic_heuristic",
        "model": "rule_based (restart worst service)",
        "results": results,
    }


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
