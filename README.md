---
title: SRE Autopilot Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - sre
  - incident-response
---

# 🚨 SRE Autopilot: Production Incident Response

This environment simulates a **microservices system under failure** for the Meta-PyTorch Hackathon. 

The agent acts as an on-call Site Reliability Engineer (SRE) receiving live, noisy telemetry across a dependency graph. Its goal is to diagnose the root cause and execute remediation actions to restore Service Level Agreement (SLA) compliance over a 30-step episode.

---

## 🧠 Why this is an Elite-Tier RL Environment

Unlike typical LLM "agentic" benchmarks (like email triage or code review) where the optimal policy can be deduced through pure zero-shot reasoning, this environment structurally **demands a learned Reinforcement Learning policy**. 

**Why naive LLMs fail here:**
1. **Hidden Failure Modes:** The true failure mode (e.g. `db_connection_leak`, `flapping_network`) is hidden latent state. Only probing and observing the system over time reveals the true root cause.
2. **Delayed & Compound Effects:** Restarts cause immediate 1-step downtime penalties but fix the long-term cascade. Scaling up takes 3 steps to take effect. 
3. **Irreversible Consequences:** Actions cost resources (budgeted restarts). Trap actions temporarily improve metrics but do not fix the root cause.
4. **Stochastic Observability:** Telemetry data contains Gaussian noise, dropping spans, and lag.

## 🚀 Tasks

We implement exactly 3 tiers of difficulty perfectly suited for continuous grading:

1. **Easy (`single_oom`)**
   - Single service OOM, clear error spike, one restart fixes it 80% of the time. Tests basic pattern recognition and categorical action execution.
2. **Medium (`db_connection_leak`)**
   - Cascading failure — DB leak causes latency to slowly spike, cascading up to the API Gateway. Fixing the leaf nodes doesn't help; the agent must traverse the trace graph and restart the true root.
3. **Hard (`flapping_network`)**
   - Intermittent flapping: network dropping on payment gateway momentarily. The metrics recover and regress. Agent must learn the specific fingerprint through trial and error across episodes.

---

## 📊 State & Action Space

### Observation (`SREObservation`)
- `metrics`: Dictionary mapping service IDs to noisy telemetry (latency, errors, CPU, RAM).
- `dependency_graph`: Adjacency list of microservice traces.
- `action_history`: The last 5 actions taken.
- `sla_status`: Boolean indicator of which services are currently meeting the 200ms latency / 1% error SLA.

### Actions (`SREAction`)
- `restart` (costs 1 from budget, induces 1 step 100% error rate downtime)
- `scale_up` (costs money, takes 3 steps to realize capacity doubling)
- `wait` (passes 1 step)

---

## 💰 Multi-Objective Reward Function

The reward is dense and compounding:
- **`+ [0.0 - 1.0]`**: Fraction of services meeting SLA per step.
- **`-0.1`**: Destructive action penalty (restarting consumes budget and causes downtime).
- **`-0.05`**: Cost penalty (scaling up).
- **`+5.0`**: Early resolution bonus if 100% SLA is restored before max steps.

---

## 🛠 Usage

### Quick Start Python Client
```python
from sre_env import SREEnv, SREAction

env = SREEnv(base_url="http://localhost:8000")

# Test the easiest failure tier
result = env.reset(task_tier="easy")

# Restart the suspected root cause
action = SREAction(action="restart", service_id="api_gateway")
result = env.step(action)

print(f"Reward: {result.reward}")
```

### Baseline Agent
Run the baseline rule-based / random agent to verify solver mechanics:
```bash
python envs/sre_env/baseline.py
```

## 🐳 Docker Deployment
```bash
docker build -t sre-env:latest -f envs/sre_env/server/Dockerfile .
docker run -p 8000:8000 sre-env:latest
```
