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

## 🧠 Why This Is an Elite-Tier RL Environment

Unlike typical LLM "agentic" benchmarks where the optimal policy can be deduced through pure zero-shot reasoning, this environment structurally **demands a learned Reinforcement Learning policy**. 

**Why naive LLMs fail here:**
1. **Hidden Failure Modes:** The true failure mode (e.g., `db_connection_leak`, `flapping_network`) is hidden latent state. Only probing and observing the system over time reveals the true root cause.
2. **Delayed & Compound Effects:** Restarts cause immediate 1-step downtime penalties but fix the long-term cascade. Scaling up takes 3 steps to take effect. 
3. **Irreversible Consequences:** Actions cost resources (budgeted restarts). Trap actions temporarily improve metrics but do not fix the root cause.
4. **Stochastic Observability:** Telemetry data contains Gaussian noise, dropping spans, and lag.

---

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
- `metrics`: Dictionary mapping service IDs to noisy telemetry (`latency_p99`, `error_rate`, `cpu_pct`, `memory_pct`).
- `dependency_graph`: Adjacency list of microservice traces.
- `action_history`: The last 5 actions taken.
- `sla_status`: Boolean indicator of which services are currently meeting the 200ms latency / 1% error SLA.
- `reward_hint`: Dense reward signal for the current step.
- `reward_breakdown`: Structured breakdown (SLA compliance, cost penalty, downtime penalty, resolution bonus).

### Actions (`SREAction`)
- `restart` — Restarts a service. Costs 1 from budget, induces 1-step 100% error rate downtime but can fix faults.
- `scale_up` — Doubles capacity. Costs money, takes 3 steps to realize effect.
- `circuit_break` — Opens circuit breaker on a service. Stops propagation but 100% error locally for 3 steps.
- `rollback` — Instant fix but very high cost penalty.
- `wait` — Passes 1 step, observes without acting.

### Services
`api_gateway`, `auth_service`, `product_service`, `order_service`, `user_db`, `inventory_db`, `payment_gateway`

---

## 💰 Multi-Objective Reward Function

The reward is dense and compounding:
- **`+ [0.0 - 1.0]`**: Fraction of services meeting SLA per step.
- **`-0.1`**: Destructive action penalty (restarting consumes budget and causes downtime).
- **`-0.05`**: Cost penalty (scaling up).
- **`+5.0`**: Resolution bonus if 100% SLA is restored at episode end.

---

## 🛠 Usage

### Environment Variables (Required for Inference)

Before running `inference.py`, set the following environment variables:

```bash
export API_BASE_URL="https://api.openai.com/v1"   # The API endpoint for the LLM
export MODEL_NAME="gpt-4o-mini"                     # The model identifier to use
export HF_TOKEN="your-api-key-here"                 # Your Hugging Face / API key
```

### Quick Start — Python Client

```python
from client import SREEnv
from models import SREAction

env = SREEnv(base_url="http://localhost:8000")

# Test the easiest failure tier
result = env.reset(task_tier="easy")

# Restart the suspected root cause
action = SREAction(action="restart", service_id="api_gateway")
result = env.step(action)

print(f"Reward: {result.reward}")
```

### Running Inference (Submission Script)

The inference script is the **required submission entry point**. It uses the OpenAI Client:

```bash
# 1. Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 2. In another terminal, run inference
python inference.py
```

This will run all 3 task tiers and output rewards + grader scores in the 0.0–1.0 range.

### Running the Baseline Agent

The baseline uses a local Qwen model for experimentation (not required for submission):

```bash
python baseline.py
```

---

## 🐳 Docker Deployment

```bash
docker build -t sre-env:latest .
docker run -p 8000:8000 sre-env:latest
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (returns 200) |
| `/reset` | POST | Reset environment (accepts `{"task_tier": "easy\|medium\|hard"}`) |
| `/step` | POST | Take an action (accepts `{"action": "...", "service_id": "..."}`) |
| `/state` | GET | Get the current hidden state |
| `/tasks` | GET | List available tasks and action schema |
| `/grader?tier=easy` | GET | Run heuristic agent + grade (returns score 0.0–1.0) |
| `/baseline` | GET | Run baseline on all 3 tiers, returns scores |

---

## ⚙️ Infra Requirements

- **Runtime:** Inference must complete in under 20 minutes
- **Resources:** Must run on vcpu=2, memory=8GB
- **LLM Calls:** Must use OpenAI Client via `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
