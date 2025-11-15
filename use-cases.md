
# **ExplainAI**

### *Transparent. Comparable. Controllable Machine Learning.*

![ExplainAI Logo](assets/logo.png)

[![Build Status](https://github.com/asearer/ExplainAI/actions/workflows/python-app.yml/badge.svg)](https://github.com/asearer/ExplainAI/actions/workflows/python-app.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

# 🌟 **What is ExplainAI?**

**ExplainAI is a unified evaluation, comparison, and explainability framework for modern language models.**
It lets you:

* Evaluate **multiple LLMs side-by-side**
* Understand **how models think**, not just what they output
* Compare **local and cloud models** in one interface
* Visualize reasoning, errors, hallucinations, and style
* Run experiments and benchmark models reliably

ExplainAI brings structure, instrumentation, visualization, and reproducibility to an increasingly complex model ecosystem.

---

# 🚀 **Why ExplainAI Exists**

Today’s ML systems are hybrid, fragmented, and often opaque.
Developers need to answer questions like:

* *Which model is best for my use case?*
* *How do their reasoning styles differ?*
* *Why is my agent failing intermittently?*
* *Is my fine-tuned model safe to deploy?*
* *How do I run local models and cloud models together?*
* *How do I prove compliance or produce reproducible research?*

**ExplainAI solves all of these systematically.**

---

# 🧠 **Core Capabilities**

### **1. Model Comparison Engine**

Evaluate any combination of local or cloud LLMs:

* OpenAI
* Anthropic
* Gemini
* Mistral
* Llama / Llama GGUF
* Custom fine-tuned models
* Ollama backends
* Local inference frameworks (llama.cpp, vLLM, etc.)

### **2. Explainability Pipeline**

Deep introspection tools:

* token-level diffing
* chain-of-thought pattern analysis
* hallucination detection
* reasoning divergence visualization
* prompt sensitivity mapping
* error heatmaps
* semantic similarity graphs

### **3. Evaluation Suite**

Run structured tests on:

* correctness
* consistency
* style/voice
* factual accuracy
* domain knowledge
* safety compliance
* regression across model versions

### **4. Local + Cloud Hybrid Support**

Use local models for cheap inference, and cloud models for high-reasoning tasks.

### **5. Dashboard & Reporting**

Visualize results and export reports for:

* engineering
* compliance
* research
* education
* audit workflows

---

# 🏗️ **How ExplainAI Fits Into a Larger ML System**

ExplainAI is designed as a **middle layer** between your ML models and your applications.

```
                     ┌────────────────────────┐
                     │        Frontends       │
     (Apps, Agents, Dashboards, Products)     │
                     └──────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │       ExplainAI           │
                   │  Evaluation + Comparison  │
                   │  Explainability Engine    │
                   │  Model Routing Layer      │
                   │  Experiment Runner        │
                   └──────────┬───────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │                          │                          │
     ▼                          ▼                          ▼
┌──────────┐        ┌─────────────────┐         ┌────────────────────┐
│ Cloud LLM│        │ Local Models    │         │ Fine-tuned Models  │
│  APIs     │        │ (GGUF, Ollama) │         │ (domain specific)  │
└──────────┘        └─────────────────┘         └────────────────────┘
```

ExplainAI acts as:

* **a benchmarking engine**
* **a debugging layer**
* **an observability dashboard**
* **a model selection oracle**
* **a safety/compliance audit tool**
* **a routing optimizer** for hybrid deployments

This makes it a foundational component in research groups, AI startups, enterprise ML stacks, and agentic application pipelines.

---

# 🌍 **Real-World Use Cases**

### **1. A Startup Choosing the Right Model**

Compare GPT-4.1, Claude, Gemini, and Llama 3.1 70B side-by-side.
Discover a hybrid approach that lowers costs by **80%**.

### **2. A Researcher Studying Reasoning Patterns**

Run 1,000 math problems through multiple models.
Visualize chain-of-thought divergence and error clusters.
Publish results with reproducible reports.

### **3. An Enterprise Auditing a Fine-Tuned Legal Model**

Test for hallucinations, factual errors, and policy violations.
Generate compliance-ready reports for deployment approvals.

### **4. A Developer Debugging an Agent**

Trace multi-step tool calls.
Identify where reasoning breaks or tools are mis-used.
Reduce error rate dramatically.

### **5. A University Teaching LLM Fundamentals**

Students load local models via Ollama.
Compare them to commercial LLMs.
Great for ML interpretability courses.

### **6. A Privacy-Constrained Medical Lab**

Use exclusively local models with zero cloud dependency.
Run explainability without compromising patient data.

### **7. A Team Monitoring Model Drift**

Automated monthly evaluations detect regressions in updated cloud LLM versions.

### **8. A Developer Optimizing Prompts**

Prompt-sweep automation reveals smaller, faster prompts that preserve quality.

---

# 📦 **Project Structure**

```
ExplainAI/
│
├── core/                     # Evaluation engine, pipelines, scoring modules
│   ├── evaluators/
│   ├── explainers/
│   ├── routers/
│   ├── prompts/
│   └── utilities/
│
├── models/                   # Model adapters (cloud + local)
│   ├── openai/
│   ├── anthropic/
│   ├── google/
│   ├── mistral/
│   ├── ollama/
│   └── llama_cpp/
│
├── dashboards/               # Streamlit front-end (interactive UI)
│   ├── screens/
│   ├── components/
│   └── assets/
│
├── tests/                    # Automated test suite
│
├── examples/                 # Example notebooks, real-world use cases
│
├── assets/                   # Logos, screenshots, diagrams
│
├── README.md                 # This file
└── pyproject.toml            # Dependencies & package metadata
```

---

# 🧪 **Getting Started**

### **Install**

```bash
pip install explainai
```

### **Run the Dashboard**

```bash
explainai dashboard
```

### **Minimal Python Example**

```python
from explainai import Experiment

exp = Experiment(models=["gpt-4.1", "claude-3.5-sonnet", "llama3.1-70b"])

result = exp.compare("Explain quantum tunneling to a child.")

print(result.best_model)
```

### **Use Local Models via Ollama**

```bash
ollama pull llama3.1
```

```python
exp = Experiment(models=["ollama:llama3.1", "gpt-4.1"])
```

---

# 🧩 **API Overview**

### **Experiment**

```python
Experiment(
    models: list,
    metrics: list = ["accuracy", "coherence", "similarity"],
    explain: bool = True
)
```

### **Key Accessors**

```python
result.outputs
result.scores
result.explanations
result.visualizations
result.best_model
```

---

# 🗺️ **Roadmap**

### **Phase 1 — Core Evaluation Engine (Current)**

* Model comparison engine
* Explainability tools
* Local + cloud adapters
* Dashboard v1

### **Phase 2 — Robust Research Tools**

* Versioned experiment tracking
* Dataset integration
* Exportable academic reports

### **Phase 3 — Production ML Integration**

* API gateway
* Inference routing optimizer
* Compliance reports

### **Phase 4 — Agentic Systems Instrumentation**

* Tool-call debugging
* Step-by-step agent visualization

---

# ❤️ Contributing

Contributions welcome after v1.0.
Guidelines coming soon.

---

# 📄 License

MIT License — free for personal and commercial use.

