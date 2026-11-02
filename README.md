# AI-Assisted System Dynamics Modeling for Cardiovascular Disease (2010–2020 Retrospective Validation)

This repository contains an end-to-end framework that combines **Retrieval-Augmented Generation (RAG)** with **Continuous System Dynamics (PySD / ODE Solvers)** to simulate population-level Cardiovascular Disease (CVD) prevalence and policy interventions in the United Kingdom.

Rather than relying on ungrounded LLM code generation, this system extracts structured epidemiological parameters from verified UK public health databases, validates parameter types via Pydantic schemas, verifies code syntax via Abstract Syntax Tree (AST) static analysis, and executes continuous differential equations over a 10-year retrospective horizon (2010–2020) for empirical ground-truth validation.

---

## Key Features & Scientific Architecture

1. **Retrospective Ground-Truth Validation (2010–2020):**
   * Establishes baseline population metrics at $t=0$ (2010) using BHF Statistics, NHS Digital QOF, and NICE NG238 guidelines.
   * Simulates disease trajectories to $t=10$ (2020), enabling direct empirical error scoring ($\text{MAE}$ and $\text{RMSE}$) against recorded NHS outcomes.

2. **Grounded RAG Retrieval (`rag_engine.py`):**
   * Utilizes `langchain-chroma` and dense HuggingFace embeddings (`all-MiniLM-L6-v2`) for semantic chunk retrieval.
   * Employs `RecursiveCharacterTextSplitter` chunking to extract context-specific relative risk multipliers and baseline incidence rates.

3. **Guaranteed Execution & Safety Layer (`ast_parser.py`):**
   * Enforces type-safe parameter bounds via `Pydantic` (`SystemDynamicsParameterSchema`).
   * Performs static AST validation to verify variable dependency trees, prevent execution halts, and block untrusted code execution prior to ODE execution.

4. **Continuous System Dynamics Engine (`cvd_sd_model.py`):**
   * Implements continuous population stock-flow differential equations ($S \rightarrow CVD$) over a 10-year period.
   * Evaluates complex multi-variable policy levers (e.g., combined smoking cessation and hypertension management).

---

## Pipeline Overview

```
[ Public Health Query / Policy Scenario ]
                   │
                   ▼
[ Clinical RAG Engine (ChromaDB + 2010 UK Corpora) ]
                   │
                   ▼
[ Parameter Calibration & Pydantic Schema Validation ]
                   │
                   ▼
[ Static AST Safety & Structural Verification ]
                   │
                   ▼
[ Continuous PySD ODE Solver (2010–2020 Trajectory) ]
                   │
                   ▼
[ Empirical Benchmarking: MAE, RMSE & Deviation Metrics ]
```

---

## Directory Structure

```text
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── rag_engine.py                      # LangChain Chroma RAG pipeline & retriever
├── ast_parser.py                      # AST static analysis and code safety checker
├── schemas.py                         # Pydantic schemas for parameter extraction
├── cvd_sd_model.py                    # PySD System Dynamics model formulation
├── pipeline.py                        # Core pipeline orchestrator
├── experiments.py                     # Experimental harness for multi-scenario testing
├── output/
│   ├── batch_experiments_summary.csv  # Empirical validation metrics (MAE / RMSE)
│   └── cvd_simulation_trajectories.png# Trajectory plots across interventions
└── chroma_db/                         # Local vector database storage
```

---

## Quickstart & Installation

### 1. Prerequisites
* Python 3.10+
* Virtual Environment (recommended)

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/cvd-system-dynamics-rag.git
cd cvd-system-dynamics-rag

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file or export your API keys in your shell:
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# Optional for multi-model comparisons:
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Running Experiments

To populate the vector store, calibrate parameters, run the ODE solver, and calculate empirical validation metrics:

```bash
python experiments.py
```

### Outputs Generated:
* **`output/batch_experiments_summary.csv`**: Contains baseline prevalence, simulated 2020 prevalence, $\text{MAE}$, and $\text{RMSE}$ scores for each intervention scenario.
* **`output/cvd_simulation_trajectories.png`**: Multi-curve visual trajectory comparison of 2010–2020 disease prevalence across scenarios.

---

## Experimental Scenarios

1. **`EXP_01_Smoking_Cessation`**: Evaluates targeted community smoking reduction programs on 10-year CVD trajectory.
2. **`EXP_02_Hypertension_Control`**: Models systematic blood pressure screening and clinical management.
3. **`EXP_03_Combined_Intervention`**: Simulates synergistic multi-variable public health policies.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
