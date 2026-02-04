🌐 **Language / 语言:** [English](README.md) | [中文](README.zh-CN.md)

<img src="./assets/logo.png" alt="PharmaLink Analyzer Logo" width="200" />

<div align="center">

# PharmaLink Analyzer

**Attention-Based Pharmaceutical Component Relationship Analysis**

*A desktop application for discovering and visualizing synergistic/antagonistic relationships between traditional Chinese medicine components using transformer attention mechanisms and symbolic regression.*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Vue 3](https://img.shields.io/badge/Vue-3.5-4FC08D?logo=vue.js&logoColor=white)](https://vuejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tauri 2](https://img.shields.io/badge/Tauri-2.x-FFC131?logo=tauri&logoColor=black)](https://tauri.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

PharmaLink Analyzer is a research tool that combines **transformer attention mechanisms** with **symbolic regression** to analyze relationships between pharmaceutical components. Given a dataset of traditional Chinese medicine components, it:

1. **Imports** CSV datasets with automatic feature detection and statistical profiling
2. **Trains** a Feature Transformer model with multi-head attention and K-Fold cross-validation
3. **Extracts** attention weights to quantify component interactions
4. **Derives** interpretable mathematical expressions via gplearn symbolic regression
5. **Evaluates** model performance with per-fold metrics (R², MSE, MAE, RMSE)
6. **Formulates** optimal component combinations based on attention-weighted analysis

### Key Features

- **Data Import** — Load CSV datasets with automatic feature detection and statistical profiling
- **Model Training** — Transformer-based training with real-time WebSocket progress updates, K-Fold cross-validation, and early stopping
- **Attention Analysis** — Heatmap visualization and synergistic/antagonistic component network graphs
- **Model Evaluation** — K-Fold metrics (R², MSE, MAE), residual distributions, and loss curves
- **Expression Derivation** — Symbolic regression via gplearn with simplify, optimize (Pareto front navigation), and undo/redo history
- **Formulation** — Optimal component combination recommendations based on attention-weighted variable impact analysis

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Desktop Shell** | [Tauri 2](https://tauri.app) | Cross-platform desktop packaging |
| **Frontend** | [Vue 3](https://vuejs.org) + TypeScript | SPA with Composition API |
| **Visualization** | [ECharts](https://echarts.apache.org) + [D3.js](https://d3js.org) | Heatmaps, network graphs, charts |
| **Math Rendering** | [KaTeX](https://katex.org) | LaTeX expression rendering |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com) | REST API + WebSocket |
| **ML Core** | [PyTorch](https://pytorch.org) | Feature Transformer with attention |
| **Symbolic Regression** | [gplearn](https://gplearn.readthedocs.io/) | Interpretable equation discovery |
| **Data** | [pandas](https://pandas.pydata.org) + [scikit-learn](https://scikit-learn.org) | Data processing and evaluation |

---

## Getting Started

### Prerequisites

- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **Node.js 18+** with npm
- **Rust** (for Tauri, optional — only needed for desktop builds)

### Install & Run (Development)

```bash
# Clone the repository
git clone https://github.com/ChHsiching/pharmalink-analyzer.git
cd pharmalink-analyzer

# Install frontend dependencies
cd src-frontend && npm install && cd ..

# Install backend dependencies (from project root)
cd src-backend && uv sync && cd ..

# Start both frontend and backend
npm run dev
```

The application will be available at:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8765/api/v1
- **API Docs:** http://localhost:8765/docs

### Run Tests

```bash
# Backend tests (471 tests)
cd src-backend && uv run pytest tests/ -v --timeout=60

# Frontend type check
cd src-frontend && npx vue-tsc --noEmit

# Frontend unit tests
cd src-frontend && npx vitest run
```

---

## Project Structure

```
pharmalink-analyzer/
├── src-frontend/              # Vue 3 + TypeScript frontend
│   └── src/
│       ├── App.vue            # Root component with workflow initialization
│       ├── composables/       # Vue Composition API hooks
│       │   ├── useApi.ts      # API client + safeRequest helper
│       │   ├── useDatasets.ts
│       │   ├── useTraining.ts # Training with WebSocket progress
│       │   ├── useAnalysis.ts
│       │   ├── useEvaluation.ts
│       │   ├── useExpression.ts
│       │   ├── useFormulation.ts  # Formulation page state
│       │   └── useWorkflow.ts # Cross-module state coordination
│       ├── views/             # Page-level components
│       └── components/        # Shared UI components (PlSelect, PlButton, etc.)
│
├── src-backend/               # Python FastAPI backend
│   └── app/
│       ├── api/               # REST API routers (FastAPI Depends injection)
│       ├── services/          # Business logic layer
│       │   ├── checkpoint_resolver.py  # Centralized checkpoint operations
│       │   ├── data_loader.py          # Dataset loading & caching
│       │   ├── analysis.py             # Attention analysis service
│       │   ├── evaluation.py           # Model evaluation service
│       │   ├── expression.py           # Symbolic regression service
│       │   ├── expression_pipeline.py  # Expression generation pipeline
│       │   ├── expression_state.py     # Expression state management
│       │   ├── formulation.py          # Component formulation service
│       │   ├── training.py             # Model training service
│       │   └── training_pipeline.py    # Training pipeline orchestration
│       ├── ml/                # Machine learning modules
│       │   ├── transformer.py          # Feature Transformer model
│       │   ├── attention_extractor.py  # Attention weight extraction
│       │   ├── evaluator.py            # K-Fold evaluation pipeline
│       │   ├── expression_tree.py      # Expression tree operations
│       │   ├── expression_impact.py    # Variable impact analysis
│       │   ├── formulator.py           # Component formulation ML
│       │   └── symbolic_regressor.py   # gplearn integration
│       ├── models/            # Pydantic request/response models
│       ├── dependencies.py    # FastAPI dependency injection factories
│       └── exceptions.py      # Domain exception hierarchy
│
├── docs/                      # Design docs, specs, and plans
└── package.json               # Dev script orchestration
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Datasets** | | |
| GET | `/api/v1/datasets` | List available datasets |
| POST | `/api/v1/datasets` | Upload CSV dataset |
| GET | `/api/v1/datasets/{dataset_id}` | Get dataset detail |
| GET | `/api/v1/datasets/{dataset_id}/stats` | Dataset statistics |
| DELETE | `/api/v1/datasets/{dataset_id}` | Delete dataset |
| **Training** | | |
| POST | `/api/v1/models/train` | Start model training |
| GET | `/api/v1/models/train/status` | Get training status |
| POST | `/api/v1/models/train/stop` | Stop training |
| GET | `/api/v1/models/checkpoints` | List saved checkpoints |
| **Analysis** | | |
| GET | `/api/v1/analysis/attention/{model_id}` | Get attention matrix |
| GET | `/api/v1/analysis/attention/{model_id}/heatmap` | Get heatmap data |
| GET | `/api/v1/analysis/network/{model_id}` | Get interaction network |
| **Evaluation** | | |
| GET | `/api/v1/evaluation/metrics/{model_id}` | K-Fold metrics |
| GET | `/api/v1/evaluation/predictions/{model_id}` | Actual vs predicted |
| GET | `/api/v1/evaluation/residuals/{model_id}` | Residual distribution |
| GET | `/api/v1/evaluation/loss-curve/{model_id}` | Training loss curves |
| **Expression** | | |
| POST | `/api/v1/expressions/generate/{model_id}` | Generate expression |
| GET | `/api/v1/expressions/result/{task_id}` | Get generation result |
| POST | `/api/v1/expressions/simplify/{expr_id}` | Simplify expression |
| POST | `/api/v1/expressions/optimize/{expr_id}` | Navigate Pareto front |
| GET | `/api/v1/expressions/tree/{expr_id}` | Get expression tree |
| GET | `/api/v1/expressions/history/{expr_id}` | Operation history |
| POST | `/api/v1/expressions/undo/{expr_id}` | Undo operation |
| **Formulation** | | |
| POST | `/api/v1/formulation` | Generate formulation |

---

## Build & Deployment

### Build Frontend

```bash
cd src-frontend && npm run build
```

### Build Desktop App (Tauri)

```bash
npm run tauri:build
```

This produces platform-specific installers (`.msi`/`.exe` for Windows, `.dmg` for macOS, `.AppImage` for Linux).

---

## Architecture

The backend follows a **layered architecture** with clear dependency boundaries:

- **API Layer** (`app/api/`) — FastAPI routers with `Depends()` injection
- **Service Layer** (`app/services/`) — Business logic, depends on `CheckpointResolver`
- **ML Layer** (`app/ml/`) — Pure ML operations (model loading, attention extraction, symbolic regression)
- **Domain Exceptions** (`app/exceptions.py`) — Service-layer errors decoupled from FastAPI

All services receive dependencies through constructor injection (`CheckpointResolver`), enabling clean testing via FastAPI's `app.dependency_overrides`.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
