🌐 **Language / 语言:** [English](README.md) | [中文](README.zh-CN.md)

<img src="./assets/logo.png" alt="PharmaLink Analyzer Logo" width="200" />

<div align="center">

# PharmaLink Analyzer

**基于注意力机制的中药成分关联性分析**

*利用 Transformer 注意力机制与符号回归，发现和可视化中药成分间协同/拮抗关系的桌面应用。*

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Vue 3](https://img.shields.io/badge/Vue-3.5-4FC08D?logo=vue.js&logoColor=white)](https://vuejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tauri 2](https://img.shields.io/badge/Tauri-2.x-FFC131?logo=tauri&logoColor=black)](https://tauri.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 概述

PharmaLink Analyzer 是一款科研工具，结合 **Transformer 注意力机制** 和 **符号回归** 分析中药成分之间的关联关系。给定中药成分数据集，它可以：

1. **训练** 具有多头注意力机制的 Feature Transformer 模型
2. **提取** 注意力权重来量化成分间的交互作用
3. **可视化** 注意力热力图和交互网络图
4. **生成** 可解释的数学表达式（符号回归）
5. **评估** 模型性能（K-Fold 交叉验证）

### 核心功能

- **数据导入** — 加载 CSV 数据集，自动检测特征列并提供统计概览
- **模型训练** — 基于 Transformer 的训练，支持 WebSocket 实时进度、K-Fold 交叉验证和早停机制
- **注意力分析** — 热力图可视化和协同/拮抗成分网络图
- **模型评估** — K-Fold 指标（R²、MSE、MAE）、残差分布和损失曲线
- **表达式推导** — 通过 PySR 进行符号回归，支持化简、优化（Pareto 前沿导航）和撤销/重做历史

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **桌面壳** | [Tauri 2](https://tauri.app) | 跨平台桌面打包 |
| **前端** | [Vue 3](https://vuejs.org) + TypeScript | SPA 单页应用 |
| **可视化** | [ECharts](https://echarts.apache.org) + [D3.js](https://d3js.org) | 热力图、网络图、图表 |
| **数学渲染** | [KaTeX](https://katex.org) | LaTeX 表达式渲染 |
| **后端** | [FastAPI](https://fastapi.tiangolo.com) | REST API + WebSocket |
| **机器学习** | [PyTorch](https://pytorch.org) | 带注意力的 Feature Transformer |
| **符号回归** | [PySR](https://astroautomata.com/PySR/) | 可解释方程发现 |
| **数据处理** | [pandas](https://pandas.pydata.org) + [scikit-learn](https://scikit-learn.org) | 数据处理与评估 |

---

## 快速开始

### 环境要求

- **Python 3.12+** 及 [uv](https://docs.astral.sh/uv/)
- **Node.js 18+** 及 npm
- **Rust**（用于 Tauri 桌面打包，可选）

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/ChHsiching/pharmalink-analyzer.git
cd pharmalink-analyzer

# 安装前端依赖
cd src-frontend && npm install && cd ..

# 安装后端依赖
cd src-backend && uv sync && cd ..

# 同时启动前后端
npm run dev
```

访问地址：
- **前端：** http://localhost:5173
- **后端 API：** http://localhost:8765/api/v1
- **API 文档：** http://localhost:8765/docs

### 运行测试

```bash
# 后端测试（167 个测试用例）
cd src-backend && uv run pytest tests/ -v --timeout=60

# 前端类型检查
cd src-frontend && npx vue-tsc --noEmit
```

---

## 项目结构

```
pharmalink-analyzer/
├── src-frontend/              # Vue 3 + TypeScript 前端
│   └── src/
│       ├── App.vue            # 根组件，工作流初始化
│       ├── composables/       # Vue Composition API hooks
│       │   ├── useApi.ts      # API 客户端 + safeRequest 辅助函数
│       │   ├── useDatasets.ts
│       │   ├── useTraining.ts # 训练 + WebSocket 进度
│       │   ├── useAnalysis.ts
│       │   ├── useEvaluation.ts
│       │   ├── useExpression.ts
│       │   └── useWorkflow.ts # 跨模块状态协调
│       ├── views/             # 页面级组件
│       └── components/        # 共享 UI 组件
│
├── src-backend/               # Python FastAPI 后端
│   └── app/
│       ├── api/               # REST API 路由（FastAPI Depends 注入）
│       ├── services/          # 业务逻辑层
│       │   ├── checkpoint_resolver.py  # 集中化 checkpoint 操作
│       │   ├── data_loader.py          # 数据集加载与缓存
│       │   ├── analysis.py             # 注意力分析服务
│       │   ├── evaluation.py           # 模型评估服务
│       │   ├── expression.py           # 符号回归服务
│       │   └── training.py             # 模型训练服务
│       ├── ml/                # 机器学习模块
│       │   ├── transformer.py          # Feature Transformer 模型
│       │   ├── attention_extractor.py  # 注意力权重提取
│       │   ├── evaluator.py            # K-Fold 评估流水线
│       │   └── symbolic_regressor.py   # PySR 集成
│       ├── models/            # Pydantic 请求/响应模型
│       ├── dependencies.py    # FastAPI 依赖注入工厂
│       └── exceptions.py      # 领域异常层次
│
├── docs/                      # 设计文档、规格说明和计划
└── package.json               # 开发脚本编排
```

---

## API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| **数据集** | | |
| GET | `/api/v1/datasets` | 列出可用数据集 |
| POST | `/api/v1/datasets/upload` | 上传 CSV 数据集 |
| **训练** | | |
| POST | `/api/v1/models/train` | 开始模型训练 |
| GET | `/api/v1/models/train/status` | 获取训练状态 |
| POST | `/api/v1/models/train/stop` | 停止训练 |
| WS | `/ws/train` | 实时训练进度 |
| GET | `/api/v1/models/checkpoints` | 列出已保存的 checkpoint |
| **分析** | | |
| GET | `/api/v1/analysis/attention/{model_id}` | 获取注意力矩阵 |
| GET | `/api/v1/analysis/attention/{model_id}/heatmap` | 获取热力图数据 |
| GET | `/api/v1/analysis/network/{model_id}` | 获取交互网络 |
| **评估** | | |
| GET | `/api/v1/evaluation/metrics/{model_id}` | K-Fold 指标 |
| GET | `/api/v1/evaluation/predictions/{model_id}` | 实际值 vs 预测值 |
| GET | `/api/v1/evaluation/residuals/{model_id}` | 残差分布 |
| GET | `/api/v1/evaluation/loss-curve/{model_id}` | 训练损失曲线 |
| **表达式** | | |
| POST | `/api/v1/expressions/generate/{model_id}` | 生成表达式 |
| POST | `/api/v1/expressions/simplify/{expr_id}` | 化简表达式 |
| POST | `/api/v1/expressions/optimize/{expr_id}` | Pareto 前沿导航 |
| GET | `/api/v1/expressions/tree/{expr_id}` | 获取表达式树 |
| GET | `/api/v1/expressions/history/{expr_id}` | 操作历史 |
| POST | `/api/v1/expressions/undo/{expr_id}` | 撤销操作 |

---

## 构建与部署

### 构建前端

```bash
cd src-frontend && npm run build
```

### 构建桌面应用（Tauri）

```bash
npm run tauri:build
```

生成平台特定的安装包（Windows `.msi`/`.exe`、macOS `.dmg`、Linux `.AppImage`）。

---

## 架构

后端遵循**分层架构**，依赖边界清晰：

- **API 层** (`app/api/`) — FastAPI 路由，使用 `Depends()` 注入
- **服务层** (`app/services/`) — 业务逻辑，依赖 `CheckpointResolver`
- **ML 层** (`app/ml/`) — 纯 ML 操作（模型加载、注意力提取、符号回归）
- **领域异常** (`app/exceptions.py`) — 服务层错误与 FastAPI 解耦

所有服务通过构造函数注入接收依赖（`CheckpointResolver`），支持通过 FastAPI 的 `app.dependency_overrides` 进行干净的测试隔离。

---

## 许可证

本项目基于 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。
