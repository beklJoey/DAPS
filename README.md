# End-to-End Financial Time Series Forecasting & Backtesting System

[中文](#中文版) | [English](#english-version)

---

# 中文版

## 项目简介

本项目构建了一个面向金融时间序列的**端到端机器学习预测与回测系统**，使用 S&P 500 与 VIX 市场数据生成 **Sell / Hold / Buy（卖出 / 持有 / 买入）** 三分类交易信号。

项目不仅关注模型本身的分类性能，还将数据获取、持久化存储、数据验证、特征工程、模型训练、预测、因果回测以及交易成本分析整合为一套完整且可复现的机器学习流水线。

系统主要比较两种模型：

* **Logistic Regression**：作为可解释的线性基线模型
* **LSTM**：用于学习金融时间序列中的时序特征

项目重点关注：

* 时间序列中的 **Data Leakage 防止**
* 实验结果的 **可复现性**
* 多随机种子下的 **模型稳定性**
* 分类指标与实际交易收益之间的差异
* 交易成本下的策略鲁棒性

---

## 系统流程

```text
Market Data Acquisition
        ↓
SQLite Persistent Cache
        ↓
Data Validation & Audit
        ↓
Feature / Label Construction
        ↓
Chronological Train / Validation / Test Split
        ↓
Window Construction
        ↓
Logistic Regression / LSTM Training
        ↓
Trading Signal Inference
        ↓
Lagged-Position Backtesting
        ↓
Performance Evaluation
```

整个系统采用模块化设计，使数据处理、模型训练和回测结果可以独立检查和复现。

---

## 数据

项目使用两类日频金融市场数据：

* **S&P 500 Index (`^GSPC`)**
* **CBOE Volatility Index (`^VIX`)**

数据时间范围：

```text
2020-01-01 → 2025-12-31
```

原始数据通过 `yfinance` 获取，并存储在本地 **SQLite 数据库**中。

经过数据清洗与验证后，最终数据集包含约：

```text
1,473 daily observations
```

使用 SQLite 缓存而不是每次重新下载数据，可以：

* 固定实验所使用的数据快照
* 减少对外部 API 的依赖
* 保证不同实验之间输入数据一致
* 提高实验可复现性

---

## 数据验证

模型训练前，系统会对原始数据进行完整性检查，包括：

* Missing values
* Duplicate timestamps
* 时间序列对齐
* 极端值诊断
* Feature / Label index alignment

验证结果可以保存为：

```text
validation_report.json
```

系统采用 **fail-fast** 机制，使数据问题在进入模型之前被发现，而不是在训练或回测阶段静默传播。

---

## 特征工程

模型使用五个主要特征：

```text
ret_1
ret_5
ma_ratio
vol_10
vix
```

这些特征用于表示：

* 短期收益变化
* 中短期价格趋势
* 移动平均关系
* 历史波动率
* 整体市场波动环境

其中：

```text
ma_ratio = MA_10 / MA_30 - 1
```

所有输入特征均严格使用预测时点之前的信息构造。

对于 Logistic Regression，标准化参数只使用训练集拟合，并固定用于 Validation 和 Test 数据，从而避免未来信息泄漏。

---

## 标签设计

项目将金融预测任务定义为三分类问题。

根据未来 **5 个交易日收益率**构造标签：

```text
Return > +δ  → Buy
Return < -δ  → Sell
Otherwise    → Hold
```

其中：

```text
Prediction Horizon H = 5
```

通过设置阈值，可以过滤幅度较小的市场波动，避免模型将轻微价格变化解释为需要立即交易的信号。

---

## 时间序列划分

为了避免随机划分造成未来信息泄漏，本项目采用严格的时间顺序进行数据划分：

```text
Training:   up to 2023-12-31
Validation: 2024
Testing:    2025
```

LSTM 的历史输入窗口为：

```text
Lookback L = 30 days
Prediction Horizon H = 5 days
```

即模型使用过去 30 个交易日的信息进行预测。

---

## 模型

### Logistic Regression

Logistic Regression 作为低复杂度的基线模型，用于判断更复杂的时序神经网络是否真正学习到了额外的信息。

---

### LSTM

LSTM 模型结构：

```text
5 Input Features
        ↓
Linear Layer
5 → 128
        ↓
LSTM
128 hidden units
        ↓
LSTM
128 hidden units
        ↓
Linear Layer
128 → 3
        ↓
Sell / Hold / Buy
```

主要训练配置：

```text
Optimizer: Adam
Learning Rate: 1e-3
Loss: Cross-Entropy
Early Stopping: Validation-based
```

---

## 多随机种子实验

为了减少神经网络随机初始化对实验结果的影响，完整训练和测试流程在多个随机种子下重复运行：

```text
42
43
44
45
46
```

最终通过多个实验结果的均值和方差评估模型，而不是只报告一次最优结果。

---

## 模型评价

分类性能主要使用：

* Accuracy
* Macro-F1
* Class-wise Recall
* Confusion Matrix
* Normalised Confusion Matrix

由于 Sell / Hold / Buy 存在类别不平衡问题，因此项目更加关注 **Macro-F1** 和 **Class-wise Recall**，避免整体 Accuracy 被多数类别主导。

---

## 特征消融实验

项目设计了 Feature Ablation 接口，通过将指定特征维度置零，在保持网络输入结构不变的情况下分析不同特征的重要性。

测试的特征包括：

```text
VIX
vol_10
ma_ratio
```

实验结果：

| Setting       |      Macro-F1 | Final Net Value |
| ------------- | ------------: | --------------: |
| Full Features | 0.344 ± 0.031 |   1.122 ± 0.020 |
| Drop VIX      | 0.190 ± 0.000 |   1.122 ± 0.000 |
| Drop vol_10   | 0.322 ± 0.035 |   1.099 ± 0.022 |
| Drop ma_ratio | 0.326 ± 0.040 |   1.111 ± 0.021 |

可以看到，移除 VIX 后 Macro-F1 明显下降，说明外部市场波动信息对不同交易类别之间的平衡判断具有重要作用。

---

## 因果回测

分类结果会进一步转换为实际交易仓位，并通过回测系统进行评价。

为了防止同日信息泄漏，系统采用：

```text
Prediction at day t
        ↓
Execution at day t + 1
```

即第 `t` 天产生交易决策，并在第 `t+1` 天执行。

这种 **lagged-position backtesting** 设计保证了整个交易评估过程保持因果关系。

---

## Benchmark

模型策略与以下基准进行比较：

* **Always Hold**
* **Buy and Hold**
* **Logistic Regression**
* **LSTM Trading Policy**

这些基准用于判断策略收益究竟来自模型的有效预测，还是仅仅来自测试阶段市场整体上涨带来的长期暴露。

---

## 交易成本分析

为了更加接近真实交易环境，系统分别测试：

```text
0 bps
5 bps
10 bps
20 bps
```

交易成本。

实验结果：

| Cost   | Final Net Value |  Sharpe Ratio |   Max Drawdown |
| ------ | --------------: | ------------: | -------------: |
| 0 bps  |   1.092 ± 0.006 | 0.723 ± 0.044 | -0.172 ± 0.006 |
| 5 bps  |   1.089 ± 0.006 | 0.714 ± 0.045 | -0.173 ± 0.006 |
| 10 bps |   1.086 ± 0.006 | 0.705 ± 0.045 | -0.174 ± 0.006 |
| 20 bps |   1.080 ± 0.006 | 0.686 ± 0.046 | -0.176 ± 0.006 |

随着交易成本增加，策略收益和 Sharpe Ratio 持续下降，说明交易摩擦是评估模型实际部署价值时不可忽略的重要因素。

---

## 主要发现

### 1. LSTM 可以学习额外的时序信息

与 Logistic Regression 相比，LSTM 在部分方向性判断中具有更好的表现，尤其提高了 **Buy** 类别的识别能力。

### 2. Sell 类别仍是主要瓶颈

由于类别不平衡和金融市场非平稳性，LSTM 对 Sell 类别的识别能力仍然较弱。

这说明单纯增加模型复杂度并不能解决类别不平衡问题。

### 3. 分类指标与交易收益并不完全一致

模型训练优化的是：

```text
Cross-Entropy Classification Loss
```

而真实交易系统关注的是：

```text
Cost-aware Risk-adjusted Return
```

因此 Macro-F1 的提升并不一定会直接转化为更高的最终收益。

### 4. VIX 提供重要的市场状态信息

Feature Ablation 结果显示，移除 VIX 后 Macro-F1 从：

```text
0.344 → 0.190
```

显著下降。

说明市场整体波动环境对于 Sell / Hold / Buy 的平衡决策具有重要作用。

### 5. 可复现性是系统设计的重要组成部分

项目通过：

* SQLite fixed snapshot
* Leak-free chronological splitting
* Multiple random seeds
* Early stopping
* Exported experimental artefacts
* Causal backtesting

保证模型实验具有更好的可重复性和可审计性。

---

## 输出结果

实验过程中会生成包括以下内容在内的结果文件：

```text
market.db
validation_report.json
dataset_report.json
best_model.pt
signals.csv
metrics.json
```

以及：

* Confusion Matrices
* Normalised Confusion Matrices
* Equity Curves
* Validation Loss Curves
* Label Distribution
* Transaction-Cost Sensitivity Curves
* t-SNE Representation Visualisation

---

## 技术栈

```text
Python
PyTorch
scikit-learn
pandas
NumPy
SQLite
yfinance
Matplotlib
```

项目涉及：

* Machine Learning
* Deep Learning
* Financial Time Series
* LSTM
* Feature Engineering
* Data Validation
* Data Leakage Prevention
* Experiment Reproducibility
* Backtesting
* Model Evaluation

---

## 项目定位

本项目不仅实现了一个金融预测模型，而是构建了一套完整的：

```text
Data
 ↓
Model
 ↓
Decision
 ↓
Backtesting
 ↓
Analysis
```

**端到端机器学习系统。**

项目重点体现了从数据工程、模型训练到最终业务指标评价之间的完整 ML workflow，并进一步分析了模型分类指标与真实交易目标之间可能存在的偏差。

---

## 局限与未来改进

当前系统仍属于实验性机器学习与回测平台，而非实际交易系统。

主要局限包括：

* 类别不平衡
* 金融市场非平稳性
* Sell 类别识别能力不足
* 简化的交易执行假设
* 未完整建模 liquidity 与 slippage
* 分类目标与交易收益目标并不完全一致

未来可以进一步探索：

* Class-weighted training
* Focal Loss
* Dynamic decision thresholds
* Cost-sensitive learning
* Regime-aware models
* Transformer-based time-series models
* Walk-forward validation
* 更真实的 liquidity / slippage modelling

---

# English Version

## Overview

This project develops an **end-to-end machine learning forecasting and backtesting system** for financial time-series data.

Using S&P 500 and VIX market data, the system generates three-class trading signals:

```text
Sell / Hold / Buy
```

Rather than focusing only on model prediction performance, the project integrates data acquisition, persistent storage, validation, feature engineering, model training, inference, causal backtesting, and transaction-cost analysis into a reproducible machine learning pipeline.

Two modelling approaches are compared:

* **Logistic Regression** as an interpretable linear baseline
* **LSTM** for temporal representation learning

The project places particular emphasis on:

* preventing **data leakage**;
* ensuring **experimental reproducibility**;
* evaluating robustness across multiple random seeds;
* comparing predictive metrics with actual trading utility;
* testing strategy performance under transaction costs.

---

## System Pipeline

```text
Market Data Acquisition
        ↓
SQLite Persistent Cache
        ↓
Data Validation & Audit
        ↓
Feature / Label Construction
        ↓
Chronological Train / Validation / Test Split
        ↓
Window Construction
        ↓
Logistic Regression / LSTM Training
        ↓
Trading Signal Inference
        ↓
Lagged-Position Backtesting
        ↓
Performance Evaluation
```

The modular pipeline makes individual stages easier to reproduce, inspect, and debug.

---

## Data

The project uses daily financial data from:

* **S&P 500 Index (`^GSPC`)**
* **CBOE Volatility Index (`^VIX`)**

Data coverage:

```text
2020-01-01 → 2025-12-31
```

Market data are acquired through `yfinance` and stored locally using **SQLite**.

After preprocessing and validation, the final dataset contains approximately:

```text
1,473 daily observations
```

Persistent SQLite storage provides fixed experimental snapshots and reduces dependence on repeated upstream API calls.

---

## Data Validation

Before modelling, the pipeline performs integrity checks including:

* missing values;
* duplicate timestamps;
* temporal alignment;
* extreme-value diagnostics;
* feature / label index alignment.

Validation results can be exported to:

```text
validation_report.json
```

A **fail-fast** design prevents invalid data from silently propagating into downstream model training and backtesting.

---

## Feature Engineering

Five primary market features are used:

```text
ret_1
ret_5
ma_ratio
vol_10
vix
```

These capture short-term returns, trend information, historical volatility, and external volatility context.

For example:

```text
ma_ratio = MA_10 / MA_30 - 1
```

All features are constructed strictly using past information.

For Logistic Regression, standardisation parameters are fitted only on the training set and then reused unchanged for validation and testing.

---

## Label Construction

The problem is formulated as three-class classification based on the future **5-day return**:

```text
Return > +δ  → Buy
Return < -δ  → Sell
Otherwise    → Hold
```

with:

```text
Prediction Horizon H = 5
```

The thresholding strategy suppresses small market movements and reduces unnecessary trading signals.

---

## Leak-Free Temporal Splitting

Instead of random splitting, the dataset is divided chronologically:

```text
Training:   up to 2023-12-31
Validation: 2024
Testing:    2025
```

For the LSTM:

```text
Lookback L = 30 days
Prediction Horizon H = 5 days
```

Each prediction is therefore based only on historical observations.

---

## Models

### Logistic Regression

Multinomial Logistic Regression is used as a low-complexity and interpretable baseline.

It helps determine whether the temporal neural network captures useful information beyond linear decision boundaries.

---

### LSTM

The LSTM architecture is:

```text
5 Input Features
        ↓
Linear
5 → 128
        ↓
LSTM
128 hidden units
        ↓
LSTM
128 hidden units
        ↓
Linear
128 → 3
        ↓
Sell / Hold / Buy
```

Training configuration:

```text
Optimizer: Adam
Learning Rate: 1e-3
Loss: Cross-Entropy
Early Stopping: Validation-based
```

---

## Multi-Seed Evaluation

To evaluate robustness against stochastic initialisation, experiments are repeated using:

```text
42
43
44
45
46
```

Performance is aggregated across runs rather than relying on a single favourable training result.

---

## Evaluation

Classification performance is evaluated using:

* Accuracy
* Macro-F1
* Class-wise Recall
* Confusion Matrix
* Normalised Confusion Matrix

Because the Sell / Hold / Buy labels are imbalanced, **Macro-F1** and **class-wise recall** are particularly important for identifying majority-class collapse.

---

## Feature Ablation

A non-destructive feature-ablation interface is implemented by zeroing selected input dimensions while preserving tensor structure.

Features examined include:

```text
VIX
vol_10
ma_ratio
```

Results:

| Setting       |      Macro-F1 | Final Net Value |
| ------------- | ------------: | --------------: |
| Full Features | 0.344 ± 0.031 |   1.122 ± 0.020 |
| Drop VIX      | 0.190 ± 0.000 |   1.122 ± 0.000 |
| Drop vol_10   | 0.322 ± 0.035 |   1.099 ± 0.022 |
| Drop ma_ratio | 0.326 ± 0.040 |   1.111 ± 0.021 |

Removing VIX causes a substantial reduction in Macro-F1, suggesting that external volatility context contributes meaningfully to class-balanced decision boundaries.

---

## Causal Backtesting

Predicted labels are mapped to trading positions and evaluated using an end-to-end backtesting system.

A causal execution convention is enforced:

```text
Prediction at day t
        ↓
Execution at day t + 1
```

This **lagged-position backtesting** design prevents the strategy from using same-day future information during execution.

---

## Benchmarks

The learned strategy is evaluated against:

* **Always Hold**
* **Buy and Hold**
* **Logistic Regression**
* **LSTM Trading Policy**

These controls help distinguish genuine model behaviour from performance caused primarily by passive market exposure.

---

## Transaction-Cost Stress Testing

The strategy is evaluated under multiple transaction-cost assumptions:

```text
0 bps
5 bps
10 bps
20 bps
```

Results:

| Cost   | Final Net Value |  Sharpe Ratio |   Max Drawdown |
| ------ | --------------: | ------------: | -------------: |
| 0 bps  |   1.092 ± 0.006 | 0.723 ± 0.044 | -0.172 ± 0.006 |
| 5 bps  |   1.089 ± 0.006 | 0.714 ± 0.045 | -0.173 ± 0.006 |
| 10 bps |   1.086 ± 0.006 | 0.705 ± 0.045 | -0.174 ± 0.006 |
| 20 bps |   1.080 ± 0.006 | 0.686 ± 0.046 | -0.176 ± 0.006 |

Performance decreases progressively as transaction costs increase, highlighting the importance of deployment-aware evaluation.

---

## Key Findings

### 1. Temporal modelling improves directional behaviour

Compared with Logistic Regression, the LSTM learns additional temporal structure and improves some directional predictions, particularly for the **Buy** class.

### 2. Sell remains the main failure mode

Class imbalance and market non-stationarity result in poor utilisation of the minority **Sell** class.

Increasing model capacity alone is therefore insufficient to solve the imbalance problem.

### 3. Classification performance is not equivalent to trading utility

The model optimises:

```text
Cross-Entropy Classification Loss
```

while deployment ultimately cares about:

```text
Cost-aware Risk-adjusted Return
```

An improvement in Macro-F1 therefore does not necessarily translate directly into higher trading returns.

### 4. VIX provides useful market-regime information

Removing VIX reduces Macro-F1 from approximately:

```text
0.344 → 0.190
```

indicating that external volatility context contributes substantially to balanced classification performance.

### 5. Reproducibility is part of model quality

The system improves reproducibility through:

* fixed SQLite snapshots;
* leak-free chronological splitting;
* multiple random seeds;
* validation-based early stopping;
* exported experimental artefacts;
* causal backtesting.

---

## Output Artefacts

The pipeline generates experimental outputs such as:

```text
market.db
validation_report.json
dataset_report.json
best_model.pt
signals.csv
metrics.json
```

along with visualisations including:

* Confusion Matrices
* Normalised Confusion Matrices
* Equity Curves
* Validation Loss Curves
* Label Distribution Plots
* Transaction-Cost Sensitivity Curves
* t-SNE Representation Visualisations

---

## Tech Stack

```text
Python
PyTorch
scikit-learn
pandas
NumPy
SQLite
yfinance
Matplotlib
```

Core areas covered include:

* Machine Learning
* Deep Learning
* Financial Time Series
* LSTM
* Feature Engineering
* Data Validation
* Data Leakage Prevention
* Experiment Reproducibility
* Backtesting
* Model Evaluation

---

## Project Scope

This project is not only a financial prediction model. It demonstrates a complete:

```text
Data
 ↓
Model
 ↓
Decision
 ↓
Backtesting
 ↓
Analysis
```

**end-to-end machine learning workflow**.

The system connects data engineering, temporal modelling, model diagnostics, and deployment-oriented evaluation while highlighting the gap that can exist between machine-learning metrics and real-world decision objectives.

---

## Limitations & Future Work

The current system is an experimental research and backtesting pipeline rather than a production trading platform.

Current limitations include:

* class imbalance;
* financial non-stationarity;
* poor Sell-class utilisation;
* simplified execution assumptions;
* limited modelling of liquidity and slippage;
* misalignment between classification and trading objectives.

Potential extensions include:

* class-weighted training;
* focal loss;
* dynamic decision thresholds;
* cost-sensitive learning;
* regime-aware models;
* Transformer-based time-series models;
* walk-forward validation;
* more realistic liquidity and slippage modelling.

---

## Disclaimer

This repository is intended for **research and educational purposes only**.

The results are based on historical data and simplified backtesting assumptions. They should not be interpreted as financial advice or as evidence of guaranteed future trading performance.

本项目仅用于**研究和学习目的**，不构成任何投资建议。

* Failure cases
* Experiment logs

---

## Key Takeaway

The main finding of this project is not simply that adding more category-aware information improves performance.

Instead:

> **How category-aware evidence is integrated matters more than simply adding it.**

Direct pooling provides almost no improvement over a strong Category-Concat BERT baseline, while gated fusion and residual logit correction achieve stronger performance and improved cross-seed stability within the evaluated setup.
