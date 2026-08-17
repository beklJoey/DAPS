# GCAP-BERT: Category-Aware Transformer for Aspect-Based Sentiment Analysis

[中文](#中文版) | [English](#english-version)

---

# 中文版

## 项目简介

本项目基于 **MAMS-ACSA** 数据集研究方面类别级情感分类（Aspect-Category Sentiment Classification, ACSA）。

与普通的句子级情感分类不同，ACSA 需要针对一个**指定的方面类别（Aspect Category）**判断对应的情感倾向。例如：

```text
Sentence:
"The food was excellent, but the service was painfully slow."

Category:
food

Prediction:
positive
```

对于同一句评论，当目标类别变为：

```text
Category:
service

Prediction:
negative
```

因此，该任务的核心不仅是识别情感，还需要将**情感证据与正确的方面类别进行对齐**。

本项目围绕以下问题展开：

> 当 BERT 已经通过 sentence-pair 输入获得目标类别信息后，显式提取类别相关证据是否还能进一步提升模型性能？如果可以，这些证据应该如何与全局语义表示进行融合？

为研究这一问题，项目构建了从传统机器学习模型到多种 BERT 架构的对照实验，并进一步设计 **GCAP-BERT（Gated Category-Aware Pooling with Residual Logit Correction）**，通过类别感知多头注意力、门控特征融合和残差 Logit 校正控制局部类别证据对最终预测的影响。

---

## 模型演进

项目在统一的数据处理、训练和评估流程下比较了以下模型：

1. Majority Baseline
2. TF-IDF + Logistic Regression
3. Sentence-only BERT
4. Category-Concat BERT
5. Direct Category-Aware Pooling BERT
6. **GCAP-BERT**

整个实验主要研究三个问题：

* 显式加入目标类别信息对情感分类有多重要？
* 当 BERT 已经获得类别信息后，显式 Category-Aware Pooling 是否仍然有效？
* 与直接拼接相比，受控的特征融合是否能够更有效地利用类别相关证据？

---

## GCAP-BERT 架构

模型采用 BERT sentence-pair 输入：

```text
[CLS] sentence [SEP] category [SEP]
```

经过 BERT Encoder 后得到：

* `h_CLS`：sentence-category pair 的全局表示
* `H_sent`：sentence tokens 的 hidden states
* `H_cat`：category tokens 的 hidden states

### 1. Category-Aware Pooling

首先对 category token 的表示进行平均池化：

```text
q_cat = mean(H_cat)
```

得到类别查询向量 `q_cat`。

随后以 `q_cat` 作为 Query，对句子 token 表示进行 Multi-Head Attention：

```text
p_c = MHA(
    query = q_cat,
    key   = H_sent,
    value = H_sent
)
```

最终得到 `p_c`，表示模型从当前句子中提取出的**类别相关证据**。

---

### 2. Gated Fusion

GCAP-BERT 不直接将 `p_c` 与 `[CLS]` 表示拼接，而是学习一个门控向量：

```text
g = sigmoid(W_g [h_CLS ; p_c])
```

随后得到融合后的表示：

```text
h_fused = g ⊙ p_c + (1 - g) ⊙ h_CLS
```

该机制允许模型根据不同隐藏维度，自适应控制：

* 使用多少类别相关局部信息 `p_c`
* 保留多少全局 BERT 表示 `h_CLS`

---

### 3. Residual Logit Correction

GCAP-BERT 保留了一条基于 `[CLS]` 的主预测分支：

```text
logits_cls = W_cls h_CLS + b_cls
```

同时使用：

```text
[h_CLS ; p_c ; h_fused]
```

构建额外的 Pooling 分支：

```text
logits_pool = MLP([h_CLS ; p_c ; h_fused])
```

最终预测：

```text
logits = logits_cls + α · logits_pool
```

其中 `α` 为可学习参数，并初始化为：

```text
α = 0.0
```

这种设计使模型从较稳定的 BERT 全局预测路径开始训练，再逐渐学习是否需要利用类别感知分支对最终预测进行修正。

---

## 实验结果

主要神经网络模型均在三个随机种子 `42 / 43 / 44` 下进行实验。

| Model                        |   Accuracy |   Macro-F1 |       Std. |
| ---------------------------- | ---------: | ---------: | ---------: |
| Majority                     |     0.4362 |     0.2025 |     0.0000 |
| TF-IDF + Logistic Regression |     0.4595 |     0.4594 |     0.0000 |
| BERT Sentence-Only           |     0.4680 |     0.4562 |     0.0181 |
| BERT Category-Concat         |     0.7788 |     0.7769 |     0.0138 |
| BERT Category-Pooling        |     0.7788 |     0.7774 |     0.0136 |
| **GCAP-BERT**                | **0.8047** | **0.8022** | **0.0037** |

实验中最大的性能提升来自显式加入目标类别信息：

```text
Sentence-only BERT
Macro-F1 = 0.4562

        ↓

Category-Concat BERT
Macro-F1 = 0.7769
```

直接加入 Category-Aware Pooling 后：

```text
0.7769 → 0.7774
```

提升非常有限。

使用门控融合与 Residual Logit Correction 后：

```text
0.7769 → 0.8022
```

同时跨随机种子 Macro-F1 标准差：

```text
0.0138 → 0.0037
```

实验表明：

> **额外提取类别相关证据本身并不足以保证性能提升，如何控制这些证据与全局语义表示之间的融合方式同样重要。**

---

## 数据集

项目使用 **MAMS-ACSA** 数据集。

处理后的数据规模：

```text
Training:    7,090
Validation:    888
Test:          901

Total:       8,879
```

预测类别包括：

```text
negative
neutral
positive
```

项目保留官方提供的 train / validation / test 数据划分。

---

## 技术栈

* Python
* PyTorch
* Hugging Face Transformers
* BERT (`bert-base-uncased`)
* Multi-Head Attention
* Scikit-learn
* Pandas
* NumPy
* SQLite
* Matplotlib

---

## 项目结构

```text
.
├── main.py
├── environment.yml
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── parse_mams.py
│   ├── validate.py
│   ├── storage.py
│   ├── datasets.py
│   ├── features.py
│   ├── preprocess.py
│   ├── models_tfidf.py
│   ├── models_bert.py
│   ├── models_bert_v2.py
│   ├── evaluate.py
│   ├── failure_analysis.py
│   ├── scoring.py
│   └── visualise.py
│
├── data/
│   └── MAMS-ACSA/
│       └── raw/
│           ├── train.xml
│           ├── val.xml
│           └── test.xml
│
└── artifacts/
    ├── models/
    ├── predictions/
    ├── tables/
    ├── figures/
    └── logs/
```

---

## 环境安装

克隆项目：

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

或者使用：

```bash
pip install -r requirements.txt
```

---

## 数据准备

将 MAMS-ACSA 的原始 XML 文件放置在：

```text
data/
└── MAMS-ACSA/
    └── raw/
        ├── train.xml
        ├── val.xml
        └── test.xml
```

预处理流程会自动完成：

```text
XML Parsing
    ↓
Data Validation
    ↓
Label Filtering
    ↓
CSV / SQLite Storage
```

---

## 运行项目

运行完整实验：

```bash
python main.py
```

完整流程包括：

```text
MAMS XML
    ↓
Parsing & Validation
    ↓
CSV / SQLite Processing
    ↓
Baseline Training
    ↓
BERT Training
    ↓
GCAP-BERT Training
    ↓
Multi-Seed Evaluation
    ↓
Ablation Analysis
    ↓
Failure Analysis
    ↓
Tables / Figures / Logs
```

---

## 实验配置

主要实验参数位于：

```text
src/config.py
```

主要配置包括：

```python
BERT_EPOCHS = 3
BERT_LR = 2e-5
BATCH_SIZE = 16
MAX_LEN = 128

SEEDS = [42, 43, 44]
USE_CLASS_WEIGHTS = True
```

BERT 模型采用统一的训练与评估流程，并使用类别加权交叉熵缓解类别不平衡问题。

---

## 输出结果

实验结果自动保存在：

```text
artifacts/
```

包括：

```text
artifacts/
├── models/
├── predictions/
├── tables/
├── figures/
└── logs/
```

其中包含：

* Model checkpoints
* Test predictions
* Accuracy / Macro-F1
* Multi-seed statistics
* Confusion matrices
* Per-category analysis
* Failure cases
* Experiment logs

---

## 核心结论

本项目最主要的发现并不是：

> “加入更多 category 信息就一定能够提升性能。”

实验结果反而表明：

> **类别相关证据如何被整合，比单纯增加额外特征更加重要。**

Direct Category-Aware Pooling 对 Category-Concat BERT 的提升十分有限，而通过 Gated Fusion 和 Residual Logit Correction 对局部证据进行受控融合后，GCAP-BERT 在当前实验设置下获得了更好的整体性能与跨随机种子稳定性。

---

# English Version

## Overview

This project studies **Aspect-Category Sentiment Classification (ACSA)** on the **MAMS-ACSA** dataset.

Unlike standard sentence-level sentiment classification, ACSA predicts sentiment toward a **specific queried aspect category**.

For example:

```text
Sentence:
"The food was excellent, but the service was painfully slow."

Category:
food

Prediction:
positive
```

For the same sentence:

```text
Category:
service

Prediction:
negative
```

The key challenge is therefore not only identifying sentiment, but also **aligning sentiment evidence with the correct aspect category**.

This project investigates the following question:

> When a BERT sentence-pair model already receives the target aspect category, can explicit category-aware evidence extraction provide additional value, and how should this evidence be integrated?

To study this question, the project builds a controlled progression from traditional machine-learning baselines to category-conditioned BERT variants and introduces **GCAP-BERT — Gated Category-Aware Pooling with Residual Logit Correction**.

---

## Model Progression

The following models are evaluated under a shared preprocessing, training, and evaluation pipeline:

1. Majority Baseline
2. TF-IDF + Logistic Regression
3. Sentence-only BERT
4. Category-Concat BERT
5. Direct Category-Aware Pooling BERT
6. **GCAP-BERT**

The experiments focus on three questions:

* How important is explicit category information?
* Does category-aware pooling improve a strong category-conditioned BERT baseline?
* Does the integration strategy determine whether pooled evidence is useful?

---

## GCAP-BERT Architecture

The model uses BERT sentence-pair input:

```text
[CLS] sentence [SEP] category [SEP]
```

After BERT encoding, three representations are extracted:

* `h_CLS`: global sentence-category representation
* `H_sent`: sentence-token hidden states
* `H_cat`: category-token hidden states

### 1. Category-Aware Pooling

Category-token representations are mean-pooled into a category query:

```text
q_cat = mean(H_cat)
```

The category query then attends over sentence-token representations using multi-head attention:

```text
p_c = MHA(
    query = q_cat,
    key   = H_sent,
    value = H_sent
)
```

The resulting `p_c` represents category-aware evidence extracted from the sentence.

---

### 2. Gated Fusion

Instead of directly concatenating the pooled evidence with the global representation, GCAP-BERT learns a feature-wise gate:

```text
g = sigmoid(W_g [h_CLS ; p_c])
```

The representations are fused as:

```text
h_fused = g ⊙ p_c + (1 - g) ⊙ h_CLS
```

This allows the model to control how strongly category-aware local evidence affects the global BERT representation.

---

### 3. Residual Logit Correction

GCAP-BERT maintains a baseline `[CLS]` prediction pathway:

```text
logits_cls = W_cls h_CLS + b_cls
```

A second branch produces category-aware correction logits:

```text
logits_pool = MLP([h_CLS ; p_c ; h_fused])
```

The final prediction is:

```text
logits = logits_cls + α · logits_pool
```

where `α` is learnable and initialized to:

```text
α = 0.0
```

This design allows training to begin from the global BERT prediction pathway while the category-aware branch learns a controlled residual correction.

---

## Experimental Results

Main results are averaged across three random seeds (`42`, `43`, `44`).

| Model                        |   Accuracy |   Macro-F1 |       Std. |
| ---------------------------- | ---------: | ---------: | ---------: |
| Majority                     |     0.4362 |     0.2025 |     0.0000 |
| TF-IDF + Logistic Regression |     0.4595 |     0.4594 |     0.0000 |
| BERT Sentence-Only           |     0.4680 |     0.4562 |     0.0181 |
| BERT Category-Concat         |     0.7788 |     0.7769 |     0.0138 |
| BERT Category-Pooling        |     0.7788 |     0.7774 |     0.0136 |
| **GCAP-BERT**                | **0.8047** | **0.8022** | **0.0037** |

The largest performance improvement comes from explicit category conditioning:

```text
Sentence-only BERT
Macro-F1 = 0.4562

        ↓

Category-Concat BERT
Macro-F1 = 0.7769
```

Direct category-aware pooling provides almost no additional gain:

```text
0.7769 → 0.7774
```

GCAP-BERT instead uses gated fusion and residual correction:

```text
0.7769 → 0.8022
```

while reducing the cross-seed Macro-F1 standard deviation:

```text
0.0138 → 0.0037
```

The experiments suggest that:

> **Extracting additional category-aware evidence is not sufficient by itself; how that evidence is integrated with the global representation matters.**

---

## Dataset

The project uses **MAMS-ACSA**.

Processed dataset size:

```text
Training:    7,090
Validation:    888
Test:          901

Total:       8,879
```

Sentiment classes:

```text
negative
neutral
positive
```

The official train / validation / test splits are preserved.

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* BERT (`bert-base-uncased`)
* Multi-Head Attention
* Scikit-learn
* Pandas
* NumPy
* SQLite
* Matplotlib

---

## Project Structure

```text
.
├── main.py
├── environment.yml
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── parse_mams.py
│   ├── validate.py
│   ├── storage.py
│   ├── datasets.py
│   ├── features.py
│   ├── preprocess.py
│   ├── models_tfidf.py
│   ├── models_bert.py
│   ├── models_bert_v2.py
│   ├── evaluate.py
│   ├── failure_analysis.py
│   ├── scoring.py
│   └── visualise.py
│
├── data/
│   └── MAMS-ACSA/
│       └── raw/
│           ├── train.xml
│           ├── val.xml
│           └── test.xml
│
└── artifacts/
    ├── models/
    ├── predictions/
    ├── tables/
    ├── figures/
    └── logs/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Place the raw MAMS-ACSA XML files under:

```text
data/
└── MAMS-ACSA/
    └── raw/
        ├── train.xml
        ├── val.xml
        └── test.xml
```

The preprocessing pipeline automatically performs:

```text
XML Parsing
    ↓
Data Validation
    ↓
Label Filtering
    ↓
CSV / SQLite Storage
```

---

## Running the Project

Run the complete experimental pipeline with:

```bash
python main.py
```

The pipeline covers:

```text
MAMS XML
    ↓
Parsing & Validation
    ↓
CSV / SQLite Processing
    ↓
Baseline Training
    ↓
BERT Training
    ↓
GCAP-BERT Training
    ↓
Multi-Seed Evaluation
    ↓
Ablation Analysis
    ↓
Failure Analysis
    ↓
Tables / Figures / Logs
```

---

## Configuration

Main experiment settings are centralized in:

```text
src/config.py
```

Default settings include:

```python
BERT_EPOCHS = 3
BERT_LR = 2e-5
BATCH_SIZE = 16
MAX_LEN = 128

SEEDS = [42, 43, 44]
USE_CLASS_WEIGHTS = True
```

The BERT-based models follow a shared training and evaluation protocol and use class-weighted cross-entropy to reduce majority-class dominance.

---

## Outputs

Experimental outputs are automatically stored under:

```text
artifacts/
```

including:

```text
artifacts/
├── models/
├── predictions/
├── tables/
├── figures/
└── logs/
```

The pipeline stores:

* Model checkpoints
* Test predictions
* Accuracy and Macro-F1 metrics
* Multi-seed summaries
* Confusion matrices
* Per-category diagnostics
* Failure cases
* Experiment logs

---

## Key Takeaway

The main finding of this project is not simply that adding more category-aware information improves performance.

Instead:

> **How category-aware evidence is integrated matters more than simply adding it.**

Direct pooling provides almost no improvement over a strong Category-Concat BERT baseline, while gated fusion and residual logit correction achieve stronger performance and improved cross-seed stability within the evaluated setup.
