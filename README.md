# 商科实证论文一站式数据分析平台

> 面向商科 / 经管类实证研究（学士、硕士、博士论文及期刊投稿）的 **Streamlit 数据分析 Web 应用**。
> 上传你的 Excel / CSV，通过点点选选即可完成从 **数据清洗 → 描述性统计 → 各类计量模型 → 综合评价** 的全流程分析，并一键导出结果。

---

## 📌 项目简介

本平台将商科实证论文中最常用的 20+ 种研究方法封装为 15 个分析章节、21 项功能模块，
无需写代码，只需在浏览器中配置变量、点击运行即可得到规范的分析结果与可导出的 Excel 报告。

- **零代码**：所有模型通过下拉框 / 多选框配置，无需 Python 基础
- **全流程**：数据清洗、变量映射、缺失值处理、缩尾、标准化一步到位
- **方法全**：涵盖经典回归、内生性、DID、RDD、空间计量、机器学习因果、综合评价等
- **可复现**：所有依赖版本已锁定在 `requirements.txt`，部署即用

---

## 🧩 功能模块（21 项）

平台以 **15 个分析章节** 组织，下列按方法拆分为 **21 项核心功能模块**：

| # | 功能模块 | 对应章节 |
|---|----------|----------|
| 1 | 数据清洗与预处理（缺失值 / 异常值 / 缩尾 / 标准化 / 变量映射） | 第 1 章 |
| 2 | 描述性统计（均值 / 标准差 / 分位数 / 分组统计） | 第 2 章 |
| 3 | 相关性矩阵分析（Pearson / Spearman / Kendall + 热力图） | 第 2 章 |
| 4 | 多重共线性检验（VIF） | 第 2 章 |
| 5 | 异方差与自相关诊断（White / BP / ARCH / Jarque-Bera） | 第 2 章 |
| 6 | OLS / 固定效应 / 随机效应回归（稳健 / 聚类标准误） | 第 3 章 |
| 7 | Logit / Probit 二值选择模型 | 第 3 章 |
| 8 | 分位数回归 | 第 3 章 |
| 9 | Lasso / ElasticNet 正则化变量选择 | 第 3 章 |
| 10 | GLM 广义线性模型（Binomial / Poisson / Gamma） | 第 3 章 |
| 11 | 工具变量 / 系统 GMM（内生性处理） | 第 3 章 |
| 12 | Heckman 样本选择模型 | 第 3 章 |
| 13 | 熵权法 + TOPSIS 综合评价 | 第 4 章 |
| 14 | CRITIC 客观赋权 | 第 4 章 |
| 15 | 主成分分析 / 因子分析（PCA / FA） | 第 4 章 |
| 16 | DEA 数据包络分析 | 第 4 章 |
| 17 | 耦合协调度模型 | 第 5 章 |
| 18 | DID + 事件研究法（含 PSM-DID） | 第 7 章 |
| 19 | RDD 断点回归（Sharp / Fuzzy） | 第 8 章 |
| 20 | 机制分析（中介效应 / 调节效应 / 门槛效应） | 第 9 章 |
| 21 | 空间计量（SLM / SEM / SDM） | 第 10 章 |

**进阶章节（同样内置）：** 时间序列分析（ADF 单位根 / ARIMA / VAR / 格兰杰因果）、
结构方程模型（SEM）、双重机器学习与因果森林（DML / Causal Forest）、
多层线性模型（HLM）、P2 综合评价进阶（模糊综合评价 / 可变权重 / AHP）。

---

## 🚀 本地运行

```bash
# 1. 建议使用虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
streamlit run app.py
```

启动后浏览器自动打开 `http://localhost:8501`。
如需配置密钥（例如后续接入 AI 功能），在 `.streamlit/secrets.toml` 中填写（参见该文件模板）。

---

## ☁️ 部署到 Streamlit Community Cloud

无需自有服务器，免费托管。详见仓库根目录 **`DEPLOY.md`**，核心步骤：

1. 将本仓库推送到 GitHub
2. 打开 https://share.streamlit.io （或 Streamlit Cloud 控制台）
3. 选择仓库与 `app.py` 作为入口，粘贴 `secrets.toml` 内容
4. 点击 Deploy，等待依赖安装完成

---

## 📁 仓库结构

```
.
├── app.py                  # Streamlit 入口（全部 15 章节 / 21 模块逻辑）
├── requirements.txt        # 锁定的依赖版本（部署必需）
├── .gitignore              # 忽略密钥、缓存、数据等大文件
├── .streamlit/
│   └── secrets.toml        # 本地密钥模板（已被 .gitignore 忽略，不入库）
├── README.md               # 本文件
└── DEPLOY.md               # GitHub + Streamlit Cloud 部署操作清单
```

---

## ⚠️ 注意事项

- 本地自动保存的清洗结果（`saved_*.parquet` / `saved_*.json`）与用户私有研究数据
  （`*.xlsx` / `*.csv`）均已被 `.gitignore` 忽略，**不会提交到 GitHub**。
- 上传的数据仅保留在会话内存中，不会上传到任何第三方服务器。
- 若后续接入 AI 辅助解读功能，请通过 `st.secrets["OPENAI_API_KEY"]` 读取密钥，
  并在 Streamlit Cloud 的 Secrets 设置中配置，切勿硬编码。
