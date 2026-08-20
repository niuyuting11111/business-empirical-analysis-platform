# 部署清单：GitHub + Streamlit Community Cloud

本文件列出了将「商科实证数据分析平台」从本地部署到公网所需的全部操作步骤。
建议按顺序执行；每完成一步可勾选 `[x]`。

---

## 前置条件

- [ ] 已安装 Git
- [ ] 拥有 GitHub 账号（https://github.com）
- [ ] 拥有 Streamlit 账号（用 GitHub 登录 https://share.streamlit.io 即可）

---

## 第 1 步：在本地整理仓库（你已完成大部分）

确认仓库根目录已包含以下文件（部署必需）：

- [x] `app.py` —— Streamlit 入口
- [x] `requirements.txt` —— 锁定版本依赖
- [x] `README.md`
- [x] `DEPLOY.md`
- [x] `.gitignore` —— 已忽略密钥 / 缓存 / 大数据文件
- [x] `.python-version` —— 指定 Python 3.12，避免 Cloud 默认使用 3.14 导致科学计算包编译失败
- [ ] `.streamlit/secrets.toml` —— **仅本地用，已被 .gitignore 忽略，不要提交**

> 验证 `.gitignore` 是否生效：
> ```bash
> git status --ignored
> ```
> 应能看到 `.streamlit/secrets.toml`、`saved_*.parquet`、`*.xlsx` 等被标记为 ignored。

---

## 第 2 步：初始化 Git 并提交（如尚未初始化）

> 在本机项目根目录 `/Users/niuyuting/Desktop/AI_Thesis_Work/` 执行：

```bash
git init
git add app.py requirements.txt README.md DEPLOY.md .gitignore
# 注意：不要 git add .streamlit/secrets.toml（已被忽略）
git commit -m "feat: 商科实证数据分析平台 初始部署版本"
```

> 如需确认密钥未被提交：
> ```bash
> git ls-files | grep secrets.toml   # 应无输出
> ```

---

## 第 3 步：推送到 GitHub

1. 在 GitHub 上新建一个 **Public**（或 Private）仓库，例如 `empirical-finance-platform`。
   - 新建时 **不要** 勾选 "Add a README file" / "Add .gitignore"（本地已有）。
2. 关联并推送：

```bash
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

3. 刷新 GitHub 页面，确认 `app.py`、`requirements.txt`、`README.md`、`DEPLOY.md`、`.gitignore` 已出现，
   且 **没有** `.streamlit/secrets.toml`、没有大体积 `.xlsx` 文件。

---

## 第 4 步：在 Streamlit Community Cloud 部署

1. 打开 https://share.streamlit.io （或 https://streamlit.io/cloud ），用 GitHub 登录。
2. 点击 **"New app"** （或 "Create app"）。
3. 配置：
   - **Repository**：选择第 3 步推送的仓库
   - **Branch**：`main`
   - **Main file path**：`app.py`
   - **Python version**：仓库根目录已放置 `.python-version` 文件指定为 `3.12`，Cloud 会自动读取
4. 配置 **Secrets**（关键）：
   - 展开 **"Advanced settings" → Secrets**
   - 将本地 `.streamlit/secrets.toml` 的内容粘贴进去，例如：
     ```toml
     OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
     ```
   - 若暂不使用 AI 功能，可先留空或填占位符，应用不会因此崩溃。
5. 点击 **"Deploy!"**。

---

## 第 5 步：等待构建并验证

- Streamlit Cloud 会自动 `pip install -r requirements.txt`（首次约 2–5 分钟）。
- 部署完成后页面自动打开。请做一次冒烟测试：
  - [ ] 页面正常加载，标题为「通用论文数据平台」
  - [ ] 左侧 15 个分析章节可正常切换
  - [ ] 上传一个 Excel/CSV，进入「数据清洗」可正常运行
- 若构建失败，查看 Cloud 的 **Logs / Manage app → Logs**，常见原因与对策见下方「排错」。

---

## 第 6 步：后续维护

- 修改代码后，本地 `git commit` + `git push`，Cloud 会 **自动重新部署**。
- 更新依赖版本时，同步修改 `requirements.txt` 并重新 push。
- 更换密钥：在 Cloud 的 App → Settings → Secrets 中编辑，无需改代码。

---

## 🔧 排错指南

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `ModuleNotFoundError: No module named 'xxx'` | `requirements.txt` 漏写依赖 | 在本地补上对应包及版本，重新 push |
| 安装超时 / 失败 | 某个包版本在 PyPI 不存在 | 核对 `requirements.txt` 中版本号是否在 PyPI 真实存在 |
| `Error installing requirements` / 编译 pandas/numpy 失败 | Streamlit Cloud 默认 Python 版本过新（如 3.14），科学计算包尚无 wheels | 在仓库根目录创建 `.python-version` 文件并写入 `3.12`，重新 push 后重启部署 |
| `FileNotFoundError` 读取本地文件 | 代码引用了仓库中不存在的文件 | 确保所有读写使用相对路径或用户上传，不要依赖本地绝对路径 |
| 页面空白 / 报错 `set_page_config` 必须在最前 | 在 `st.set_page_config` 之前调用了其他 `st.*` | 本仓库已确认 `set_page_config` 是第一个 Streamlit 调用，无需修改 |
| Secrets 未生效 | 未粘贴到 Cloud Secrets，或格式错误 | 确认粘贴内容与 `secrets.toml` 一致，TOML 语法正确 |
| 应用能跑但某模块报错 | 上传数据缺少必要变量 | 按模块提示正确映射「被解释变量 / 解释变量 / 控制变量 / ID / 年份」列 |

---

## ✅ 部署兼容性已确认项（本仓库现状）

- [x] `st.set_page_config` 是 `app.py` 中第一个 Streamlit 调用
- [x] 所有文件路径均为相对路径 / 内存对象（BytesIO），无硬编码绝对路径
- [x] API Key 通过 `st.secrets["OPENAI_API_KEY"]` 读取（当前无 AI 功能，模板已预置）
- [x] `requirements.txt` 所有版本均已锁定，并在 PyPI 验证存在
- [x] `.python-version` 已指定 Python 3.12，确保 Streamlit Cloud 使用与科学计算包兼容的解释器
- [x] `.gitignore` 已忽略 `secrets.toml`、`.env`、`__pycache__`、`*.pyc`、`venv/`
- [x] 异常处理完整，关键计算均有 `try/except` 兜底
- [x] 无 AI 功能时应用可正常降级运行，不崩溃
