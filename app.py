import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import skew
from linearmodels.panel import PanelOLS, RandomEffects
import warnings
import traceback
import os
import json
import arch.unitroot as au
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

st.set_page_config(page_title="实证派", layout="wide")

# ==================== 公共辅助函数（所有分析页面共享） ====================
def _fmt_coef(param, se, pval):
    """格式化系数为 系数***(标准误) 学术格式。"""
    if param is None or se is None:
        return ""
    try:
        if pd.isna(param) or pd.isna(se):
            return ""
    except Exception:
        pass
    star = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{param:.4f}{star}({se:.4f})"


def _get_se(model, name):
    """兼容 OLS(bse) 与 PanelOLS/IV(std_errors) 的标准误提取。"""
    if hasattr(model, "std_errors"):
        return model.std_errors.get(name, None)
    if hasattr(model, "bse"):
        return model.bse.get(name, None)
    return None


def _show_table(display_df, fname, sheet="结果"):
    """以学术 HTML 表格展示结果，并提供 Excel 下载（聚类稳健结果亦可导出）。"""
    st.markdown(display_df.to_html(index=False, escape=False), unsafe_allow_html=True)
    _buf = BytesIO()
    with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
        display_df.to_excel(_w, index=False, sheet_name=sheet)
    _buf.seek(0)
    st.download_button(
        "📥 下载结果 (Excel)",
        data=_buf.getvalue(),
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{fname}",
    )


def _fe_clusters(d):
    """构造与 d 同索引的个体聚类 DataFrame（linearmodels 7.x 正确用法：clusters 需为带相同索引的 DataFrame）。"""
    return pd.DataFrame({"entity": d.index.get_level_values(0)}, index=d.index)


def _drop_absorbed(d, exog_df, use_entity, use_time):
    """丢弃在固定效应维度上无变异（会被完全吸收）的列，避免 AbsorbingEffectError。"""
    gi = d.index.get_level_values(0)
    gt = d.index.get_level_values(1)
    keep = []
    for c in exog_df.columns:
        v = exog_df[c]
        bad = False
        if use_entity:
            s = v.groupby(gi)
            if (s.transform("max") - s.transform("min")).abs().max() < 1e-12:
                bad = True
        if (not bad) and use_time:
            s = v.groupby(gt)
            if (s.transform("max") - s.transform("min")).abs().max() < 1e-12:
                bad = True
        if not bad:
            keep.append(c)
    return exog_df[keep]


# ==================== AI 智能助手（侧边栏混合模式） ====================
AI_SYSTEM_PROMPT = (
    "你是商科实证分析助手，帮助用户理解和使用网站的21个分析模块。"
    "网站功能包括：数据清洗、描述性统计、基准回归（OLS/FE）、指标测算"
    "（熵权法、TOPSIS、PCA、DEA、CRITIC）、因果推断（IV/2SLS、DID、RDD、PSM、"
    "系统GMM、合成控制法）、机制分析（中介、调节、门槛）、空间计量"
    "（SLM/SEM/SDM）、稳健性检验。请用简洁专业的语言回答用户问题，"
    "必要时引导用户到对应页面操作。"
)

PLATFORM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
PLATFORM_MODEL = "glm-4.7-flash"
PLATFORM_FREE_LIMIT = 15
DEFAULT_BYOK_BASE = "https://api.deepseek.com"
DEFAULT_BYOK_MODEL = "deepseek-v4-flash"


def _resolve_ai_mode():
    """根据 session_state 与 secrets 解析当前调用模式，返回 (api_key, base_url, model, is_free)。"""
    _free = st.session_state.ai_use_platform_free or (not st.session_state.ai_user_api_key.strip())
    if _free:
        _api_key = (st.secrets.get("PLATFORM_API_KEY", "") or "").strip()
        return _api_key, PLATFORM_API_BASE, PLATFORM_MODEL, True
    _api_key = st.session_state.ai_user_api_key.strip()
    _base = st.session_state.ai_user_base_url.strip() or DEFAULT_BYOK_BASE
    _model = st.session_state.ai_user_model.strip() or DEFAULT_BYOK_MODEL
    return _api_key, _base, _model, False


def _handle_ai_prompt(prompt):
    """处理一次用户提问：解析模式、调用模型、流式显示、限流与异常处理。"""
    import json as _json
    import httpx as _httpx

    st.session_state.ai_messages.append({"role": "user", "content": prompt})
    _api_key, _base, _model, _free = _resolve_ai_mode()

    if _free:
        if not _api_key:
            st.session_state.ai_messages.append({
                "role": "assistant",
                "content": "平台免费额度未配置（管理员需在 Streamlit Secrets 中设置 PLATFORM_API_KEY），请填写自己的 API Key 后使用。",
            })
            return
        if st.session_state.ai_platform_calls >= PLATFORM_FREE_LIMIT:
            st.session_state.ai_messages.append({
                "role": "assistant",
                "content": "体验额度已用完，请填写自己的 API Key 继续使用。",
            })
            return

    _history = [{"role": "system", "content": AI_SYSTEM_PROMPT}] + st.session_state.ai_messages
    _payload = {"model": _model, "messages": _history, "stream": True}
    _headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "text/event-stream; charset=utf-8",
    }
    try:
        with _httpx.Client(timeout=60) as _client:
            with _client.stream(
                "POST",
                f"{_base.rstrip('/')}/chat/completions",
                json=_payload,
                headers=_headers,
            ) as _resp:
                # 非 200 时直接把智谱返回的原始错误展示出来，便于定位问题
                if _resp.status_code != 200:
                    _body = _resp.read().decode("utf-8", errors="replace")
                    _tip = (
                        f"⚠️ API 返回 {_resp.status_code} 错误。\n\n"
                        f"原始响应：{_body[:600]}\n\n"
                        f"（如提示模型不存在，请确认 Model={_model}；"
                        f"如提示鉴权失败，请确认 API Key 是否正确）"
                    )
                    st.session_state.ai_messages.append({"role": "assistant", "content": _tip})
                    return

                _reply = ""
                with st.sidebar.chat_message("assistant"):
                    _ph = st.empty()
                    for _line in _resp.iter_lines():
                        if not _line:
                            continue
                        # httpx iter_lines() 已返回 str；做类型兼容避免不同版本差异
                        _text = _line if isinstance(_line, str) else _line.decode("utf-8", errors="replace")
                        if not _text.startswith("data: "):
                            continue
                        _data = _text[6:]
                        if _data == "[DONE]":
                            break
                        try:
                            _chunk = _json.loads(_data)
                        except _json.JSONDecodeError:
                            continue
                        _delta = (_chunk.get("choices", [{}])[0].get("delta") or {}).get("content")
                        if _delta:
                            _reply += _delta
                            _ph.markdown(_reply + "▌")
                    _ph.markdown(_reply)
                st.session_state.ai_messages.append({"role": "assistant", "content": _reply})
                if _free:
                    st.session_state.ai_platform_calls += 1
    except Exception as _e:  # noqa: BLE001 - 统一兜底，避免整页崩溃
        # 暴露真实异常原文，避免误判（如把鉴权失败错当成模型不存在）
        _tip = f"⚠️ AI 调用出现异常：{_e}\n\n（请截图发我以便定位）"
        st.session_state.ai_messages.append({"role": "assistant", "content": _tip})


def render_ai_assistant():
    """在侧边栏渲染混合模式 AI 对话助手（平台免费额度 / 用户 BYOK）。"""
    _defaults = {
        "ai_messages": [],
        "ai_user_api_key": "",
        "ai_user_base_url": DEFAULT_BYOK_BASE,
        "ai_user_model": DEFAULT_BYOK_MODEL,
        "ai_use_platform_free": False,
        "ai_platform_calls": 0,
    }
    for _k, _v in _defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    st.sidebar.markdown("---")
    st.sidebar.title("💬 AI 智能助手")

    with st.sidebar.expander("⚙️ API 设置", expanded=False):
        st.session_state.ai_use_platform_free = st.checkbox(
            "使用平台免费额度",
            value=st.session_state.ai_use_platform_free,
            key="ai_cb_free",
            on_change=st.rerun,
        )
        if not st.session_state.ai_use_platform_free:
            st.session_state.ai_user_api_key = st.text_input(
                "API Key", type="password",
                value=st.session_state.ai_user_api_key, key="ai_in_key",
            )
            st.session_state.ai_user_base_url = st.text_input(
                "Base URL", value=st.session_state.ai_user_base_url, key="ai_in_url",
            )
            st.session_state.ai_user_model = st.text_input(
                "Model 名称", value=st.session_state.ai_user_model, key="ai_in_model",
            )
        else:
            st.info("已启用平台免费额度（智谱 GLM-4.7-Flash，每会话限 15 次）。")

    for _msg in st.session_state.ai_messages:
        with st.sidebar.chat_message(_msg["role"]):
            st.markdown(_msg["content"])

    if st.sidebar.button("🔄 重置对话", key="ai_btn_reset"):
        st.session_state.ai_messages = []
        st.session_state.ai_platform_calls = 0
        st.rerun()

    _prompt = st.sidebar.chat_input("向 AI 助手提问…", key="ai_chat_in")
    if _prompt and _prompt.strip():
        _handle_ai_prompt(_prompt.strip())
        st.rerun()


st.title("📊 实证派")

# ==================== 侧边栏：AI 智能助手（顶部） ====================
render_ai_assistant()

# ==================== 侧边栏导航 ====================
st.sidebar.title("📑 分析目录")
page = st.sidebar.radio(
    "请选择分析阶段：",
    ["1. 数据清洗", "2. 描述性统计与模型诊断", "3. 回归分析", "4. 指标测算", "5. 耦合协调度模型", "6. 内生性检验", "7. DID + 事件研究法", "8. RDD", "9. 机制分析（中介/调节/门槛）", "10. 空间计量（SLM/SEM/SDM）", "11. 时间序列分析", "12. 结构方程模型 SEM", "13. 双重机器学习 / 因果森林", "14. 多层线性模型", "15. P2 综合评价进阶（模糊/可变权/AHP）"],
)

# ==================== 初始化会话状态 ====================
if "file_data" not in st.session_state:
    st.session_state.file_data = {}
if "merged_df" not in st.session_state:
    st.session_state.merged_df = None
if "_auto_loaded" not in st.session_state:
    st.session_state._auto_loaded = False
if "col_id" not in st.session_state:
    st.session_state.col_id = None
if "col_year" not in st.session_state:
    st.session_state.col_year = None
if "col_dv" not in st.session_state:
    st.session_state.col_dv = []
if "col_iv" not in st.session_state:
    st.session_state.col_iv = []
if "col_cv" not in st.session_state:
    st.session_state.col_cv = []
if "col_mv" not in st.session_state:
    st.session_state.col_mv = []
if "col_industry" not in st.session_state:
    st.session_state.col_industry = "无"
if "year_start" not in st.session_state:
    st.session_state.year_start = 2000
if "year_end" not in st.session_state:
    st.session_state.year_end = 2030
if "fill_method" not in st.session_state:
    st.session_state.fill_method = "线性插值 + 前后填充"
if "do_winsorize" not in st.session_state:
    st.session_state.do_winsorize = False
if "auto_log" not in st.session_state:
    st.session_state.auto_log = True
if "selected_industries" not in st.session_state:
    st.session_state.selected_industries = []

# ================================================================
#                    第一阶段：数据清洗
# ================================================================
if page == "1. 数据清洗":
    st.header("第一章：数据清洗与预处理")
    st.markdown("上传任意格式的Excel/CSV文件，通过点选配置，一键完成数据清洗")

    # 自动加载本地缓存
    cache_file = "saved_main_data.parquet"
    if (
        os.path.exists(cache_file)
        and not st.session_state.file_data
        and not st.session_state.get("_auto_loaded", False)
    ):
        try:
            loaded_df = pd.read_parquet(cache_file)
            st.session_state.file_data = {"auto_loaded_data.parquet": loaded_df}
            st.session_state.merged_df = loaded_df.copy()
            # 恢复上次保存的列映射，使下游回归模块可直接使用
            try:
                with open("saved_main_data_meta.json") as _mf:
                    _meta = json.load(_mf)
                for _k, _v in _meta.items():
                    st.session_state[_k] = _v
            except Exception:
                pass
            st.session_state._auto_loaded = True
            st.info("💾 已自动加载上次保存的清洗数据")
        except Exception as e:
            st.error(f"读取本地缓存失败: {e}")

    # Step 1: 上传数据文件
    st.subheader("Step 1: 上传数据文件 (支持多文件、即时预览/修改)")
    uploaded_files = st.file_uploader(
        "上传多个 Excel/CSV 文件",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )

    # 文件删除后清理缓存（自动加载的数据不会被误清理）
    if (
        not uploaded_files
        and st.session_state.file_data
        and not st.session_state.get("_auto_loaded", False)
    ):
        st.session_state.file_data = {}
        st.session_state.merged_df = None
        st.rerun()

    if uploaded_files:
        new_file_processed = False
        for uf in uploaded_files:
            file_name = uf.name
            if file_name not in st.session_state.file_data:
                try:
                    if file_name.endswith(".csv"):
                        df = pd.read_csv(uf)
                    else:
                        df = pd.read_excel(uf)
                    st.session_state.file_data[file_name] = df
                    new_file_processed = True
                    st.success(
                        f"✅ 成功加载: {file_name} | 数据大小: {df.shape[0]}行 × {df.shape[1]}列"
                    )
                except Exception as e:
                    st.error(f"❌ 读取文件 {file_name} 失败: {str(e)}")
        if new_file_processed:
            st.session_state._auto_loaded = False
            st.rerun()

    # 文件预览 + 列重命名
    if st.session_state.file_data:
        st.divider()
        st.subheader(f"📊 共加载 {len(st.session_state.file_data)} 个数据集")
        for idx, file_name in enumerate(st.session_state.file_data.keys()):
            df = st.session_state.file_data[file_name]
            with st.expander(f"📂 数据集 {idx+1}: {file_name}", expanded=(idx == 0)):
                st.markdown("#### 🔍 数据预览")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("#### 🛠️ 列名修改")
                cols = df.columns.tolist()
                new_cols = []
                num_cols_per_row = 3
                for row_idx in range(0, len(cols), num_cols_per_row):
                    cols_in_row = cols[row_idx : row_idx + num_cols_per_row]
                    row_columns = st.columns(num_cols_per_row)
                    for col_idx, col_name in enumerate(cols_in_row):
                        new_col_name = row_columns[col_idx].text_input(
                            label=f"第 {row_idx+col_idx+1} 列",
                            value=col_name,
                            key=f"rename_{file_name}_{col_name}",
                        )
                        new_cols.append(new_col_name)
                if new_cols != cols:
                    df.columns = new_cols
                    st.session_state.file_data[file_name] = df
                    st.success(f"✅ 文件 {file_name} 的列名已更新！")
                st.markdown("---")
    else:
        st.info(
            "👆 请在上方上传 Excel/CSV 文件，系统将自动加载并展示预览、支持列名修改～"
        )

    # Step 2: 列映射与变量选择
    if st.session_state.file_data:
        st.subheader("Step 2: 列映射与变量选择")
        all_cols = []
        for name, df in st.session_state.file_data.items():
            all_cols.extend(df.columns.tolist())
        all_cols = sorted(list(set(all_cols)))
        st.session_state.all_cols = all_cols

        # 关键列映射
        st.session_state.col_id = st.selectbox(
            "【股票代码/公司ID】列",
            options=all_cols,
            index=0,
            key="col_id_select",
        )

        st.session_state.col_year = st.selectbox(
            "【年份】列",
            options=all_cols,
            index=0,
        )
        st.session_state.col_dv = st.multiselect(
            "【因变量】（如TobinQ、ROA）", options=all_cols
        )
        st.session_state.col_iv = st.multiselect(
            "【自变量】（如AI词频）", options=all_cols
        )
        st.session_state.col_cv = st.multiselect(
            "【控制变量】（如Size、Lev）", options=all_cols
        )
        st.session_state.col_mv = st.multiselect(
            "【中介变量】（如研发投入）", options=all_cols
        )
        st.session_state.col_industry = st.selectbox(
            "【行业代码】列（可选）", options=["无"] + all_cols
        )

        # 年份范围
        min_year = 2000
        max_year = 2030
        if st.session_state.col_year != "无":
            try:
                all_years = pd.concat(
                    [
                        df[st.session_state.col_year]
                        for df in st.session_state.file_data.values()
                    ]
                )
                min_year = int(all_years.min())
                max_year = int(all_years.max())
            except Exception:
                pass
        st.session_state.year_start, st.session_state.year_end = st.slider(
            "选择年份范围",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

    # Step 3: 清洗参数设置
    st.subheader("Step 3: 清洗参数设置")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**已选行业列**: {st.session_state.get('col_industry', '无')}")
    with col2:
        st.session_state.fill_method = st.selectbox(
            "缺失值处理方法",
            ["线性插值 + 前后填充", "仅线性插值", "仅前后填充", "删除缺失值"],
        )
    with col3:
        st.session_state.do_winsorize = st.checkbox(
            "对连续变量进行1%缩尾处理", value=False
        )
        st.session_state.auto_log = st.checkbox(
            "自动对偏态变量取对数 ln(1+x)", value=True
        )

    # 行业筛选
    selected_industries = []
    if st.session_state.col_industry != "无":
        col_industry = st.session_state.col_industry
        all_industries = set()
        for name, df in st.session_state.file_data.items():
            if col_industry in df.columns:
                all_industries.update(df[col_industry].astype(str).unique())
        all_industries = sorted(list(all_industries))
        st.session_state.selected_industries = st.multiselect(
            "保留的行业代码",
            options=all_industries,
            default=all_industries[:3] if len(all_industries) >= 3 else all_industries,
        )

    # Step 4: 执行清洗
    st.subheader("Step 4: 执行清洗")
    if st.button("▶️ 开始清洗", type="primary"):
        col_id = st.session_state.col_id
        col_year = st.session_state.col_year
        col_dv = st.session_state.col_dv
        col_iv = st.session_state.col_iv
        col_cv = st.session_state.col_cv
        col_mv = st.session_state.col_mv
        col_industry = st.session_state.col_industry
        year_start = st.session_state.year_start
        year_end = st.session_state.year_end
        fill_method = st.session_state.fill_method
        do_winsorize = st.session_state.do_winsorize
        auto_log = st.session_state.auto_log
        selected_industries = st.session_state.selected_industries

        if not col_id or not col_year:
            st.error("请至少指定股票代码列和年份列！")
        elif not col_dv and not col_iv:
            st.error("请至少选择一个因变量或自变量！")
        else:
            with st.spinner("正在清洗数据，请稍候..."):
                try:
                    merged = None
                    for name, df in st.session_state.file_data.items():
                        df = df.copy()

                        needed_cols = [col_id, col_year]
                        if col_industry != "无" and col_industry in df.columns:
                            needed_cols.append(col_industry)
                        for v in col_dv + col_iv + col_cv + col_mv:
                            if v in df.columns and v not in needed_cols:
                                needed_cols.append(v)

                        exist_cols = [c for c in needed_cols if c in df.columns]
                        df_sub = df[exist_cols].copy()
                        df_sub[col_id] = df_sub[col_id].astype(str)
                        df_sub[col_year] = df_sub[col_year].astype(int)

                        if merged is None:
                            merged = df_sub
                        else:
                            merged = merged.merge(
                                df_sub,
                                on=[col_id, col_year],
                                how="outer",
                                suffixes=("", "_dup"),
                            )
                            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
                            merged.drop(columns=dup_cols, inplace=True)

                    if merged is None or merged.empty:
                        st.error("合并后数据为空，请检查文件内容和列映射。")
                        st.stop()

                    merged = merged[
                        (merged[col_year] >= year_start)
                        & (merged[col_year] <= year_end)
                    ]

                    if col_industry != "无" and selected_industries:
                        merged = merged[
                            merged[col_industry].astype(str).isin(selected_industries)
                        ]

                    # 缺失值处理
                    num_cols = merged.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    num_cols = [c for c in num_cols if c not in [col_id, col_year]]

                    if fill_method == "线性插值 + 前后填充":
                        merged[num_cols] = merged.groupby(col_id)[
                            num_cols
                        ].transform(
                            lambda g: g.interpolate(
                                method="linear", limit_direction="both"
                            )
                            .bfill()
                            .ffill()
                        )
                    elif fill_method == "仅线性插值":
                        merged[num_cols] = merged.groupby(col_id)[
                            num_cols
                        ].transform(
                            lambda g: g.interpolate(
                                method="linear", limit_direction="both"
                            )
                        )
                    elif fill_method == "仅前后填充":
                        merged[num_cols] = merged.groupby(col_id)[
                            num_cols
                        ].transform(lambda g: g.bfill().ffill())
                    elif fill_method == "删除缺失值":
                        merged = merged.dropna(subset=num_cols)

                    # 缩尾处理
                    if do_winsorize:

                        def winsorize_series(s):
                            lower = s.quantile(0.01)
                            upper = s.quantile(0.99)
                            return s.clip(lower, upper)

                        merged[num_cols] = merged[num_cols].apply(winsorize_series)

                    # 自动对数变换（修复：只对数值列操作）
                    if auto_log:
                        all_numeric = merged.select_dtypes(
                            include=[np.number]
                        ).columns.tolist()
                        exclude_log = [col_id, col_year] + [
                            col for col in all_numeric if "ln_" in col
                        ]
                        cols_to_check = [c for c in all_numeric if c not in exclude_log]
                        transformed = []
                        for col in cols_to_check:
                            if merged[col].nunique() > 10:
                                col_skew = skew(merged[col].dropna())
                                if abs(col_skew) > 2:
                                    merged[f"ln_{col}"] = np.log1p(merged[col])
                                    transformed.append(col)
                        if transformed:
                            st.info(
                                f"✅ 已自动对以下变量取对数 ln(1+x)：{', '.join(transformed)}"
                            )
                        else:
                            st.info("ℹ️ 未检测到需要取对数的变量（偏度条件不满足）。")

                    st.session_state.merged_df = merged

                    st.success(
                        f"✅ 清洗完成！共 {merged.shape[0]} 行，{merged.shape[1]} 列"
                    )
                    st.subheader("📋 数据预览（前100行）")
                    st.dataframe(merged.head(100), use_container_width=True)
                    st.subheader("📈 描述性统计")
                    display_cols = [
                        c for c in merged.columns if c not in [col_id, col_year]
                    ]
                    if col_industry != "无":
                        display_cols = [c for c in display_cols if c != col_industry]
                    if display_cols:
                        st.dataframe(
                            merged[display_cols].describe().round(4),
                            use_container_width=True,
                        )

                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        merged.to_excel(
                            writer, index=False, sheet_name="清洗后数据"
                        )
                    output.seek(0)
                    st.download_button(
                        label="⬇️ 下载清洗后的数据 (Excel)",
                        data=output,
                        file_name=f"cleaned_data_{year_start}_{year_end}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                except Exception as e:
                    st.error(f"❌ 清洗失败：{str(e)}")
                    st.code(traceback.format_exc())
    # 保存清洗结果为本地缓存（下次自动加载）
    if st.session_state.merged_df is not None:
        if st.button("💾 保存为本地缓存（下次自动加载）", key="save_cache"):
            try:
                st.session_state.merged_df.to_parquet(
                    "saved_main_data.parquet", index=False
                )
                # 一并保存列映射，供下次自动加载时恢复
                _meta = {
                    "col_id": st.session_state.col_id,
                    "col_year": st.session_state.col_year,
                    "col_dv": st.session_state.col_dv,
                    "col_iv": st.session_state.col_iv,
                    "col_cv": st.session_state.col_cv,
                    "col_mv": st.session_state.col_mv,
                    "col_industry": st.session_state.col_industry,
                }
                with open("saved_main_data_meta.json", "w") as _mf:
                    json.dump(_meta, _mf, ensure_ascii=False)
                st.success(
                    "✅ 已保存至 saved_main_data.parquet，下次打开将自动加载。"
                )
            except Exception as e:
                st.error(f"保存失败：{e}")

# ================================================================
#                    第二阶段：描述性统计与模型诊断
# ================================================================
elif page == "2. 描述性统计与模型诊断":  # 注意：侧边栏radio也要同步改为这个名称
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先前往「数据清洗」阶段完成数据清洗！")
    else:
        df = st.session_state.merged_df
        stat_cols = df.select_dtypes(include="number").columns.tolist()

        # ---------- 1. 描述性统计 ----------
        st.subheader("📊 描述性统计")
        st.caption("展示每个变量的均值、标准差、最值等，快速了解数据的基本分布。")
        if stat_cols:
            st.write(f"样本量: {df.shape[0]} 条观测值, {len(stat_cols)} 个数值变量")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("---")

            desc = df[stat_cols].describe().T.round(4)
            desc["median"] = df[stat_cols].median().round(4)
            preferred_order = [
                "mean",
                "std",
                "min",
                "25%",
                "75%",
                "max",
                "median",
            ]
            final_cols = [c for c in preferred_order if c in desc.columns]
            desc = desc[final_cols]
            st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

            output_desc = BytesIO()
            with pd.ExcelWriter(output_desc, engine="openpyxl") as writer:
                desc.to_excel(writer, sheet_name="描述性统计")
            output_desc.seek(0)
            st.download_button(
                label="⬇️ 下载描述性统计表 (Excel)",
                data=output_desc,
                file_name="descriptive_statistics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("💡 未找到适合统计的数值变量。")

        # ---------- 2. 相关性矩阵 ----------
        st.markdown("---")
        st.subheader("🔗 相关性矩阵")
        st.caption("计算两两变量之间的相关系数，初步判断变量关系及共线性迹象。")
        if len(stat_cols) >= 2:
            corr_method = st.selectbox(
                "相关系数类型",
                options=["Pearson", "Spearman", "Kendall"],
                index=0,
                key="corr_method",
            )
            corr_matrix = df[stat_cols].corr(method=corr_method.lower()).round(4)
            st.dataframe(corr_matrix, use_container_width=True)
            output_corr = BytesIO()
            with pd.ExcelWriter(output_corr, engine="openpyxl") as writer:
                corr_matrix.to_excel(writer, sheet_name="相关性矩阵")
            output_corr.seek(0)
            st.download_button(
                label="⬇️ 下载相关性矩阵 (Excel)",
                data=output_corr,
                file_name="correlation_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("💡 变量数量不足，无法计算相关性矩阵。")

        # ---------- 3. VIF 多重共线性检验 ----------
        st.markdown("---")
        st.subheader("📐 VIF 多重共线性检验")
        st.caption(
            "检查解释变量之间是否存在高度相关（VIF>10提示严重共线性），避免模型估计失真。"
        )
        # 需要用户选择解释变量（核心变量+控制变量）
        # 这里可以复用 session_state 中已有的变量选择，或者让用户在此处选择
        # 为简化，我们假设用户已经在清洗阶段选择了自变量和控制变量，但为了灵活性，此处允许用户选择
        all_num_cols = [
            c
            for c in stat_cols
            if c not in [st.session_state.col_id, st.session_state.col_year]
        ]
        vif_vars = st.multiselect(
            "选择要检验共线性的变量（通常为核心解释变量+控制变量）",
            options=all_num_cols,
            default=all_num_cols[: min(5, len(all_num_cols))],
            key="vif_vars",
        )
        if st.button("运行 VIF 检验", key="run_vif"):
            if len(vif_vars) < 2:
                st.warning("至少选择两个变量才能计算 VIF。")
            else:
                from statsmodels.stats.outliers_influence import (
                    variance_inflation_factor,
                )

                X = df[vif_vars].dropna()
                vif_data = pd.DataFrame()
                vif_data["变量"] = X.columns
                vif_data["VIF"] = [
                    variance_inflation_factor(X.values, i) for i in range(X.shape[1])
                ]
                vif_data["1/VIF"] = 1 / vif_data["VIF"]
                st.dataframe(
                    vif_data.style.format({"VIF": "{:.4f}", "1/VIF": "{:.4f}"}),
                    use_container_width=True,
                )
                # 提示
                high_vif = vif_data[vif_data["VIF"] > 10]
                if not high_vif.empty:
                    st.warning(
                        f"⚠️ 以下变量 VIF > 10，存在严重共线性：{', '.join(high_vif['变量'].tolist())}"
                    )
                else:
                    st.success("✅ 所有变量 VIF 均小于 10，共线性在可接受范围。")
        # ---------- 异方差与正态性诊断 ----------
        st.markdown("---")
        st.subheader("🩺 异方差与正态性诊断（回归诊断）")
        st.caption("对指定 OLS 模型残差检验异方差（Breusch-Pagan / White）与误差正态性（Jarque-Bera），是回归模型可信度的基础诊断。")
        _het_y = st.selectbox("因变量 Y", options=all_num_cols, key="het_y")
        _het_x = st.multiselect("解释变量 X", options=[c for c in all_num_cols if c != _het_y], default=[c for c in all_num_cols if c != _het_y][:min(3, len(all_num_cols) - 1)], key="het_x")
        if _het_y and _het_x:
            if st.button("运行诊断", key="het_btn"):
                try:
                    import statsmodels.api as sm
                    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
                    from statsmodels.stats.stattools import jarque_bera
                    _dd = df[[_het_y] + _het_x].dropna()
                    _X = sm.add_constant(_dd[_het_x].astype(float))
                    _m = sm.OLS(_dd[_het_y].astype(float), _X).fit()
                    _resid = _m.resid
                    _bp = het_breuschpagan(_resid, _X)
                    _wh = het_white(_resid, _X)
                    _jb = jarque_bera(_resid)
                    _diag = pd.DataFrame([
                        {"检验": "Breusch-Pagan 异方差", "统计量": f"{_bp[0]:.4f}", "p值": f"{_bp[1]:.4f}", "结论": "p<0.05 → 存在异方差"},
                        {"检验": "White 异方差", "统计量": f"{_wh[0]:.4f}", "p值": f"{_wh[1]:.4f}", "结论": "p<0.05 → 存在异方差"},
                        {"检验": "Jarque-Bera 正态性", "统计量": f"{_jb[0]:.4f}", "p值": f"{_jb[1]:.4f}", "结论": "p<0.05 → 残差非正态"},
                    ])
                    st.subheader("📊 诊断结果")
                    _show_table(_diag, "reg_diagnostics.xlsx", "诊断")
                    _msg = []
                    if _bp[1] < 0.05 or _wh[1] < 0.05:
                        _msg.append("⚠️ 检测到异方差，建议使用稳健标准误(HC1)、WLS 或聚类标准误。")
                    else:
                        _msg.append("✅ 未检测到显著异方差。")
                    if _jb[1] < 0.05:
                        _msg.append("⚠️ 残差非正态，大样本下 OLS 仍一致，但小样本推断需谨慎。")
                    else:
                        _msg.append("✅ 残差近似正态。")
                    for _m2 in _msg:
                        st.info(_m2)
                except Exception as _e:
                    st.error(f"诊断失败：{_e}"); st.code(traceback.format_exc())
        # ---------- 4. 单位根检验（平稳性检验） ----------
        st.markdown("---")
        st.subheader("📈 单位根检验（平稳性检验）")
        st.write("Debug: 单位根检验模块已进入")
        st.caption(
            "💡 短面板数据（T<20）可使用'单序列检验'逐变量判断；长面板或宏观数据（T≥20）建议使用'面板单位根检验'。"
        )

        import arch.unitroot as au
        from statsmodels.tsa.stattools import adfuller
        from scipy.stats import norm
        import numpy as np

        # 使用选项卡
        tab1, tab2 = st.tabs(["单序列单位根检验", "面板单位根检验 (LLC)"])

        # ========== 选项卡1：单序列单位根检验 ==========
        with tab1:
            st.markdown("**适用于**：任意数据。对选中的变量逐个进行 ADF/PP/KPSS 检验。")
            st.caption(
                "💡 可同时选择多个变量，系统将对每个变量**分别**进行单序列单位根检验，结果汇总在同一张表格中。"
            )

            unit_root_vars = st.multiselect(
                "选择需要检验平稳性的变量",
                options=all_num_cols,
                default=all_num_cols[: min(3, len(all_num_cols))],
                key="unit_root_vars_single",
            )
            method_options = ["ADF", "Phillips-Perron", "KPSS"]
            selected_method = st.selectbox(
                "选择检验方法", options=method_options, index=0, key="method_single"
            )
            if not unit_root_vars:
                st.warning("请至少选择一个变量。")
            else:
                results = []
                for var in unit_root_vars:
                    series = df[var].dropna()
                    if len(series) < 10:
                        results.append(
                            {
                                "变量": var,
                                "检验统计量": "样本不足",
                                "p值": "-",
                                "结论(5%)": "需≥10观测",
                            }
                        )
                        continue
                    try:
                        if selected_method == "ADF":
                            test = au.ADF(series, trend="c")
                        elif selected_method == "Phillips-Perron":
                            test = au.PhillipsPerron(series, trend="c")
                        elif selected_method == "KPSS":
                            test = au.KPSS(series, trend="c")
                        stat = test.stat
                        pval = test.pvalue
                        # ADF/PP: p<0.05 → 平稳；KPSS: p<0.05 → 非平稳
                        if selected_method == "KPSS":
                            conclusion = "非平稳" if pval < 0.05 else "平稳"
                        else:
                            conclusion = "平稳" if pval < 0.05 else "非平稳"
                        results.append(
                            {
                                "变量": var,
                                f"{selected_method}统计量": f"{stat:.4f}",
                                "p值": f"{pval:.4f}",
                                "结论(5%)": conclusion,
                            }
                        )
                    except Exception as e:
                        st.error(f"❌ 计算变量 '{var}' 时发生错误，详细信息如下：")
                        st.code(traceback.format_exc())
                        results.append(
                            {
                                "变量": var,
                                f"{selected_method}统计量": "错误",
                                "p值": "错误",
                                "结论(5%)": str(e),
                            }
                        )

                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
                stable = [r["变量"] for r in results if r["结论(5%)"] == "平稳"]
                unstable = [r["变量"] for r in results if r["结论(5%)"] == "非平稳"]
                if stable:
                    st.success(f"✅ 平稳变量：{', '.join(stable)}")
                if unstable:
                    st.warning(
                        f"⚠️ 非平稳变量：{', '.join(unstable)}，建议进行差分处理（Δy_t = y_t - y_{{t-1}}）"
                    )
                st.info("💡 建议同时运行 ADF 和 KPSS 以交叉验证结论。")

        # ========== 选项卡2：面板单位根检验（LLC） ==========
        with tab2:
            st.markdown(
                "**适用于**：面板数据（同时有个体ID和时间维度）。LLC 检验是目前最主流的面板单位根检验方法。"
            )
            st.markdown(
                "**原理**：先对每个个体分别计算 ADF 统计量，再对这些 t 统计量求平均并标准化，得到服从标准正态分布的 LLC 统计量。"
            )

            # 让用户指定个体ID和时间列
            id_col_options = [
                c
                for c in df.columns
                if df[c].dtype == object or df[c].nunique() < df.shape[0] * 0.5
            ]
            time_col_options = [
                c for c in df.columns if df[c].dtype in [np.int64, np.float64]
            ]

            if not id_col_options or not time_col_options:
                st.warning(
                    "⚠️ 未检测到合适的个体ID列或时间列，无法进行面板单位根检验。请确保在数据清洗阶段正确指定了股票代码和年份列。"
                )
            else:
                panel_id_col = st.selectbox(
                    "选择个体ID列（如股票代码）",
                    options=id_col_options,
                    index=0,
                    key="panel_id",
                )
                panel_time_col = st.selectbox(
                    "选择时间列（如年份）",
                    options=df.columns.tolist(),
                    index=0,
                    key="panel_time",
                )

                panel_vars = st.multiselect(
                    "选择需要检验的变量",
                    options=all_num_cols,
                    default=all_num_cols[: min(3, len(all_num_cols))],
                    key="panel_vars",
                )

                if st.button("运行 LLC 面板单位根检验", key="run_llc"):
                    if not panel_vars:
                        st.warning("请至少选择一个变量。")
                    else:
                        results = []
                        for var in panel_vars:
                            try:
                                # 提取该变量的面板数据
                                data = df[[panel_id_col, panel_time_col, var]].dropna()
                                # 按个体分组，对每个个体的时间序列做 ADF 检验
                                adf_stats = []
                                for _, group in data.groupby(panel_id_col):
                                    group_sorted = group.sort_values(panel_time_col)
                                    ts = group_sorted[var].values
                                    if len(ts) >= 10:  # 每个个体的时间序列至少10个观测
                                        try:
                                            adf_res = adfuller(ts)
                                            adf_stats.append(adf_res[0])  # ADF t 统计量
                                        except Exception:
                                            pass

                                if len(adf_stats) < 2:
                                    results.append(
                                        {
                                            "变量": var,
                                            "LLC统计量": "样本不足",
                                            "p值": "-",
                                            "结论(5%)": "需≥2个个体且每序列≥10观测",
                                        }
                                    )
                                    continue

                                # LLC 统计量计算
                                mean_adf = np.mean(adf_stats)
                                std_adf = np.std(adf_stats, ddof=1)
                                llc_stat = mean_adf / (
                                    std_adf / np.sqrt(len(adf_stats))
                                )

                                # LLC 统计量渐近服从标准正态分布
                                p_value = 2 * (1 - norm.cdf(abs(llc_stat)))

                                conclusion = "平稳" if p_value < 0.05 else "非平稳"
                                results.append(
                                    {
                                        "变量": var,
                                        "LLC统计量": f"{llc_stat:.4f}",
                                        "p值": f"{p_value:.4f}",
                                        "结论(5%)": conclusion,
                                    }
                                )
                            except Exception as e:
                                results.append(
                                    {
                                        "变量": var,
                                        "LLC统计量": "错误",
                                        "p值": "错误",
                                        "结论(5%)": str(e),
                                    }
                                )

                        res_df = pd.DataFrame(results)
                        st.dataframe(res_df, use_container_width=True)
                        stable = [r["变量"] for r in results if r["结论(5%)"] == "平稳"]
                        unstable = [
                            r["变量"] for r in results if r["结论(5%)"] == "非平稳"
                        ]
                        if stable:
                            st.success(f"✅ 平稳变量：{', '.join(stable)}")
                        if unstable:
                            st.warning(
                                f"⚠️ 非平稳变量：{', '.join(unstable)}，建议进行差分处理后重新估计模型。"
                            )
                        st.info(
                            "💡 LLC 检验原假设为'存在单位根'（非平稳），p<0.05 拒绝原假设，认为序列平稳。"
                        )

        # ---------- 5/6/7. F检验、LM检验、Hausman检验（合并变量选择）----------
        st.markdown("---")
        st.subheader("🔬 模型设定检验（F检验 / LM检验 / Hausman检验）")
        st.caption(
            "请先选择被解释变量(Y)和解释变量(X)，然后点击下方按钮运行全部三个检验。"
        )

        # 让用户选择Y和X
        all_num_cols_for_test = [
            c
            for c in stat_cols
            if c not in [st.session_state.col_id, st.session_state.col_year]
        ]
        y_for_test = st.selectbox(
            "选择被解释变量 (Y)",
            options=all_num_cols_for_test,
            index=len(all_num_cols_for_test) - 1 if all_num_cols_for_test else 0,
            key="y_for_test",
        )
        x_for_test = st.multiselect(
            "选择解释变量 (X，至少选一个)",
            options=all_num_cols_for_test,
            default=[c for c in all_num_cols_for_test if c != y_for_test][:3],
            key="x_for_test",
        )

        if st.button("运行 F / LM / Hausman 检验", key="run_all_tests"):
            if not x_for_test or not y_for_test:
                st.warning("请至少选择一个Y和一个X。")
            else:
                # 准备面板数据
                df_panel = df.copy()
                _id_col = st.session_state.col_id
                _year_col = st.session_state.col_year
                if _id_col not in df_panel.columns:
                    df_panel[_id_col] = (
                        df_panel.index.get_level_values(0)
                        if isinstance(df_panel.index, pd.MultiIndex)
                        else range(len(df_panel))
                    )
                if _year_col not in df_panel.columns:
                    df_panel[_year_col] = (
                        df_panel.index.get_level_values(1)
                        if isinstance(df_panel.index, pd.MultiIndex)
                        else range(len(df_panel))
                    )
                df_panel = df_panel.set_index([_id_col, _year_col])
                exog = sm.add_constant(df_panel[x_for_test])

                # ---------------- F检验 --------------
                st.markdown("##### F检验（混合OLS vs 固定效应）")
                mod_pooled = PanelOLS(
                    df_panel[y_for_test], exog, entity_effects=False, time_effects=False
                )
                res_pooled = mod_pooled.fit()
                mod_fe = PanelOLS(
                    df_panel[y_for_test], exog, entity_effects=True, time_effects=False
                )
                res_fe = mod_fe.fit()
                ssr_pooled = res_pooled.resids.dot(res_pooled.resids)
                ssr_fe = res_fe.resids.dot(res_fe.resids)
                n = df_panel.index.get_level_values(0).nunique()
                T = df_panel.index.get_level_values(1).nunique()
                k = len(x_for_test)
                F_stat = ((ssr_pooled - ssr_fe) / (n - 1)) / (ssr_fe / (n * T - n - k))
                p_val_F = 1 - stats.f.cdf(F_stat, n - 1, n * T - n - k)
                st.write(f"F统计量 = {F_stat:.4f}, p值 = {p_val_F:.4f}")
                if p_val_F < 0.05:
                    st.success("✅ p < 0.05，拒绝混合OLS，建议使用固定效应模型。")
                else:
                    st.info("ℹ️ p ≥ 0.05，无法拒绝混合OLS，可使用混合模型。")

                # ------------- LM检验（手动计算，基于残差，兼容所有版本）-----------
                st.markdown("##### LM检验（Breusch-Pagan，混合OLS vs 随机效应）")
                try:
                    # 1. 混合OLS模型（无个体效应）
                    mod_pooled = PanelOLS(
                        df_panel[y_for_test],
                        exog,
                        entity_effects=False,
                        time_effects=False,
                    )
                    res_pooled = mod_pooled.fit()

                    # 2. 随机效应模型
                    mod_re = RandomEffects(df_panel[y_for_test], exog)
                    res_re = mod_re.fit()

                    # 3. 计算LM统计量（Breusch-Pagan形式）
                    # 公式：LM = (N*T/2) * [ (sum_i (sum_t e_it)^2 / sum_i sum_t e_it^2 ) - 1 ]^2
                    # 其中 e_it 是混合OLS的残差
                    resid_pooled = res_pooled.resids
                    # 获取个体索引
                    entity_ids = resid_pooled.index.get_level_values(0)
                    # 按个体求和残差
                    resid_sum_by_entity = resid_pooled.groupby(entity_ids).sum()
                    # 计算分子：sum_i (sum_t e_it)^2
                    numerator = (resid_sum_by_entity**2).sum()
                    # 分母：sum_i sum_t e_it^2
                    denominator = (resid_pooled**2).sum()
                    # LM统计量
                    n = df_panel.index.get_level_values(0).nunique()
                    T = df_panel.index.get_level_values(1).nunique()
                    LM_stat = (n * T / 2) * ((numerator / denominator) - 1) ** 2
                    # P值（卡方分布，自由度1）
                    p_val_LM = 1 - stats.chi2.cdf(LM_stat, df=1)

                    st.write(f"LM统计量 = {LM_stat:.4f}, p值 = {p_val_LM:.4f}")
                    if p_val_LM < 0.05:
                        st.success("✅ p < 0.05，拒绝混合OLS，建议使用随机效应模型。")
                    else:
                        st.info("ℹ️ p ≥ 0.05，无法拒绝混合OLS，可使用混合模型。")

                except Exception as e:
                    st.warning(f"LM检验暂不可用：{e}")
                # --------------- Hausman检验 -------------
                st.markdown("##### Hausman检验（固定效应 vs 随机效应）")
                try:
                    # 重新运行固定效应模型（确保独立性）
                    mod_fe_h = PanelOLS(
                        df_panel[y_for_test],
                        exog,
                        entity_effects=True,
                        time_effects=False,
                    )
                    res_fe_h = mod_fe_h.fit()
                    # 重新运行随机效应模型
                    mod_re_h = RandomEffects(df_panel[y_for_test], exog)
                    res_re_h = mod_re_h.fit()
                    b_fe = res_fe_h.params
                    b_re = res_re_h.params
                    v_fe = res_fe_h.cov
                    v_re = res_re_h.cov
                    common_params = [p for p in b_fe.index if p in b_re.index]
                    diff = b_fe[common_params] - b_re[common_params]
                    v_diff = (
                        v_fe.loc[common_params, common_params]
                        - v_re.loc[common_params, common_params]
                    )
                    inv_v_diff = np.linalg.inv(v_diff)
                    hausman_stat = diff.T @ inv_v_diff @ diff
                    p_val_H = 1 - stats.chi2.cdf(hausman_stat, len(common_params))
                    st.write(
                        f"Hausman χ²统计量 = {hausman_stat:.4f}, p值 = {p_val_H:.4f}"
                    )
                    if p_val_H < 0.05:
                        st.success("✅ p < 0.05，拒绝随机效应，建议使用固定效应模型。")
                    else:
                        st.info("ℹ️ p ≥ 0.05，无法拒绝随机效应，建议使用随机效应模型。")
                except np.linalg.LinAlgError:
                    st.warning("方差矩阵奇异，无法计算Hausman检验。")
                except Exception as e:
                    st.warning(f"Hausman检验出错：{e}")

# ================================================================
#                    第三阶段：回归分析（核心部分）
# ================================================================
elif page == "3. 回归分析":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from io import BytesIO
    from scipy.stats import skew
    from linearmodels.panel import PanelOLS, RandomEffects
    import warnings
    import traceback
    import statsmodels.api as sm
    from scipy import stats

    st.header("📈 回归分析与结果导出")

    # --- 数据状态强制同步与检查 ---
    data_panel = None

    # 1. 优先检查 data_panel
    if "data_panel" in st.session_state and st.session_state["data_panel"] is not None:
        data_panel = st.session_state["data_panel"]

    # 2. 如果 data_panel 不存在，尝试从 merged_df 抢救
    elif "merged_df" in st.session_state and st.session_state["merged_df"] is not None:
        st.warning("⚠️ 未检测到面板数据，但检测到清洗后的数据。正在自动为您转换...")
        st.session_state["data_panel"] = st.session_state["merged_df"]
        data_panel = st.session_state["data_panel"]
        st.rerun()  # 触发页面刷新

    # 3. 如果两者都没有，报错并停止
    else:
        st.warning("请先在「1. 数据清洗」页面完成数据清洗。")
        st.stop()

    # --- 调试信息 ---
    st.sidebar.write(f"Debug: 当前数据维度: {data_panel.shape}")
    st.sidebar.write(data_panel.head(10))  # 显示前10行数据

    # ========== 变量选择与模型设定 ==========
    st.subheader("🔧 模型设定")

    # 1. 获取所有列名
    all_cols = data_panel.columns.tolist()

    # 2. 从 session_state 获取用户在清洗阶段选择的 ID 和 Year 列名
    id_col = st.session_state.get("col_id", None)
    year_col = st.session_state.get("col_year", None)

    # 如果没选，直接报错并停止
    if id_col is None or year_col is None:
        st.error("请先在「数据清洗」页面选择「公司代码列」和「年份列」。")
        st.stop()

    # 4. 排除 ID 和 Year 列，使它们不出现在 X/Y/控制变量的选项中
    exclude_cols = [id_col, year_col]
    _candidate_vars = [c for c in all_cols if c not in exclude_cols]
    # 仅保留数值型变量：回归目前不做类别编码，若把字符串/类别列
    # （如行业代码）选入 X/Y，statsmodels 会抛 dtype=object 错误。
    analysis_vars = [
        c for c in _candidate_vars
        if pd.api.types.is_numeric_dtype(data_panel[c])
    ]
    _non_numeric = [c for c in _candidate_vars if c not in analysis_vars]
    if _non_numeric:
        st.caption(
            f"ℹ️ 以下非数值列已自动从回归变量中排除（回归暂不支持类别/文本变量）："
            f" {', '.join(_non_numeric)}"
        )

    # 5. 核心解释变量 (X)
    default_x = analysis_vars[:1] if len(analysis_vars) >= 1 else []
    x_vars = st.multiselect(
        "选择核心解释变量 (X)", options=analysis_vars, default=default_x
    )

    # 6. 控制变量
    remaining_controls = [c for c in analysis_vars if c not in x_vars]
    default_ctrl = (
        remaining_controls[:2] if len(remaining_controls) >= 2 else remaining_controls
    )
    control_vars = st.multiselect(
        "选择控制变量", options=remaining_controls, default=default_ctrl
    )

    # 7. 固定效应选项（PanelOLS 以 [id, year] 为索引：第一层=个体、第二层=年份，
    #    故用开关直接控制是否加入对应固定效应，避免原 selectbox 选列却无效的死代码）
    use_entity = st.checkbox("个体固定效应 (entity_effects)", value=True)
    use_time = st.checkbox("年份固定效应 (time_effects)", value=True)

    # 8. 目标变量 (Y)
    if len(analysis_vars) == 0:
        st.error("没有可用的变量，请检查数据清洗是否正确保留了变量。")
        st.stop()
    default_y_idx = len(analysis_vars) - 1
    y_var = st.selectbox(
        "选择被解释变量 (Y)", options=analysis_vars, index=default_y_idx
    )

    # 9. 运行按钮
    run_regression = st.button("🚀 开始回归", type="primary")

    # ========== 执行回归（仅当点击按钮时）==========
    if run_regression:
        if not x_vars or not y_var:
            st.warning("请至少选择一个核心解释变量和被解释变量。")
        elif not all(pd.api.types.is_numeric_dtype(data_panel[v]) for v in (x_vars + control_vars + [y_var]) if v in data_panel.columns):
            st.error("所选变量包含非数值类型，无法参与回归。请从变量选择中仅保留数值型变量（已自动排除类别/文本列）。")
        else:
            try:
                with st.spinner("正在计算回归模型，请稍候..."):
                    # 准备数据
                    df_model = data_panel.copy()

                    # === 设置索引 ===
                    if not isinstance(df_model.index, pd.MultiIndex):
                        if id_col in df_model.columns and year_col in df_model.columns:
                            try:
                                df_model = df_model.set_index([id_col, year_col])
                                st.info(f"已自动将 {id_col} 和 {year_col} 设置为索引。")
                            except Exception as e:
                                st.error(f"设置索引失败: {e}")
                                st.stop()
                        else:
                            st.error(
                                f"缺少设置面板索引所需的列（{id_col} 或 {year_col} 不在数据中）。"
                            )
                            st.stop()

                    # ========== 生成回归结果表格 ==========
                    st.subheader("📊 回归结果对比表（学术格式）")

                    # -------------------- 1. 生成滞后变量 --------------------
                    df_lag = df_model.copy()
                    for x in x_vars:
                        df_lag[f"{x}_L1"] = df_lag.groupby(level=0)[x].shift(1)
                        df_lag[f"{x}_L2"] = df_lag.groupby(level=0)[x].shift(2)

                    # 删除因滞后产生的缺失值，保证样本一致
                    df_clean = df_lag.dropna()

                    # 个体聚类维度（使 OLS 列与 FE 列一致按个体聚类，输出聚类稳健标准误）
                    try:
                        _entity_groups = df_clean.index.get_level_values(0).to_numpy()
                    except Exception:
                        _entity_groups = None
                    # linearmodels 7.0：聚类需传入与数据同索引的 DataFrame（按个体/公司聚类）
                    try:
                        _entity_clusters = pd.DataFrame(
                            {"entity": df_clean.index.get_level_values(0).to_numpy()},
                            index=df_clean.index,
                        )
                    except Exception:
                        _entity_clusters = None

                    # -------------------- 2. 定义控制变量集合 --------------------
                    # 动态划分控制变量：第(2)列用前一半，第(3)列用后一半
                    n_ctrl = len(control_vars)
                    if n_ctrl <= 1:
                        # 控制变量不足2个时，第(2)列用全部，第(3)列为空（与第(1)列相同）
                        ctrl_subset = control_vars
                        ctrl_remaining = []
                    else:
                        split_point = n_ctrl // 2
                        ctrl_subset = control_vars[:split_point]  # 前一半
                        ctrl_remaining = control_vars[split_point:]  # 后一半

                    if id_col in df_clean.columns and year_col in df_clean.columns:
                        df_clean = df_clean.set_index([id_col, year_col])
                    model_results = {}
                    # 第(1)列：仅当期X (OLS)
                    X1 = sm.add_constant(df_clean[x_vars])
                    model_results["(1) 仅X"] = sm.OLS(df_clean[y_var], X1).fit(
                        cov_type="cluster", cov_kwds={"groups": _entity_groups}
                    )

                    # 第(2)列：+前两个控制变量 (OLS)
                    X2 = sm.add_constant(df_clean[x_vars + ctrl_subset])
                    model_results["(2) +前两个控制"] = sm.OLS(df_clean[y_var], X2).fit(
                        cov_type="cluster", cov_kwds={"groups": _entity_groups}
                    )

                    # 第(3)列：+剩余控制变量 (OLS)
                    X3 = sm.add_constant(df_clean[x_vars + ctrl_remaining])
                    model_results["(3) +剩余控制"] = sm.OLS(df_clean[y_var], X3).fit(
                        cov_type="cluster", cov_kwds={"groups": _entity_groups}
                    )

                    # 第(4)列：+所有控制变量 + 固定效应 (PanelOLS)  ★ 真正加入固定效应
                    # 双向固定效应会吸收常数项，故仅当非「个体+年份」同时开启时才加截距
                    def _fe_exog(_cols):
                        _x = df_clean[_cols]
                        if not (use_entity and use_time):
                            _x = sm.add_constant(_x)
                        return _x

                    X4 = _fe_exog(x_vars + control_vars)
                    model_results["(4) 全控制"] = PanelOLS(
                        df_clean[y_var],
                        X4,
                        entity_effects=use_entity,  # 个体固定效应（用户可控）
                        time_effects=use_time,  # 年份固定效应（用户可控）
                    ).fit(
                        cov_type="clustered", clusters=_entity_clusters
                    )  # 聚类稳健标准误（按个体聚类，clusters 为同索引 DataFrame）

                    # 第(5)列：L1.X + 所有控制变量 + 固定效应 (PanelOLS)
                    l1_vars = [f"{x}_L1" for x in x_vars]
                    X5 = _fe_exog(l1_vars + control_vars)
                    model_results["(5) L1.X"] = PanelOLS(
                        df_clean[y_var], X5, entity_effects=use_entity, time_effects=use_time
                    ).fit(cov_type="clustered", clusters=_entity_clusters)

                    # 第(6)列：L2.X + 所有控制变量 + 固定效应 (PanelOLS)
                    l2_vars = [f"{x}_L2" for x in x_vars]
                    X6 = _fe_exog(l2_vars + control_vars)
                    model_results["(6) L2.X"] = PanelOLS(
                        df_clean[y_var], X6, entity_effects=use_entity, time_effects=use_time
                    ).fit(cov_type="clustered", clusters=_entity_clusters)

                    # 第(7)列：全控制 + 无固定效应 (OLS) —— 新增
                    X7 = sm.add_constant(df_clean[x_vars + control_vars])
                    model_results["(7) 全控制OLS"] = sm.OLS(df_clean[y_var], X7).fit(
                        cov_type="cluster", cov_kwds={"groups": _entity_groups}
                    )

                    # -------------------- 4. 辅助函数：安全获取标准误 --------------------
                    def get_se(model, var_name):
                        """兼容 OLS (bse) 和 PanelOLS (std_errors)"""
                        if hasattr(model, "std_errors"):
                            return model.std_errors.get(var_name, None)
                        elif hasattr(model, "bse"):
                            return model.bse.get(var_name, None)
                        return None

                    def format_coef_se(param, se, pvalue):
                        if param is None or se is None:
                            return ""
                        stars = ""
                        if pvalue < 0.01:
                            stars = "***"
                        elif pvalue < 0.05:
                            stars = "**"
                        elif pvalue < 0.1:
                            stars = "*"
                        return f"{param:.4f}{stars}({se:.4f})"

                    # -------------------- 5. 构建横向表格（行为变量，列为模型） --------------------
                    # 列顺序：1-7
                    model_names = [
                        "(1) 仅X",
                        "(2) +前两个控制",
                        "(3) +剩余控制",
                        "(4) 全控制",
                        "(5) L1.X",
                        "(6) L2.X",
                        "(7) 全控制OLS",
                    ]

                    rows = {}

                    # 5.1 核心解释变量行：每个X变量生成三行（当期X、L1.X、L2.X）
                    for x in x_vars:
                        # 当期X行
                        rows[x] = {"变量": x}
                        for mn in model_names:
                            m = model_results.get(mn)
                            if m is None:
                                rows[x][mn] = ""
                                continue
                            # 当期X只在第(1)-(4)列和第(7)列中出现
                            if mn in [
                                "(1) 仅X",
                                "(2) +前两个控制",
                                "(3) +剩余控制",
                                "(4) 全控制",
                                "(7) 全控制OLS",
                            ]:
                                var_name = x
                            else:
                                rows[x][mn] = ""
                                continue
                            if var_name in m.params:
                                param = m.params[var_name]
                                se = get_se(m, var_name)
                                pvalue = m.pvalues[var_name]
                                rows[x][mn] = format_coef_se(param, se, pvalue)
                            else:
                                rows[x][mn] = ""

                        # L1.X行
                        l1_key = f"{x}_L1"
                        rows[l1_key] = {"变量": l1_key}
                        for mn in model_names:
                            m = model_results.get(mn)
                            if m is None:
                                rows[l1_key][mn] = ""
                                continue
                            # L1.X只在第(5)列中出现
                            if mn == "(5) L1.X":
                                var_name = l1_key
                            else:
                                rows[l1_key][mn] = ""
                                continue
                            if var_name in m.params:
                                param = m.params[var_name]
                                se = get_se(m, var_name)
                                pvalue = m.pvalues[var_name]
                                rows[l1_key][mn] = format_coef_se(param, se, pvalue)
                            else:
                                rows[l1_key][mn] = ""

                        # L2.X行
                        l2_key = f"{x}_L2"
                        rows[l2_key] = {"变量": l2_key}
                        for mn in model_names:
                            m = model_results.get(mn)
                            if m is None:
                                rows[l2_key][mn] = ""
                                continue
                            # L2.X只在第(6)列中出现
                            if mn == "(6) L2.X":
                                var_name = l2_key
                            else:
                                rows[l2_key][mn] = ""
                                continue
                            if var_name in m.params:
                                param = m.params[var_name]
                                se = get_se(m, var_name)
                                pvalue = m.pvalues[var_name]
                                rows[l2_key][mn] = format_coef_se(param, se, pvalue)
                            else:
                                rows[l2_key][mn] = ""

                    # 5.2 控制变量行
                    for cv in control_vars:
                        rows[cv] = {"变量": cv}
                        for mn in model_names:
                            m = model_results.get(mn)
                            if m is None:
                                rows[cv][mn] = ""
                                continue
                            if cv in m.params:
                                param = m.params[cv]
                                se = get_se(m, cv)
                                pvalue = m.pvalues[cv]
                                rows[cv][mn] = format_coef_se(param, se, pvalue)
                            else:
                                rows[cv][mn] = ""

                    # 5.3 常数项
                    rows["常数项"] = {"变量": "常数项"}
                    for mn in model_names:
                        m = model_results.get(mn)
                        if m is None:
                            rows["常数项"][mn] = ""
                            continue
                        if "const" in m.params:
                            param = m.params["const"]
                            se = get_se(m, "const")
                            pvalue = m.pvalues["const"]
                            rows["常数项"][mn] = format_coef_se(param, se, pvalue)
                        else:
                            rows["常数项"][mn] = ""

                    # 5.4 固定效应标注（根据模型名称判断）
                    rows["个体固定效应"] = {"变量": "个体固定效应"}
                    rows["年份固定效应"] = {"变量": "年份固定效应"}
                    fe_label_e = "是" if use_entity else "否"
                    fe_label_t = "是" if use_time else "否"
                    for mn in model_names:
                        if mn in ["(4) 全控制", "(5) L1.X", "(6) L2.X"]:
                            rows["个体固定效应"][mn] = fe_label_e
                            rows["年份固定效应"][mn] = fe_label_t
                        else:
                            rows["个体固定效应"][mn] = "否"
                            rows["年份固定效应"][mn] = "否"

                    # 5.5 观测数和R²
                    rows["观测数"] = {"变量": "观测数"}
                    rows["R²"] = {"变量": "R²"}
                    for mn in model_names:
                        m = model_results.get(mn)
                        if m is None:
                            rows["观测数"][mn] = ""
                            rows["R²"][mn] = ""
                            continue
                        rows["观测数"][mn] = str(int(m.nobs))
                        if hasattr(m, "rsquared_within"):
                            rows["R²"][mn] = f"{m.rsquared_within:.3f}"
                        else:
                            rows["R²"][mn] = f"{m.rsquared:.3f}"

                    # -------------------- 6. 转换为 DataFrame 并显示 --------------------
                    # 构建变量顺序：先列出所有核心解释变量行（当期X、L1.X、L2.X），再控制变量，最后统计行
                    core_rows = []
                    for x in x_vars:
                        core_rows.append(x)  # 当期X
                        core_rows.append(f"{x}_L1")  # L1.X
                        core_rows.append(f"{x}_L2")  # L2.X
                    variable_order = (
                        core_rows
                        + control_vars
                        + ["常数项", "个体固定效应", "年份固定效应", "观测数", "R²"]
                    )
                    table_data = [rows[var] for var in variable_order if var in rows]
                    display_df = pd.DataFrame(table_data)
                    cols = ["变量"] + model_names
                    display_df = display_df[cols]

                    st.markdown(
                        display_df.to_html(index=False, escape=False),
                        unsafe_allow_html=True,
                    )
                    # ========== 导出 Excel ==========
                    st.subheader("📥 导出结果")
                    if not display_df.empty:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            # Sheet1：回归结果对比表
                            display_df.to_excel(
                                writer, index=False, sheet_name="回归结果"
                            )
                            # Sheet2：回归详情（系数/标准误/p值）
                            detail_data = []
                            for mk in model_names:
                                res = model_results.get(mk)
                                if res is None:
                                    continue
                                row = {"模型": mk}
                                # 判断该列是否为滞后模型
                                is_lag1 = mk == "(5) L1.X"
                                is_lag2 = mk == "(6) L2.X"
                                # 核心解释变量
                                for xv in x_vars:
                                    cv_key = (
                                        f"{xv}_L1"
                                        if is_lag1
                                        else f"{xv}_L2"
                                        if is_lag2
                                        else xv
                                    )
                                    if cv_key in res.params:
                                        row[f"{cv_key}_系数"] = res.params[cv_key]
                                        row[f"{cv_key}_标准误"] = get_se(res, cv_key)
                                        row[f"{cv_key}_p值"] = res.pvalues[cv_key]
                                # 控制变量
                                for cv in control_vars:
                                    if cv in res.params:
                                        row[f"{cv}_系数"] = res.params[cv]
                                        row[f"{cv}_标准误"] = get_se(res, cv)
                                        row[f"{cv}_p值"] = res.pvalues[cv]
                                # 常数项
                                if "const" in res.params:
                                    row["常数项_系数"] = res.params["const"]
                                    row["常数项_标准误"] = get_se(res, "const")
                                    row["常数项_p值"] = res.pvalues["const"]
                                row["观测数"] = int(res.nobs)
                                row["R²"] = res.rsquared
                                detail_data.append(row)
                            pd.DataFrame(detail_data).to_excel(
                                writer, sheet_name="回归详情", index=False
                            )
                        excel_buffer.seek(0)
                        st.download_button(
                            label="📥 下载回归结果 (Excel)",
                            data=excel_buffer.getvalue(),
                            file_name="regression_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        st.warning("⚠️ 当前筛选条件下没有数据可导出！")

            except Exception as e:
                # 固定效应完全吸收某变量时的友好提示（双向FE常见）
                if "absorb" in str(e).lower():
                    msg = str(e)
                    absorbed = ""
                    try:
                        absorbed = msg.split("absorbed")[-1].split(":")[-1].strip()
                    except Exception:
                        pass
                    st.error(
                        "⚠️ 固定效应完全吸收了以下变量（在个体/年份维度缺乏变异，无法识别）：\n"
                        f"**{absorbed}**\n"
                        "建议：在「固定效应」中仅保留一种，或从控制变量中移除这些变量后重试。"
                    )
                else:
                    st.error(f"发生错误: {str(e)}")
                st.code(traceback.format_exc())

    # ========== 其他回归模型：Logit/Probit 与分位数回归 ==========
    st.divider()
    st.subheader("🧩 其他回归模型（二值选择 & 分位数）")
    _t_logit, _t_qr, _t_lasso, _t_glm, _t_boot = st.tabs([
        "🔶 Logit / Probit 二值选择模型", "🔷 分位数回归 QuantReg",
        "🟢 Lasso / ElasticNet 变量筛选", "🟡 GLM 广义线性回归", "🟩 Bootstrap / 置换检验",
    ])

    with _t_logit:
        st.caption("二值选择模型用于因变量为 0/1 的场景。可选 Logit 或 Probit，支持稳健/聚类标准误，并报告系数、优势比（Logit）与平均边际效应（AME）。")
        _lg_y = st.selectbox("二值因变量 Y（将自动二值化）", options=analysis_vars, key="lg_y")
        _lg_x = st.multiselect(
            "解释变量 X",
            options=[c for c in analysis_vars if c != _lg_y],
            default=[c for c in analysis_vars if c != _lg_y][: min(3, len(analysis_vars) - 1)],
            key="lg_x",
        )
        _lg_model = st.radio("模型形式", ["Logit", "Probit"], horizontal=True, key="lg_model")
        _lg_se = st.radio("标准误", ["稳健(heteroskedasticity-robust)", "聚类(按公司代码)"], horizontal=True, key="lg_se")
        _lg_thr = st.radio("Y 二值化方式", ["按中位数切分(>中位=1)", "已是0/1则直接用"], index=0, key="lg_thr")
        if _lg_y and _lg_x:
            if st.button("🚀 估计二值选择模型", type="primary", key="lg_btn"):
                try:
                    with st.spinner("估计中..."):
                        _dd = data_panel[[_lg_y] + _lg_x].dropna()
                        _yraw = _dd[_lg_y].astype(float).values
                        if _lg_thr.startswith("已是"):
                            _y = (_yraw > 0).astype(int)
                        else:
                            _med = np.median(_yraw)
                            _y = (_yraw > _med).astype(int)
                        _X = sm.add_constant(_dd[_lg_x].astype(float))
                        if data_panel.index.nlevels > 1:
                            _ent = data_panel.index.get_level_values(0)
                        else:
                            _ent = data_panel[id_col]
                        _cl = pd.Series(_ent, index=data_panel.index).reindex(_dd.index).values
                        if _lg_se.startswith("聚类"):
                            _res = (sm.Logit(_y, _X) if _lg_model == "Logit" else sm.Probit(_y, _X)).fit(
                                cov_type="cluster", cov_kwds={"groups": _cl}, disp=0
                            )
                        else:
                            _res = (sm.Logit(_y, _X) if _lg_model == "Logit" else sm.Probit(_y, _X)).fit(
                                cov_type="HC1", disp=0
                            )
                        _params, _bse, _pvals = _res.params, _res.bse, _res.pvalues
                        _ci = _res.conf_int()
                        _or_label = "优势比OR" if _lg_model == "Logit" else "比率比"
                        _rows = []
                        for _n in _params.index:
                            _rows.append({
                                "变量": _n,
                                "系数": _fmt_coef(_params[_n], _bse[_n], _pvals[_n]),
                                "标准误": f"{_bse[_n]:.4f}",
                                "z值": f"{_params[_n] / _bse[_n]:.4f}",
                                "P>|z|": f"{_pvals[_n]:.4f}",
                                "95%下限": f"{_ci.loc[_n, 0]:.4f}",
                                "95%上限": f"{_ci.loc[_n, 1]:.4f}",
                                _or_label: f"{np.exp(_params[_n]):.4f}",
                            })
                        _coef_df = pd.DataFrame(_rows)
                        st.subheader("📊 回归系数表")
                        _show_table(_coef_df, "logit_probit_coef.xlsx", "系数")
                        # 平均边际效应 AME
                        _xb = _res.predict(_X)
                        if _lg_model == "Logit":
                            _f = _xb * (1 - _xb)
                        else:
                            _xb_lp = stats.norm.ppf(np.clip(_xb, 1e-6, 1 - 1e-6))
                            _f = stats.norm.pdf(_xb_lp)
                        _ame = (_f[:, None] * _params.values[None, :]).mean(axis=0)
                        _me_df = pd.DataFrame({"变量": list(_params.index), "平均边际效应AME": [f"{v:.4f}" for v in _ame]})
                        st.subheader("📈 平均边际效应（AME）")
                        _show_table(_me_df, "logit_probit_margin.xlsx", "边际效应")
                        st.caption(
                            f"伪R² = {_res.prsquared:.4f}；样本量 N = {int(_res.nobs)}；"
                            f"正例占比 = {_y.mean():.3f}；标准误 = {'聚类' if _lg_se.startswith('聚类') else '稳健HC1'}"
                        )
                except Exception as e:
                    st.error(f"二值选择模型失败：{e}")
                    st.code(traceback.format_exc())

    with _t_qr:
        st.caption("分位数回归（Quantile Regression）刻画解释变量对不同条件分位点上因变量的边际影响，比 OLS 均值回归更能反映分布异质性。")
        _qr_y = st.selectbox("因变量 Y", options=analysis_vars, key="qr_y")
        _qr_x = st.multiselect(
            "解释变量 X",
            options=[c for c in analysis_vars if c != _qr_y],
            default=[c for c in analysis_vars if c != _qr_y][: min(3, len(analysis_vars) - 1)],
            key="qr_x",
        )
        _qr_qs = st.multiselect("分位点", options=[0.1, 0.25, 0.5, 0.75, 0.9], default=[0.1, 0.25, 0.5, 0.75, 0.9], key="qr_qs")
        if _qr_y and _qr_x and _qr_qs:
            if st.button("🚀 估计分位数回归", type="primary", key="qr_btn"):
                try:
                    with st.spinner("估计中..."):
                        _dd = data_panel[[_qr_y] + _qr_x].dropna()
                        _Y = _dd[_qr_y].astype(float)
                        _X = sm.add_constant(_dd[_qr_x].astype(float))
                        _qrows = []
                        for _q in _qr_qs:
                            _rq = sm.QuantReg(_Y, _X).fit(q=_q)
                            for _n in _X.columns:
                                _qrows.append({
                                    "分位点": f"{_q:.2f}",
                                    "变量": _n,
                                    "系数": _fmt_coef(_rq.params[_n], _rq.bse[_n], _rq.pvalues[_n]),
                                    "标准误": f"{_rq.bse[_n]:.4f}",
                                    "P>|t|": f"{_rq.pvalues[_n]:.4f}",
                                })
                        _qr_df = pd.DataFrame(_qrows)
                        st.subheader("📊 分位数回归系数表")
                        _show_table(_qr_df, "quantile_regression.xlsx", "分位数回归")
                        _pivot = _qr_df.pivot(index="变量", columns="分位点", values="系数").reset_index()
                        st.subheader("📐 各变量系数跨分位点对比")
                        _show_table(_pivot, "quantile_regression_pivot.xlsx", "对比")
                except Exception as e:
                    st.error(f"分位数回归失败：{e}")
                    st.code(traceback.format_exc())

    # ========== Lasso / ElasticNet 变量筛选 ==========
    with _t_lasso:
        st.caption("Lasso(L1)/ElasticNet 通过正则化实现高维变量选择，自动将不重要变量的系数压缩为 0，适合控制变量众多或存在多重共线性的场景。")
        _num_cols = data_panel.select_dtypes(include=[np.number]).columns.tolist()
        _ls_y = st.selectbox("被解释变量 Y", options=_num_cols, key="ls_y")
        _ls_x = st.multiselect("候选解释变量 X（含全部控制变量）", options=[c for c in _num_cols if c != _ls_y], default=[c for c in _num_cols if c != _ls_y][:min(8, len(_num_cols) - 1)], key="ls_x")
        _ls_model = st.radio("模型", ["Lasso (L1)", "ElasticNet"], horizontal=True, key="ls_model")
        _ls_cv = st.checkbox("交叉验证选择 α", value=True, key="ls_cv")
        if _ls_y and _ls_x:
            if st.button("🚀 运行变量筛选", type="primary", key="ls_btn"):
                try:
                    from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
                    from sklearn.preprocessing import StandardScaler
                    _dd = data_panel[[_ls_y] + _ls_x].dropna()
                    _Y = _dd[_ls_y].astype(float).values
                    _Xs = StandardScaler().fit_transform(_dd[_ls_x].astype(float).values)
                    if _ls_cv:
                        _m = LassoCV(cv=5, random_state=0, max_iter=10000) if _ls_model.startswith("Lasso") else ElasticNetCV(cv=5, random_state=0, max_iter=10000)
                        _m.fit(_Xs, _Y)
                        _alpha = _m.alpha_
                    else:
                        _alpha = st.slider("正则化强度 α", 0.001, 1.0, 0.01, key="ls_a")
                        _m = Lasso(alpha=_alpha, max_iter=10000) if _ls_model.startswith("Lasso") else ElasticNet(alpha=_alpha, l1_ratio=0.5, max_iter=10000)
                        _m.fit(_Xs, _Y)
                    _coef = pd.DataFrame({"变量": _ls_x, "系数": np.round(_m.coef_, 4), "是否选中": [_c != 0 for _c in _m.coef_]})
                    _coef = _coef.sort_values("系数", key=lambda s: s.abs(), ascending=False)
                    st.subheader("📊 Lasso / ElasticNet 系数与变量筛选")
                    _show_table(_coef, "lasso_selection.xlsx", "筛选")
                    _sel = _coef[_coef["是否选中"]]["变量"].tolist()
                    st.success(f"✅ 选中 {len(_sel)} / {len(_ls_x)} 个变量：{', '.join(_sel) if _sel else '（无，尝试减小 α）'}")
                    if _ls_cv:
                        st.caption(f"交叉验证最优 α = {_alpha:.4f}")
                except Exception as _e:
                    st.error(f"变量筛选失败：{_e}"); st.code(traceback.format_exc())

    # ========== GLM 广义线性回归 ==========
    with _t_glm:
        st.caption("广义线性回归(GLM)放宽 OLS 正态同方差假设，支持二值(Binomial)、计数(Poisson)、非负(Gamma)等因变量。默认异方差稳健标准误(HC1)。")
        _num_cols = data_panel.select_dtypes(include=[np.number]).columns.tolist()
        _glm_y = st.selectbox("因变量 Y", options=_num_cols, key="glm_y")
        _glm_x = st.multiselect("解释变量 X", options=[c for c in _num_cols if c != _glm_y], default=[c for c in _num_cols if c != _glm_y][:min(5, len(_num_cols) - 1)], key="glm_x")
        _glm_family = st.selectbox("连接族 Family", ["Gaussian(连续)", "Binomial(二值0/1)", "Poisson(计数)", "Gamma(非负连续)"], key="glm_family")
        if _glm_y and _glm_x:
            if st.button("🚀 估计 GLM", type="primary", key="glm_btn"):
                try:
                    from statsmodels.genmod.families import Gaussian, Binomial, Poisson, Gamma
                    _dd = data_panel[[_glm_y] + _glm_x].dropna()
                    _Y = _dd[_glm_y].astype(float)
                    if _glm_family.startswith("Binomial"):
                        _Y = (_Y > _Y.median()).astype(int)
                    _X = sm.add_constant(_dd[_glm_x].astype(float))
                    _fam = {"Gaussian(连续)": Gaussian(), "Binomial(二值0/1)": Binomial(), "Poisson(计数)": Poisson(), "Gamma(非负连续)": Gamma()}[_glm_family]
                    _glm = sm.GLM(_Y, _X, family=_fam).fit(cov_type="HC1")
                    _rows = []
                    for _n in _X.columns:
                        _rows.append({"变量": _n, "系数": _fmt_coef(_glm.params[_n], _glm.bse[_n], _glm.pvalues[_n]), "标准误": f"{_glm.bse[_n]:.4f}", "P>|z|": f"{_glm.pvalues[_n]:.4f}"})
                    _gdf = pd.DataFrame(_rows)
                    st.subheader("📊 GLM 系数表")
                    _show_table(_gdf, "glm_results.xlsx", "GLM")
                    try:
                        st.caption(f"模型：{_glm_family}；伪 R² = {_glm.pseudo_rsquared(kind='cs'):.4f}")
                    except Exception:
                        st.caption(f"模型：{_glm_family}")
                except Exception as _e:
                    st.error(f"GLM 失败：{_e}"); st.code(traceback.format_exc())

    # ========== Bootstrap / 置换检验 ==========
    with _t_boot:
        st.caption("对 OLS 系数做 Bootstrap 百分位置信区间与置换检验 p 值，作为系数显著性的稳健性补充（不依赖正态/异方差假设）。")
        _num_cols = data_panel.select_dtypes(include=[np.number]).columns.tolist()
        _b_y = st.selectbox("因变量 Y", options=_num_cols, key="b_y")
        _b_x = st.multiselect("解释变量 X", options=[c for c in _num_cols if c != _b_y], default=[c for c in _num_cols if c != _b_y][:min(3, len(_num_cols) - 1)], key="b_x")
        _b_rep = st.number_input("Bootstrap 重复次数", min_value=100, max_value=2000, value=500, step=100, key="b_rep")
        if _b_y and _b_x:
            if st.button("🚀 运行 Bootstrap / 置换检验", type="primary", key="b_btn"):
                try:
                    _dd = data_panel[[_b_y] + _b_x].dropna().reset_index(drop=True)
                    _Y = _dd[_b_y].astype(float).values
                    _X = sm.add_constant(_dd[_b_x].astype(float)).values
                    def _ols(X, y):
                        return np.linalg.lstsq(X, y, rcond=None)[0]
                    _b_hat = _ols(_X, _Y)
                    _B = int(_b_rep); _boot = np.zeros((_B, _X.shape[1]))
                    _rng = np.random.default_rng(0)
                    for i in range(_B):
                        _idx = _rng.integers(0, len(_Y), len(_Y))
                        _boot[i] = _ols(_X[_idx], _Y[_idx])
                    _lo = np.percentile(_boot, 2.5, axis=0); _hi = np.percentile(_boot, 97.5, axis=0)
                    _xvar = _b_x[0]; _obs = _b_hat[1]
                    _perm = np.zeros(_B)
                    for i in range(_B):
                        _yp = _rng.permutation(_Y)
                        _perm[i] = _ols(_X, _yp)[1]
                    _p = np.mean(np.abs(_perm) >= abs(_obs))
                    _names = ["const"] + _b_x
                    _bdf = pd.DataFrame({"变量": _names, "点估计": np.round(_b_hat, 4), "Bootstrap 95%下限": np.round(_lo, 4), "Bootstrap 95%上限": np.round(_hi, 4)})
                    st.subheader("📊 Bootstrap 系数与 95% 置信区间")
                    _show_table(_bdf, "bootstrap_ci.xlsx", "Bootstrap")
                    _pdf = pd.DataFrame({"变量": _names, "原始系数": np.round(_b_hat, 4), "置换 p 值": ["%.4f" % _p if n == _xvar else "—" for n in _names]})
                    st.subheader("📊 置换检验 p 值（针对首个解释变量）")
                    _show_table(_pdf, "permutation_p.xlsx", "Permutation")
                    st.caption(f"变量 {_xvar} 的置换检验 p 值 = {_p:.4f}（<0.05 表示显著）")
                except Exception as _e:
                    st.error(f"Bootstrap 失败：{_e}"); st.code(traceback.format_exc())

# ================================================================
#                    第四阶段：指标测算（熵权法 + TOPSIS）
# ================================================================
elif page == "4. 指标测算":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from io import BytesIO
    import traceback

    st.header("📐 第四阶段：指标测算（熵权法客观赋权 + TOPSIS 综合得分）")

    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先前往「1. 数据清洗」完成数据清洗，再来进行指标测算。")
        st.stop()

    df = st.session_state.merged_df
    id_col = st.session_state.get("col_id", None)
    year_col = st.session_state.get("col_year", None)
    exclude = [c for c in [id_col, year_col] if c is not None]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    st.subheader("1. 选择参与测算的指标")
    st.caption("从清洗后的数值变量中挑选评价指标（指标越多，综合评价体系越完整，建议 ≥ 3 个）。")
    indicator_cols = st.multiselect(
        "评价指标列",
        options=num_cols,
        default=num_cols[: min(5, len(num_cols))],
        key="topsis_indicators",
    )

    if indicator_cols:
        st.subheader("2. 指标方向设置")
        st.caption("正向指标：数值越大越好（如 ROA、ROE、营业收入）；负向指标：数值越小越好（如 资产负债率、成本费用率）。")
        cols_ui = st.columns(min(len(indicator_cols), 3))
        directions = {}
        for i, c in enumerate(indicator_cols):
            with cols_ui[i % len(cols_ui)]:
                directions[c] = st.selectbox(
                    f"{c} 方向",
                    ["正向(越大越好)", "负向(越小越好)"],
                    index=0,
                    key=f"dir_{c}",
                )

        agg_opt = st.selectbox(
            "面板数据聚合方式",
            ["不聚合（每条观测作为一行）"]
            + ([f"按 {id_col} 聚合后取均值（每行=一个{id_col}）"] if id_col else []),
            index=0,
            key="topsis_agg",
        )

        weight_method = st.radio(
            "权重确定方法",
            ["熵权法（客观赋权）", "等权", "自定义权重"],
            index=0,
            key="topsis_weight",
        )
        custom_weights = None
        if weight_method == "自定义权重":
            custom_weights = {}
            cw_cols = st.columns(min(len(indicator_cols), 3))
            for i, c in enumerate(indicator_cols):
                with cw_cols[i % len(cw_cols)]:
                    custom_weights[c] = st.number_input(
                        f"{c} 权重", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}"
                    )

        if st.button("🚀 开始测算", type="primary"):
            if len(indicator_cols) < 2:
                st.warning("请至少选择 2 个指标。")
            else:
                try:
                    with st.spinner("正在计算熵权法与 TOPSIS..."):
                        data = df.copy()
                        if agg_opt.startswith("按"):
                            data = data.groupby(id_col)[indicator_cols].mean().reset_index()
                        X = data[indicator_cols].astype(float)

                        # ---- 极差标准化（消除量纲，按方向处理）----
                        Xn = pd.DataFrame(index=X.index)
                        for c in indicator_cols:
                            x = X[c]
                            if directions.get(c, "正向(越大越好)").startswith("正向"):
                                Xn[c] = (x - x.min()) / (x.max() - x.min() + 1e-12)
                            else:
                                Xn[c] = (x.max() - x) / (x.max() - x.min() + 1e-12)
                        Xn = Xn.clip(1e-12, None)  # 避免 0，熵权法需取对数

                        # ---- 熵权法客观赋权 ----
                        if weight_method == "等权":
                            weights = pd.Series(1.0 / len(indicator_cols), index=indicator_cols)
                        elif weight_method == "自定义权重":
                            s = pd.Series(custom_weights)[indicator_cols]
                            weights = (s / s.sum()) if s.sum() > 0 else pd.Series(1.0 / len(indicator_cols), index=indicator_cols)
                        else:
                            P = Xn.div(Xn.sum(axis=0), axis=1)        # 列归一化（比重）
                            k = 1.0 / np.log(len(Xn))                 # 熵系数
                            E = -k * (P * np.log(P + 1e-12)).sum(axis=0)  # 信息熵
                            D = 1 - E                                # 冗余度
                            weights = D / D.sum()

                        # ---- TOPSIS ----
                        Z = Xn * weights                             # 加权标准化矩阵
                        A_plus = Z.max(axis=0)                       # 正理想解
                        A_minus = Z.min(axis=0)                     # 负理想解
                        D_plus = np.sqrt(((Z - A_plus) ** 2).sum(axis=1))
                        D_minus = np.sqrt(((Z - A_minus) ** 2).sum(axis=1))
                        C = D_minus / (D_plus + D_minus + 1e-12)     # 贴近度

                        result = pd.DataFrame({
                            "样本标识": (data[id_col] if (agg_opt.startswith("按") and id_col) else range(len(data))),
                            "综合得分": C.round(4),
                        })
                        result = result.sort_values("综合得分", ascending=False).reset_index(drop=True)
                        result.insert(0, "排名", result.index + 1)

                        st.subheader("📊 熵权法权重")
                        wdf = weights.round(4).reset_index()
                        wdf.columns = ["指标", "熵权权重"]
                        st.dataframe(wdf, use_container_width=True)

                        st.subheader("🏆 TOPSIS 综合得分与排名（学术格式）")
                        st.dataframe(result, use_container_width=True)
                        top_n = min(20, len(result))
                        st.bar_chart(result.head(top_n).set_index("样本标识")["综合得分"])

                        # ---- Excel 下载 ----
                        out = BytesIO()
                        with pd.ExcelWriter(out, engine="openpyxl") as w:
                            wdf.to_excel(w, sheet_name="熵权法权重", index=False)
                            result.to_excel(w, sheet_name="TOPSIS得分", index=False)
                        out.seek(0)
                        st.download_button(
                            label="⬇️ 下载指标测算结果 (Excel)",
                            data=out,
                            file_name="entropy_topsis_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as e:
                    st.error(f"测算失败：{e}")
                    st.code(traceback.format_exc())

    # ==================== 其他综合评价方法（Tab 组） ====================
    st.markdown("---")
    st.subheader("🔬 其他综合评价方法（CRITIC / PCA / DEA）")
    _t_crit, _t_pca, _t_dea, _t_std, _t_comp, _t_fa = st.tabs([
        "🔵 CRITIC 权重法", "🟠 PCA 主成分分析", "🟣 DEA 数据包络分析",
        "⚪ 指标标准化/归一化", "🟤 综合评价法", "🔶 多因子分析 FA",
    ])
    with _t_crit:
        st.caption("CRITIC 法（Criteria Importance Through Intercriteria Correlation）利用指标标准差（对比强度）与指标间冲突性（相关系数）确定客观权重，无需主观判断。")
        _crit_ind = st.multiselect("评价指标列", options=num_cols, default=num_cols[:min(5, len(num_cols))], key="critic_ind")
        if _crit_ind:
            _crit_dir = {}
            _cc = st.columns(min(len(_crit_ind), 3))
            for i, c in enumerate(_crit_ind):
                with _cc[i % len(_cc)]:
                    _crit_dir[c] = st.selectbox(f"{c} 方向", ["正向(越大越好)", "负向(越小越好)"], index=0, key=f"crit_dir_{c}")
            if st.button("🚀 计算 CRITIC 权重", type="primary", key="critic_btn"):
                try:
                    with st.spinner("计算 CRITIC 权重..."):
                        _X = df[_crit_ind].astype(float)
                        _Xn = pd.DataFrame(index=_X.index)
                        for c in _crit_ind:
                            x = _X[c]
                            if _crit_dir.get(c, "正向(越大越好)").startswith("正向"):
                                _Xn[c] = (x - x.min()) / (x.max() - x.min() + 1e-12)
                            else:
                                _Xn[c] = (x.max() - x) / (x.max() - x.min() + 1e-12)
                        _Xn = _Xn.clip(1e-12, None)
                        _std = _Xn.std()
                        _corr = _Xn.corr().abs()
                        _c = _std * (1 - _corr).sum(axis=1)
                        _w = _c / _c.sum()
                        _score = (_Xn * _w).sum(axis=1)
                        _wdf = _w.round(4).reset_index(); _wdf.columns = ["指标", "CRITIC权重"]
                        st.subheader("📊 CRITIC 权重")
                        st.dataframe(_wdf, use_container_width=True)
                        _res = pd.DataFrame({
                            "样本标识": (df[id_col] if id_col else range(len(df))),
                            "综合得分": _score.round(4),
                        }).sort_values("综合得分", ascending=False).reset_index(drop=True)
                        _res.insert(0, "排名", _res.index + 1)
                        st.subheader("🏆 CRITIC 综合得分与排名")
                        st.dataframe(_res, use_container_width=True)
                        _buf = BytesIO()
                        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
                            _wdf.to_excel(_w, sheet_name="CRITIC权重", index=False)
                            _res.to_excel(_w, sheet_name="CRITIC得分", index=False)
                        _buf.seek(0)
                        st.download_button("⬇️ 下载 CRITIC 结果 (Excel)", _buf, "critic_results.xlsx",
                                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="critic_dl")
                except Exception as e:
                    st.error(f"CRITIC 计算失败：{e}"); st.code(traceback.format_exc())
    with _t_pca:
        st.caption("主成分分析（PCA）通过正交变换将多个相关指标降维为少数互不相关的主成分，方差贡献率衡量信息保留程度。")
        _pca_ind = st.multiselect("分析变量", options=num_cols, default=num_cols[:min(5, len(num_cols))], key="pca_ind")
        if _pca_ind:
            _pca_n = st.number_input("提取主成分数（0=按累计贡献率≥85%自动）", min_value=0, max_value=len(_pca_ind), value=0, key="pca_n")
            if st.button("🚀 运行 PCA", type="primary", key="pca_btn"):
                try:
                    with st.spinner("运行 PCA..."):
                        from sklearn.decomposition import PCA as SKPCA
                        _Z = df[_pca_ind].astype(float)
                        _Zs = (_Z - _Z.mean()) / _Z.std().replace(0, 1)
                        _n = _pca_n if _pca_n > 0 else len(_pca_ind)
                        _p = SKPCA(n_components=_n).fit(_Zs)
                        _evr = _p.explained_variance_ratio_
                        _cum = np.cumsum(_evr)
                        if _pca_n == 0:
                            _keep = int(np.searchsorted(_cum, 0.85) + 1)
                            _keep = min(_keep, len(_pca_ind))
                            _n = _keep
                            _p = SKPCA(n_components=_n).fit(_Zs)
                            _evr = _p.explained_variance_ratio_
                            _cum = np.cumsum(_evr)
                        _comp = _p.components_
                        _scores = _p.transform(_Zs)
                        st.subheader("📊 方差解释率")
                        _vdf = pd.DataFrame({
                            "主成分": [f"PC{i+1}" for i in range(_n)],
                            "特征值": np.round(_p.explained_variance_, 4),
                            "方差贡献率": np.round(_evr, 4),
                            "累计贡献率": np.round(_cum, 4),
                        })
                        st.dataframe(_vdf, use_container_width=True)
                        st.subheader("🧭 因子载荷矩阵（指标×主成分）")
                        _load = pd.DataFrame(_comp.T, index=_pca_ind, columns=[f"PC{i+1}" for i in range(_n)])
                        st.dataframe(_load.round(4), use_container_width=True)
                        st.subheader("🏆 主成分得分（前20样本）")
                        _sdf = pd.DataFrame(_scores, columns=[f"PC{i+1}" for i in range(_n)], index=_Z.index)
                        st.dataframe(_sdf.head(20).round(4), use_container_width=True)
                        _composite = (_scores * _evr).sum(axis=1) / _evr.sum()
                        _cdf = pd.DataFrame({"样本标识": (df[id_col] if id_col else range(len(df))), "PCA综合得分": _composite.round(4)}).sort_values("PCA综合得分", ascending=False).reset_index(drop=True)
                        _cdf.insert(0, "排名", _cdf.index + 1)
                        st.dataframe(_cdf, use_container_width=True)
                        _buf = BytesIO()
                        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
                            _vdf.to_excel(_w, sheet_name="方差解释率", index=False)
                            _load.to_excel(_w, sheet_name="因子载荷", index=False)
                            _sdf.to_excel(_w, sheet_name="主成分得分", index=False)
                            _cdf.to_excel(_w, sheet_name="综合得分", index=False)
                        _buf.seek(0)
                        st.download_button("⬇️ 下载 PCA 结果 (Excel)", _buf, "pca_results.xlsx",
                                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pca_dl")
                except Exception as e:
                    st.error(f"PCA 失败：{e}"); st.code(traceback.format_exc())
    with _t_dea:
        st.caption("数据包络分析（DEA）以相对效率前沿衡量决策单元（DMU）的技术效率。此处实现 CCR 模型（规模报酬不变，输入导向）。")
        _dea_inputs = st.multiselect("投入指标", options=num_cols, default=num_cols[:min(2, len(num_cols))], key="dea_in")
        _dea_out_cands = [c for c in num_cols if c not in _dea_inputs]
        _dea_outputs = st.multiselect("产出指标", options=_dea_out_cands, default=_dea_out_cands[:min(2, len(_dea_out_cands))], key="dea_out")
        if _dea_inputs and _dea_outputs:
            if st.button("🚀 运行 DEA (CCR)", type="primary", key="dea_btn"):
                try:
                    with st.spinner("运行 DEA..."):
                        from scipy.optimize import linprog
                        _X = df[_dea_inputs].astype(float).values
                        _Y = df[_dea_outputs].astype(float).values
                        _n = _X.shape[0]
                        _eff = np.ones(_n)
                        for k in range(_n):
                            _m_in = _X.shape[1]; _m_out = _Y.shape[1]
                            _nvar = 1 + _n
                            _c = np.zeros(_nvar); _c[0] = 1.0
                            _A = []; _b = []
                            for i in range(_m_in):
                                _row = np.zeros(_nvar)
                                _row[0] = -_X[k, i]
                                _row[1:] = _X[:, i]
                                _A.append(_row); _b.append(0.0)
                            for r in range(_m_out):
                                _row = np.zeros(_nvar)
                                _row[1:] = -_Y[:, r]
                                _A.append(_row); _b.append(-_Y[k, r])
                            _bounds = [(0, None)] * _nvar
                            _res = linprog(_c, A_ub=np.array(_A), b_ub=np.array(_b), bounds=_bounds, method="highs")
                            _eff[k] = _res.x[0] if _res.success else np.nan
                        _ddf = pd.DataFrame({
                            "DMU标识": (df[id_col] if id_col else range(_n)),
                            "CCR效率值": np.round(_eff, 4),
                            "是否有效": np.where(_eff >= 0.999, "有效", "无效"),
                        })
                        st.subheader("📊 DEA-CCR 技术效率")
                        st.dataframe(_ddf, use_container_width=True)
                        _buf = BytesIO()
                        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
                            _ddf.to_excel(_w, sheet_name="DEA效率", index=False)
                        _buf.seek(0)
                        st.download_button("⬇️ 下载 DEA 结果 (Excel)", _buf, "dea_results.xlsx",
                                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dea_dl")
                except Exception as e:
                    st.error(f"DEA 失败：{e}"); st.code(traceback.format_exc())

    # ===================== 指标标准化 / 归一化 =====================
    with _t_std:
        st.caption("将指标转换为可比量纲，是赋权与综合评价的前置步骤。支持正向/负向指标的方向处理。")
        _std_method = st.selectbox(
            "标准化 / 归一化方法",
            ["离差标准化 (0-1)", "极差标准化 (-1~1)", "Z-score 标准化", "MinMax 归一化 (0-1)", "向量单位化", "对数化"],
            key="std_method",
        )
        if indicator_cols:
            try:
                _X = df[indicator_cols].astype(float)
                _out = pd.DataFrame(index=_X.index)
                for c in indicator_cols:
                    _s = _X[c]
                    _mn, _mx, _mu, _sd = _s.min(), _s.max(), _s.mean(), _s.std(ddof=0)
                    _m = _std_method
                    if _m == "离差标准化 (0-1)":
                        _v = (_s - _mn) / (_mx - _mn) if _mx > _mn else _s * 0
                    elif _m == "极差标准化 (-1~1)":
                        _r = (_mx - _mn) / 2
                        _v = (_s - _mu) / _r if _r > 0 else _s * 0
                    elif _m == "Z-score 标准化":
                        _v = (_s - _mu) / _sd if _sd > 0 else _s * 0
                    elif _m == "MinMax 归一化 (0-1)":
                        _v = (_s - _mn) / (_mx - _mn) if _mx > _mn else _s * 0
                    elif _m == "向量单位化":
                        _nrm = np.sqrt((_s ** 2).sum())
                        _v = _s / _nrm if _nrm > 0 else _s * 0
                    else:
                        _minpos = _s[_s > 0].min() if (_s > 0).any() else 1.0
                        _v = np.log(_s.clip(lower=_minpos))
                    if "负向" in directions.get(c, "正向(越大越好)"):
                        if _m in ("Z-score 标准化", "向量单位化", "对数化", "极差标准化 (-1~1)"):
                            _v = -_v
                        else:
                            _v = 1 - _v
                    _out[c] = _v
                st.subheader("📊 标准化结果（前 50 行）")
                st.dataframe(_out.round(4).head(50))
                _show_table(_out.round(4), "standardized_result.xlsx", "标准化")
            except Exception as _e:
                st.error(f"标准化失败：{_e}"); st.code(traceback.format_exc())

    # ===================== 综合评价法 =====================
    with _t_comp:
        st.caption("将多个指标合成为单一综合得分：① 加权平均（自定义权重或等权）；② Z-score 综合法（各指标标准化后等权求和，自动处理方向）。")
        _comp_method = st.radio("合成方式", ["加权平均（自定义权重）", "加权平均（等权）", "Z-score 综合法"], horizontal=True, key="comp_method")
        if indicator_cols:
            try:
                _X = df[indicator_cols].astype(float)
                _Xz = (_X - _X.mean()) / _X.std(ddof=0)
                for c in indicator_cols:
                    if "负向" in directions.get(c, "正向(越大越好)"):
                        _Xz[c] = -_Xz[c]
                if "Z-score" in _comp_method:
                    _w = pd.Series(1.0 / len(indicator_cols), index=indicator_cols)
                    _score = _Xz.mean(axis=1)
                else:
                    if "等权" in _comp_method:
                        _w = pd.Series(1.0 / len(indicator_cols), index=indicator_cols)
                    else:
                        _wc = st.columns(min(len(indicator_cols), 3))
                        _wd = {}
                        for i, c in enumerate(indicator_cols):
                            with _wc[i % len(_wc)]:
                                _wd[c] = st.number_input(f"{c} 权重", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}")
                        _ws = sum(_wd.values()) or 1.0
                        _w = pd.Series({c: v / _ws for c, v in _wd.items()})
                    _score = (_X * _w).sum(axis=1)
                _res = pd.DataFrame({"综合得分": _score.round(4)})
                if id_col is not None:
                    _res.insert(0, id_col, df[id_col].values)
                if year_col is not None:
                    _res.insert(1 if id_col is not None else 0, year_col, df[year_col].values)
                _res["排名"] = _res["综合得分"].rank(ascending=False, method="min").astype(int)
                st.subheader("📊 综合评价结果")
                st.dataframe(_res.sort_values("排名").head(100))
                _show_table(_res, "composite_score.xlsx", "综合评价")
                st.caption(f"使用权重：{dict(_w.round(4))}")
            except Exception as _e:
                st.error(f"综合评价失败：{_e}"); st.code(traceback.format_exc())

    # ===================== 多因子分析 FA =====================
    with _t_fa:
        st.caption("因子分析(FA)通过潜在公共因子解释观测变量间的相关性，关注共享方差（与 PCA 纯方差分解不同）。需先标准化。")
        _fa_ind = st.multiselect("因子分析指标列", options=num_cols, default=num_cols[:min(5, len(num_cols))], key="fa_ind")
        if _fa_ind:
            _fa_nc = st.slider("提取因子数", 1, len(_fa_ind), min(3, len(_fa_ind)), key="fa_nc")
            if st.button("🚀 估计因子模型", type="primary", key="fa_btn"):
                try:
                    from sklearn.decomposition import FactorAnalysis
                    from sklearn.preprocessing import StandardScaler
                    _X = df[_fa_ind].astype(float).dropna()
                    _Xs = StandardScaler().fit_transform(_X)
                    _fa = FactorAnalysis(n_components=_fa_nc, random_state=0).fit(_Xs)
                    _load = pd.DataFrame(_fa.components_.T, index=_fa_ind, columns=[f"F{i+1}" for i in range(_fa_nc)])
                    _cols = [f"F{i+1}" for i in range(_fa_nc)]
                    _load["共同度"] = (_load[_cols] ** 2).sum(axis=1).round(4)
                    st.subheader("📊 因子载荷矩阵")
                    _show_table(_load.round(4), "factor_loadings.xlsx", "载荷")
                    _unique = pd.DataFrame({"唯一度(1-共同度)": (1 - _load["共同度"]).round(4)}, index=_fa_ind)
                    _show_table(_unique, "factor_uniqueness.xlsx", "唯一度")
                    st.info("载荷绝对值越大，该变量与因子关联越强；共同度越高说明该变量被因子解释的比例越大。")
                except Exception as _e:
                    st.error(f"因子分析失败：{_e}"); st.code(traceback.format_exc())

# ================================================================
#                    第五章：耦合协调度模型 (CCDM)
# ================================================================
elif page == "5. 耦合协调度模型":
    st.header("第五章：耦合协调度模型 (Coupling Coordination Degree Model)")
    st.caption(
        "用于衡量两个子系统（如「数字化水平」与「企业经营绩效」）之间的耦合互动程度与协调发展水平。"
        "输出各子系统综合发展指数 U、耦合度 C、耦合协调度 D 及协调等级，并提供分年度趋势图与 Excel 下载。"
    )
    if "merged_df" not in st.session_state or st.session_state.merged_df is None:
        st.warning("⚠️ 尚未生成清洗后的数据，请先在「1. 数据清洗」页点击「开始清洗」。")
    else:
        df = st.session_state.merged_df.copy()
        col_id = st.session_state.get("col_id")
        col_year = st.session_state.get("col_year")
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ([col_id, col_year] if col_id and col_year else [])
        ]
        if len(num_cols) < 2:
            st.warning("数值指标不足，无法进行耦合协调度分析。")
        else:
            st.subheader("① 选择两个子系统指标")
            sub1 = st.multiselect("子系统1 指标（如数字化水平）", options=num_cols, key="ccdm_s1")
            rest = [c for c in num_cols if c not in sub1]
            sub2 = st.multiselect("子系统2 指标（如企业经营绩效）", options=rest, key="ccdm_s2")
            neg = st.multiselect("反向指标（越小越好，将做正向化处理）", options=sub1 + sub2, key="ccdm_neg")

            st.subheader("② 权重与参数")
            w_method = st.radio("指标权重方法", ["熵权法（客观赋权）", "等权"], key="ccdm_w")
            alpha = st.slider(
                "子系统1 在综合发展指数 T 中的权重 α（子系统2 权重 = 1−α）",
                0.0, 1.0, 0.5, 0.05, key="ccdm_alpha",
            )

            if st.button("▶️ 计算耦合协调度", key="run_ccdm"):
                if not sub1 or not sub2:
                    st.warning("请分别为两个子系统至少选择一个指标。")
                else:
                    try:
                        used = list(dict.fromkeys(sub1 + sub2))
                        dd = df[used].apply(pd.to_numeric, errors="coerce").dropna()

                        def _norm(s):
                            col = s.name
                            x = s
                            mn, mx = x.min(), x.max()
                            if mx - mn < 1e-12:
                                return pd.Series(0.5, index=x.index)
                            return (mx - x) / (mx - mn) if col in neg else (x - mn) / (mx - mn)

                        def _weights(cols):
                            if w_method.startswith("等权"):
                                return np.ones(len(cols)) / len(cols)
                            X = dd[cols].apply(_norm).clip(1e-12, None)   # 熵权法基于已[0,1]化数据
                            P = X.div(X.sum(axis=0), axis=1)
                            k = 1 / np.log(len(X))
                            E = -k * (P * np.log(P)).sum(axis=0)
                            w = (1 - E) / (1 - E).sum()
                            return w.values

                        w1, w2 = _weights(sub1), _weights(sub2)
                        X1 = dd[sub1].apply(_norm)
                        X2 = dd[sub2].apply(_norm)
                        U1 = (X1 * w1).sum(axis=1)
                        U2 = (X2 * w2).sum(axis=1)
                        eps = 1e-12
                        C = 2 * np.sqrt(U1 * U2) / (U1 + U2 + eps)          # 两系统耦合度
                        T = alpha * U1 + (1 - alpha) * U2                   # 综合发展指数
                        D = np.sqrt(C * T)                                  # 耦合协调度

                        def _level(d):
                            return ("极度失调" if d < 0.1 else "严重失调" if d < 0.2 else "中度失调" if d < 0.3
                                    else "轻度失调" if d < 0.4 else "濒临失调" if d < 0.5 else "勉强协调" if d < 0.6
                                    else "初级协调" if d < 0.7 else "中级协调" if d < 0.8 else "良好协调" if d < 0.9
                                    else "优质协调")

                        level = D.apply(_level)
                        id_ser = df.loc[dd.index, col_id] if (col_id and col_id in df.columns) else pd.Series(dd.index, index=dd.index)
                        yr_ser = df.loc[dd.index, col_year] if (col_year and col_year in df.columns) else pd.Series(range(len(dd)), index=dd.index)
                        res = pd.DataFrame({
                            (col_id or "个体"): id_ser.values,
                            (col_year or "年份"): yr_ser.values,
                            "子系统1综合指数U1": U1.round(4).values,
                            "子系统2综合指数U2": U2.round(4).values,
                            "耦合度C": C.round(4).values,
                            "综合发展指数T": T.round(4).values,
                            "耦合协调度D": D.round(4).values,
                            "协调等级": level.values,
                        })
                        st.markdown("##### 耦合协调度测算结果（前 50 行）")
                        st.dataframe(res.head(50), use_container_width=True)

                        # 分年度趋势
                        if col_year and col_year in df.columns:
                            _yr = pd.Series(yr_ser.values)
                            _trend = pd.DataFrame({"年份": _yr, "耦合协调度D": D.round(4).values}).groupby("年份").mean().reset_index()
                            st.markdown("##### 耦合协调度 D 的年度趋势")
                            st.line_chart(_trend.set_index("年份")["耦合协调度D"])

                        # Excel 下载
                        _buf = BytesIO()
                        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
                            res.to_excel(_w, index=False, sheet_name="耦合协调度")
                        _buf.seek(0)
                        st.download_button(
                            "📥 下载耦合协调度结果 (Excel)",
                            data=_buf.getvalue(),
                            file_name="coupling_coordination.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_ccdm",
                        )
                        st.info(
                            f"样本量 {len(res)} 条；平均耦合度 C = {C.mean():.4f}，平均耦合协调度 D = {D.mean():.4f}。"
                            f"D∈[0,1]，越接近 1 表示两系统协调发展水平越高。"
                        )
                    except Exception as _e:
                        st.error(f"耦合协调度计算失败：{_e}")
                        st.code(traceback.format_exc())

# ================================================================
#                    第五阶段：内生性检验
# ================================================================
# ================================================================
#                    第七章：双重差分 DID + 事件研究法
# ================================================================
elif page == "7. DID + 事件研究法":
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm
    from io import BytesIO

    st.header("第七章：双重差分 (DID) 与事件研究法")

    # ---------- 数据与变量准备（动态获取，不硬编码） ----------
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先前往「1. 数据清洗」完成数据清洗！")
        st.stop()
    _df_raw = st.session_state.merged_df
    _col_id = st.session_state.get("col_id")
    _col_year = st.session_state.get("col_year")
    if _col_id is None or _col_year is None:
        st.error("请先在「数据清洗」页面选择「实体列」和「时间列」。")
        st.stop()

    # 构造面板（保留原始列名，不做 rename）
    _df_panel = _df_raw.dropna(subset=[_col_id, _col_year]).copy()
    _df_panel[_col_id] = _df_panel[_col_id].astype(str)
    try:
        _df_panel[_col_year] = _df_panel[_col_year].astype(int)
    except Exception:
        pass
    _df_panel = _df_panel.set_index([_col_id, _col_year]).sort_index()

    _all_cols = _df_panel.columns.tolist()
    _num_cols = _df_panel.select_dtypes(include=[np.number]).columns.tolist()

    # ---------- 公共辅助函数（与内生性检验页一致） ----------
    def _get_se(model, name):
        if hasattr(model, "std_errors"):
            return model.std_errors.get(name, None)
        if hasattr(model, "bse"):
            return model.bse.get(name, None)
        return None

    def _fmt_coef(param, se, pval):
        if param is None or se is None:
            return ""
        try:
            if pd.isna(param) or pd.isna(se):
                return ""
        except Exception:
            pass
        star = (
            "***" if pval < 0.01
            else "**" if pval < 0.05
            else "*" if pval < 0.1
            else ""
        )
        return f"{param:.4f}{star}({se:.4f})"

    def _show_table(display_df, fname, sheet="结果"):
        st.markdown(display_df.to_html(index=False, escape=False), unsafe_allow_html=True)
        _buf = BytesIO()
        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
            display_df.to_excel(_w, index=False, sheet_name=sheet)
        _buf.seek(0)
        st.download_button(
            "📥 下载结果 (Excel)",
            data=_buf.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{fname}",
        )

    def _fe_clusters(d):
        return pd.DataFrame({"entity": d.index.get_level_values(0)}, index=d.index)

    def _drop_absorbed(d, exog_df, use_entity, use_time):
        """丢弃在固定效应维度上无变异（会被完全吸收）的列，避免 AbsorbingEffectError。"""
        gi = d.index.get_level_values(0)
        gt = d.index.get_level_values(1)
        keep = []
        for c in exog_df.columns:
            v = exog_df[c]
            bad = False
            if use_entity:
                s = v.groupby(gi)
                if (s.transform("max") - s.transform("min")).abs().max() < 1e-12:
                    bad = True
            if (not bad) and use_time:
                s = v.groupby(gt)
                if (s.transform("max") - s.transform("min")).abs().max() < 1e-12:
                    bad = True
            if not bad:
                keep.append(c)
        return exog_df[keep]

    # ---------- 变量选择 ----------
    _y_var = st.selectbox("被解释变量 (Y)", options=_num_cols, key="did_y")
    _ctrl_candidates = [c for c in _num_cols if c != _y_var]
    _control_vars = st.multiselect("控制变量", options=_ctrl_candidates, key="did_ctrl")

    st.markdown(
        "**选择指导**：当存在一项政策或外部冲击，使得部分个体受到处理（处理组）、"
        "部分未受处理（对照组），且你有处理前后的数据时使用。需指定处理组标识列（0/1）"
        "和政策时点列（0/1）。输出交互项系数、平行趋势检验（事件研究图）。"
    )
    _treat_col = st.selectbox("处理组虚拟变量列（1=处理组）", options=_all_cols, key="did_treat")
    _post_col = st.selectbox("政策时点虚拟变量列（1=政策后）", options=_all_cols, key="did_post")
    _ref_rel = st.selectbox("事件研究参考期（省略的相对年份）", options=["-1", "0"], index=0, key="did_ref")

    if st.button("▶️ 运行 DID + 事件研究", key="run_did"):
        if not _treat_col or not _post_col or not _y_var:
            st.warning("请选择 Y、处理组列、政策时点列。")
        else:
            try:
                _dp = _df_panel.copy()
                _dp["__did__"] = _dp[_treat_col].astype(float) * _dp[_post_col].astype(float)
                _use = [_y_var, _treat_col, _post_col, "__did__"] + _control_vars
                _dp = _dp[_use].apply(pd.to_numeric, errors="coerce").dropna()
                _clu = _fe_clusters(_dp)
                # 双向 FE 下 treat(个体不变)、post(时间不变) 会被吸收，仅保留交互项 did 等可识别变量；不加入 const
                _exog = _drop_absorbed(_dp, _dp[[_treat_col, _post_col, "__did__"] + _control_vars], True, True)
                _did = PanelOLS(_dp[_y_var], _exog, entity_effects=True, time_effects=True).fit(
                    cov_type="clustered", clusters=_clu
                )

                # ---------- 事件研究：全相对年份 lead/lag（含政策后动态效应） ----------
                _yr = _dp.index.get_level_values(1)
                _yrs_u = sorted(pd.Series(_yr).unique().tolist())
                _post_yrs = _dp[_dp[_post_col].astype(float) == 1].index.get_level_values(1).unique()
                _treat_yr = min(_post_yrs) if len(_post_yrs) > 0 else (_yrs_u[-1] if _yrs_u else 0)
                _ref_rel_int = int(_ref_rel)
                _dp_ev = _dp.copy()
                _ev_cols = []  # (列名, 相对年份)
                for _y in _yrs_u:
                    _rel = _y - _treat_yr
                    if _rel == _ref_rel_int:
                        continue
                    _cn = f"__ev_{_rel}__"
                    _dp_ev[_cn] = _dp_ev[_treat_col].astype(float) * (_yr == _y).astype(float)
                    _ev_cols.append((_cn, _rel))
                _ev_rows = []
                if _ev_cols:
                    _clu_ev = _fe_clusters(_dp_ev)
                    _exog_ev = _drop_absorbed(_dp_ev, _dp_ev[[c for c, _ in _ev_cols] + _control_vars], True, True)
                    _ev = PanelOLS(_dp_ev[_y_var], _exog_ev, entity_effects=True, time_effects=True).fit(
                        cov_type="clustered", clusters=_clu_ev
                    )
                    for _cn, _rel in _ev_cols:
                        if _cn in _ev.params:
                            _ev_rows.append({
                                "相对年份": _rel,
                                "相对年份标签": f"t={_rel:+d}",
                                "事件研究系数": float(_ev.params[_cn]),
                                "标准误": _get_se(_ev, _cn),
                                "p值": float(_ev.pvalues[_cn]),
                            })
                    # 平行趋势联合检验：所有相对年份 < 0（政策前）的系数联合为 0
                    try:
                        _pre = [c for c, r in _ev_cols if r < 0]
                        if _pre:
                            _n = len(_ev.params)
                            _R = np.zeros((len(_pre), _n))
                            for _i, _c in enumerate(_pre):
                                _R[_i, list(_ev.params.index).index(_c)] = 1.0
                            _wald = _ev.wald_test(_R)
                            _pt_f = float(_wald.stat)
                            _pt_p = float(_wald.pval)
                        else:
                            _pt_f = _pt_p = np.nan
                    except Exception:
                        _pt_f = _pt_p = np.nan
                else:
                    _pt_f = _pt_p = np.nan

                # ---------- DID 主结果表 ----------
                _vars_show = ["__did__", _treat_col, _post_col] + _control_vars + ["const"]
                _rows = {}
                for _v in _vars_show:
                    if _v in _did.params:
                        _rows[_v] = {
                            "变量": _v.replace("__did__", "处理×政策(DID)"),
                            "DID 估计": _fmt_coef(_did.params[_v], _get_se(_did, _v), _did.pvalues[_v]),
                        }
                _rows["平行趋势F"] = {
                    "变量": "平行趋势联合检验 F(p)",
                    "DID 估计": f"{_pt_f:.4f}(p={_pt_p:.4f})" if not np.isnan(_pt_f) else "样本不足",
                }
                _rows["观测数"] = {"变量": "观测数", "DID 估计": int(_did.nobs)}
                _disp = pd.DataFrame(
                    [_rows[k] for k in (_vars_show + ["平行趋势F", "观测数"]) if k in _rows]
                )
                _disp = _disp[["变量", "DID 估计"]]
                st.markdown("##### DID 结果")
                _show_table(_disp, "did_results.xlsx", "DID")

                if _ev_rows:
                    _ev_df = pd.DataFrame(_ev_rows).sort_values("相对年份").reset_index(drop=True)
                    st.markdown("##### 事件研究图（平行趋势检验）")
                    st.dataframe(
                        _ev_df[["相对年份标签", "事件研究系数", "标准误", "p值"]],
                        use_container_width=True,
                    )
                    # 折线图 + 0 参考线
                    _chart = _ev_df.set_index("相对年份标签")[["事件研究系数"]].copy()
                    _chart["0参考线"] = 0.0
                    st.line_chart(_chart)
                    st.info(
                        f"平行趋势联合检验 p = {_pt_p:.4f}（应 > 0.10，说明政策前处理组与对照组无显著差异）；"
                        "政策后相对年份系数反映动态调整效应。"
                    )
                st.info("DID 交互项系数即平均处理效应 (ATT)；事件研究图展示处理前后的动态效应。")
            except Exception as _e:
                st.error(f"DID 出错：{_e}")
                st.code(traceback.format_exc())


# ================================================================
#                    第六章：内生性检验（IV/2SLS 等）
# ================================================================
elif page == "6. 内生性检验":
    from linearmodels.iv import IV2SLS, IVGMM
    from linearmodels.panel import PanelOLS, RandomEffects
    import statsmodels.api as sm
    from scipy import stats
    from io import BytesIO

    st.header("第五章：内生性检验")

    # ---------- 数据与变量准备（全部动态获取，不硬编码） ----------
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先前往「1. 数据清洗」完成数据清洗！")
        st.stop()

    _df_raw = st.session_state.merged_df
    _col_id = st.session_state.get("col_id")
    _col_year = st.session_state.get("col_year")
    if _col_id is None or _col_year is None:
        st.error("请先在「数据清洗」页面选择「实体列」和「时间列」。")
        st.stop()

    # 构造面板（保留原始列名，不做任何 rename）
    _df_panel = _df_raw.dropna(subset=[_col_id, _col_year]).copy()
    _df_panel[_col_id] = _df_panel[_col_id].astype(str)
    try:
        _df_panel[_col_year] = _df_panel[_col_year].astype(int)
    except Exception:
        pass
    _df_panel = _df_panel.set_index([_col_id, _col_year]).sort_index()

    _all_cols = _df_panel.columns.tolist()
    _num_cols = _df_panel.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("🔧 通用变量选择（所有模块共享）")
    _y_var = st.selectbox("被解释变量 (Y)", options=_num_cols, key="endo_y")
    _x_candidates = [c for c in _num_cols if c != _y_var]
    _x_vars = st.multiselect("核心解释变量 (X)", options=_x_candidates, key="endo_x")
    _ctrl_candidates = [c for c in _num_cols if c not in [_y_var] + _x_vars]
    _control_vars = st.multiselect("控制变量", options=_ctrl_candidates, key="endo_ctrl")

    # ---------- 公共辅助函数 ----------
    def _get_se(model, name):
        """兼容 OLS(bse) 与 PanelOLS/IV(std_errors) 的标准误提取。"""
        if hasattr(model, "std_errors"):
            return model.std_errors.get(name, None)
        if hasattr(model, "bse"):
            return model.bse.get(name, None)
        return None

    def _fmt_coef(param, se, pval):
        if param is None or se is None:
            return ""
        try:
            if pd.isna(param) or pd.isna(se):
                return ""
        except Exception:
            pass
        star = (
            "***" if pval < 0.01
            else "**" if pval < 0.05
            else "*" if pval < 0.1
            else ""
        )
        return f"{param:.4f}{star}({se:.4f})"

    def _show_table(display_df, fname, sheet="结果"):
        st.markdown(
            display_df.to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )
        _buf = BytesIO()
        with pd.ExcelWriter(_buf, engine="openpyxl") as _w:
            display_df.to_excel(_w, index=False, sheet_name=sheet)
        _buf.seek(0)
        st.download_button(
            "📥 下载结果 (Excel)",
            data=_buf.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{fname}",
        )

    def _clean_panel(vars_used):
        """按所用变量删缺失，返回干净的面板 DataFrame（去重列名）。"""
        need = list(dict.fromkeys([c for c in vars_used if c in _df_panel.columns]))
        return _df_panel[need].dropna()

    def _fe_clusters(d):
        """构造与 d 同索引的个体聚类 DataFrame（linearmodels 7.x 正确用法：clusters 需为带相同索引的 DataFrame）。"""
        return pd.DataFrame({"entity": d.index.get_level_values(0)}, index=d.index)

    def _drop_absorbed(d, exog_df, use_entity, use_time):
        """丢弃在固定效应维度上无变异（会被完全吸收）的列，避免 AbsorbingEffectError。

        双向 / 单向 FE 下常数项必然被吸收，故调用方不应再 add_constant；
        任何在个体(或年份)维度上取值恒定的变量都会被相应 FE 吸收，此处自动剔除。
        """
        gi = d.index.get_level_values(0)
        gt = d.index.get_level_values(1)
        keep = []
        for c in exog_df.columns:
            v = exog_df[c]
            bad = False
            if use_entity:
                s = v.groupby(gi)
                if (s.transform("max") - s.transform("min")).abs().max() < 1e-12:
                    bad = True
            if (not bad) and use_time:
                s = v.groupby(gt)
                if (s.transform("max") - s.transform("min")).abs().max() < 1e-12:
                    bad = True
        if not bad:
            keep.append(c)
        return exog_df[keep]

    def _ab_gmm(_y, _xex, _idv, _tv, _arl=1, _gmax=4, _sys=True):
        """Arellano-Bond 差分 / 系统 GMM（2 步最优权重）。
        _y, _xex: 数组（_xex 为外生/前定 regressor，可为空）；_idv, _tv: 实体/时间标识。
        仅支持 AR(1)（滞后因变量作为内生项）。返回 (beta, se, diag)。"""
        from scipy.stats import chi2, norm
        _y = np.asarray(_y, float)
        if _xex.ndim == 1:
            _xex = _xex.reshape(-1, 1)
        _n_x = _xex.shape[1]
        _idv = np.asarray(_idv); _tv = np.asarray(_tv)
        _uniq = np.unique(_idv)
        _dX, _dZ, _dY, _dMeta, _lX, _lZ, _lY, _lMeta = [], [], [], [], [], [], [], []
        _ninstr = (_gmax - 1) + _n_x * (_gmax - 1)
        _tot = _ninstr + _n_x
        for _i in _uniq:
            _m = _idv == _i
            _t = _tv[_m]; _o = np.argsort(_t); _t = _t[_o]
            _yi = _y[_m][_o]; _xi = _xex[_m][_o]
            _T = len(_t)
            if _T < 4:
                continue
            _dy = np.diff(_yi); _dx = np.diff(_xi, axis=0)
            for _s in range(1, _T - 1):
                _tt = _s + 1
                _xr = [_dy[_s - 1]] + list(_dx[_s])
                _zr = [0.0] * _tot
                _c = 0
                for _l in range(2, min(_gmax, _tt) + 1):
                    _zr[_c] = _yi[_tt - _l] if _tt - _l >= 0 else 0.0; _c += 1
                for _kx in range(_n_x):
                    for _l in range(2, min(_gmax, _tt) + 1):
                        _zr[(_gmax - 1) + _kx * (_gmax - 1) + (_l - 2)] = _xi[_tt - _l, _kx] if _tt - _l >= 0 else 0.0
                for _kx in range(_n_x):
                    _zr[_ninstr + _kx] = _dx[_s, _kx]
                _dX.append(_xr); _dZ.append(_zr); _dY.append(_dy[_s]); _dMeta.append(_i)
                if _sys and _tt >= 3:
                    _xrl = [_yi[_tt - 1]] + list(_xi[_tt])
                    _zrl = [0.0] * _tot
                    _c = 0
                    for _l in range(1, min(_gmax - 1, _tt - 1) + 1):
                        _zrl[_c] = _dy[_tt - 1 - _l] if _tt - 1 - _l >= 0 else 0.0; _c += 1
                    for _kx in range(_n_x):
                        for _l in range(1, min(_gmax - 1, _tt - 1) + 1):
                            _zrl[(_gmax - 1) + _kx * (_gmax - 1) + (_l - 1)] = _dx[_tt - 1 - _l, _kx] if _tt - 1 - _l >= 0 else 0.0
                    for _kx in range(_n_x):
                        _zrl[_ninstr + _kx] = _xi[_tt, _kx]
                    _lX.append(_xrl); _lZ.append(_zrl); _lY.append(_yi[_tt]); _lMeta.append(_i)
        _X = np.array(_dX); _Z = np.array(_dZ); _Y = np.array(_dY); _meta = _dMeta
        if _sys and _lX:
            _X = np.vstack([_X, np.array(_lX)]); _Z = np.vstack([_Z, np.array(_lZ)])
            _Y = np.concatenate([_Y, np.array(_lY)]); _meta = _dMeta + _lMeta
        _n, _k = _X.shape; _m = _Z.shape[1]
        _re = np.array(_meta)
        _ZtX = _Z.T @ _X; _ZtZ = _Z.T @ _Z; _ZtY = _Z.T @ _Y
        _b1 = np.linalg.solve(_ZtX.T @ np.linalg.solve(_ZtZ, _ZtX), _ZtX.T @ np.linalg.solve(_ZtZ, _ZtY))
        _u = _Y - _X @ _b1
        _W = np.zeros((_m, _m))
        for _i in np.unique(_re):
            _gi = (_Z[_re == _i].T * _u[_re == _i]).sum(axis=1); _W += np.outer(_gi, _gi)
        _W /= max(1, len(np.unique(_re)))
        _Wi = np.linalg.pinv(_W); _A = _ZtX.T @ _Wi @ _ZtX
        _b2 = np.linalg.solve(_A, _ZtX.T @ _Wi @ _ZtY)
        _u2 = _Y - _X @ _b2
        _W2 = np.zeros((_m, _m))
        for _i in np.unique(_re):
            _gi = (_Z[_re == _i].T * _u2[_re == _i]).sum(axis=1); _W2 += np.outer(_gi, _gi)
        _W2 /= max(1, len(np.unique(_re)))
        try:
            _V = np.linalg.inv(_A) @ (_ZtX.T @ _Wi @ _W2 @ _Wi @ _ZtX) @ np.linalg.inv(_A)
            _se = np.sqrt(np.maximum(np.diag(_V), 0))
        except Exception:
            _se = np.full(_k, np.nan)
        _g2 = np.zeros(_m)
        for _i in np.unique(_re):
            _gi = (_Z[_re == _i].T * _u2[_re == _i]).sum(axis=1); _g2 += _gi
        _g2 /= len(np.unique(_re))
        _J = _n * (_g2.T @ _Wi @ _g2)
        _hansen_p = chi2.sf(_J, max(1, _m - _k))
        # Arellano-Bond 序列相关检验：仅基于"差分方程"残差（_dY 部分），按实体分块后
        # 计算每实体的滞后1/滞后2自相关，跨实体平均并以单样本 t 检验 H0: 平均自相关=0。
        _d_resid = np.asarray(_u2[:len(_dY)], float)
        _ar1_vals, _ar2_vals = [], []
        for _i in np.unique(_dMeta):
            _e = _d_resid[np.array(_dMeta) == _i]
            if len(_e) < 3:
                continue
            _ar1_vals.append(np.corrcoef(_e[1:], _e[:-1])[0, 1])
            _ar2_vals.append(np.corrcoef(_e[2:], _e[:-2])[0, 1])
        _ar1 = float(np.mean(_ar1_vals)) if _ar1_vals else np.nan
        _ar2 = float(np.mean(_ar2_vals)) if _ar2_vals else np.nan
        def _ab_ttest(_vals):
            _v = np.array(_vals, float); _k = len(_v)
            if _k < 2:
                return np.nan
            _mm = np.mean(_v); _ss = np.std(_v, ddof=1)
            if _ss <= 0:
                return 0.0 if abs(_mm) < 1e-9 else np.nan
            return 2 * (1 - norm.cdf(abs(_mm) / (_ss / np.sqrt(_k))))
        _ar1_p = float(_ab_ttest(_ar1_vals)) if _ar1_vals else np.nan
        _ar2_p = float(_ab_ttest(_ar2_vals)) if _ar2_vals else np.nan
        return _b2, _se, {"ar1": float(_ar1), "ar1_p": float(_ar1_p), "ar2": float(_ar2) if not np.isnan(_ar2) else np.nan,
                          "ar2_p": float(_ar2_p) if not np.isnan(_ar2_p) else np.nan, "hansen": float(_J),
                          "hansen_p": float(_hansen_p), "nobs": int(_n), "n_instr": int(_m)}

    # ---------- 扩展方法 tab（PSM / 合成控制法 / 系统 GMM） ----------
    _tab_x1, _tab_x2, _tab_x3, _tab_x4, _tab_x5 = st.tabs([
        "🟪 倾向得分匹配 PSM / PSM-DID",
        "🟫 合成控制法 SCM",
        "⬛ 系统 GMM（动态面板 Arellano-Bond）",
        "🟦 Heckman 两步法",
        "🟧 DDD 三重差分",
    ])

    # ===================== PSM / PSM-DID =====================
    with _tab_x1:
        st.caption("倾向得分匹配（PSM）：用可观测协变量估计接受处理的概率（倾向得分），在得分相近的样本间做最近邻匹配，降低选择性偏差；PSM-DID 在匹配样本上再做双重差分。")
        _psm_treat = st.selectbox("处理变量 D (0/1)", options=_all_cols, key="psm_treat")
        _psm_y = st.selectbox("结果变量 Y", options=_num_cols, key="psm_y")
        _psm_cov = st.multiselect("协变量（估计倾向得分）", options=_num_cols, default=_num_cols[:min(4, len(_num_cols))], key="psm_cov")
        _psm_k = st.number_input("最近邻匹配数 k", min_value=1, max_value=5, value=1, key="psm_k")
        _psm_caliper = st.number_input("卡尺（倾向得分标准差倍数，0=不限）", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="psm_caliper")
        if _psm_treat and _psm_y and _psm_cov:
            if st.button("🚀 运行 PSM", type="primary", key="psm_btn"):
                try:
                    with st.spinner("估计倾向得分并匹配..."):
                        from sklearn.linear_model import LogisticRegression
                        _d = _df_panel[[_psm_treat] + _psm_cov + [_psm_y]].dropna()
                        _D = _d[_psm_treat].astype(float).values
                        _Xp = _d[_psm_cov].astype(float).values
                        _lr = LogisticRegression(max_iter=1000).fit(_Xp, _D)
                        _ps = _lr.predict_proba(_Xp)[:, 1]
                        _ti = np.where(_D == 1)[0]; _ci = np.where(_D == 0)[0]
                        _psc = _ps[_ci]
                        _thr = _psm_caliper * _psc.std() if _psm_caliper > 0 else np.inf
                        _used = set(); _mt = []; _mc = []
                        for _t in _ti:
                            _dist = np.abs(_psc - _ps[_t]); _ord = np.argsort(_dist); _cnt = 0
                            for _j in _ord:
                                if _j in _used:
                                    continue
                                if _dist[_j] <= _thr:
                                    _mt.append(_t); _mc.append(_ci[_j]); _used.add(_j); _cnt += 1
                                    if _cnt >= _psm_k:
                                        break
                            if _cnt == 0:
                                _mt.append(_t); _mc.append(_ci[_ord[0]]); _used.add(_ord[0])
                        _Yv = _d[_psm_y].astype(float).values
                        _att = _Yv[_mt].mean() - _Yv[_mc].mean()
                        _bal = []
                        for _c in _psm_cov:
                            _v = _d[_c].astype(float).values; _s = _v.std()
                            _smd_b = (_v[_ti].mean() - _v[_ci].mean()) / _s
                            _smd_a = (_v[_mt].mean() - _v[_mc].mean()) / ((_v[_mt].var() + _v[_mc].var()) ** 0.5)
                            _bal.append({"协变量": _c, "匹配前|SMD|": f"{abs(_smd_b):.4f}", "匹配后|SMD|": f"{abs(_smd_a):.4f}"})
                        _att_df = pd.DataFrame([
                            {"统计量": "处理组 Y 均值", "数值": f"{_Yv[_mt].mean():.4f}"},
                            {"统计量": "匹配对照组 Y 均值", "数值": f"{_Yv[_mc].mean():.4f}"},
                            {"统计量": "ATT（处理效应）", "数值": f"{_att:.4f}"},
                            {"统计量": "匹配处理单元数", "数值": str(len(_mt))},
                            {"统计量": "匹配对照单元数", "数值": str(len(set(_mc)))},
                        ])
                        st.subheader("📊 PSM 处理结果")
                        _show_table(_att_df, "psm_att.xlsx", "ATT")
                        st.subheader("⚖️ 协变量平衡性（|SMD|<0.1 为佳）")
                        _show_table(pd.DataFrame(_bal), "psm_balance.xlsx", "Balance")
                except Exception as _e:
                    st.error(f"PSM 失败：{_e}"); st.code(traceback.format_exc())

        with st.expander("🔁 PSM-DID（在匹配样本上做双重差分）"):
            _psm_post = st.selectbox("政策后虚拟列", options=_all_cols, key="psm_post")
            _psm_did_ctrl = st.multiselect("DID 控制变量", options=[c for c in _num_cols if c != _psm_y], key="psm_did_ctrl")
            if st.button("🚀 运行 PSM-DID", key="psmdid_btn"):
                try:
                    with st.spinner("匹配并估计 PSM-DID..."):
                        from sklearn.linear_model import LogisticRegression
                        _cols = list(dict.fromkeys([_psm_treat] + _psm_cov + [_psm_y] + [_psm_post] + _psm_did_ctrl))
                        _d = _df_panel[_cols].dropna()
                        _D = _d[_psm_treat].astype(float).values
                        _lr = LogisticRegression(max_iter=1000).fit(_d[_psm_cov].astype(float).values, _D)
                        _ps = _lr.predict_proba(_d[_psm_cov].astype(float).values)[:, 1]
                        _ti = np.where(_D == 1)[0]; _ci = np.where(_D == 0)[0]
                        _psc = _ps[_ci]
                        _thr = _psm_caliper * _psc.std() if _psm_caliper > 0 else np.inf
                        _used = set(); _mt = []; _mc = []
                        for _t in _ti:
                            _dist = np.abs(_psc - _ps[_t]); _ord = np.argsort(_dist); _cnt = 0
                            for _j in _ord:
                                if _j in _used:
                                    continue
                                if _dist[_j] <= _thr:
                                    _mt.append(_t); _mc.append(_ci[_j]); _used.add(_j); _cnt += 1
                                    if _cnt >= _psm_k:
                                        break
                            if _cnt == 0:
                                _mt.append(_t); _mc.append(_ci[_ord[0]]); _used.add(_ord[0])
                        _mm = _d.iloc[np.concatenate([_mt, _mc])].copy()
                        _mm["__D__"] = _mm[_psm_treat].astype(float)
                        _mm["__P__"] = _mm[_psm_post].astype(float)
                        _mm["__DP__"] = _mm["__D__"] * _mm["__P__"]
                        _clu = pd.DataFrame({"entity": _mm.index.get_level_values(0)}, index=_mm.index)
                        _did_exog = _drop_absorbed(_mm, _mm[["__D__", "__P__", "__DP__"] + _psm_did_ctrl], True, True)
                        _did = PanelOLS(_mm[_psm_y], _did_exog, entity_effects=True, time_effects=True).fit(cov_type="clustered", clusters=_clu)
                        _rows = {}
                        for _v in ["__D__", "__P__", "__DP__"] + _psm_did_ctrl + ["const"]:
                            if _v in _did.params:
                                _rows[_v] = {"变量": _v.replace("__DP__", "处理×政策(DID)").replace("__D__", "处理组").replace("__P__", "政策后"),
                                             "PSM-DID 估计": _fmt_coef(_did.params[_v], _get_se(_did, _v), _did.pvalues[_v])}
                        _rows["观测数"] = {"变量": "观测数", "PSM-DID 估计": int(_did.nobs)}
                        _disp = pd.DataFrame([_rows[k] for k in (["__D__", "__P__", "__DP__"] + _psm_did_ctrl + ["const", "观测数"]) if k in _rows])
                        _disp = _disp[["变量", "PSM-DID 估计"]]
                        st.subheader("📊 PSM-DID 结果（匹配样本）")
                        _show_table(_disp, "psm_did.xlsx", "PSM-DID")
                        st.info("交互项系数即匹配样本上的平均处理效应（ATT-DID）。")
                except Exception as _e:
                    st.error(f"PSM-DID 失败：{_e}"); st.code(traceback.format_exc())

    # ===================== 合成控制法 SCM =====================
    with _tab_x2:
        st.caption("合成控制法（SCM）：为处理单元构造一个由未处理单元加权合成的'反事实'，权重通过最小化处理前预测变量差异求得，再比较处理后真实值与合成值的差距。")
        _scm_y = st.selectbox("结果变量 Y", options=_num_cols, key="scm_y")
        _scm_pred = st.multiselect("预测变量（拟合权重，建议含各预处理期结果）", options=_num_cols, default=_num_cols[:min(3, len(_num_cols))], key="scm_pred")
        _scm_tu = st.selectbox("处理单元", options=sorted(_df_panel.index.get_level_values(0).unique().tolist()), key="scm_tu")
        _scm_tmin = int(_df_panel.index.get_level_values(1).min())
        _scm_tmax = int(_df_panel.index.get_level_values(1).max())
        _scm_cut = st.number_input("政策时点（含该年及之后为处理后）", min_value=_scm_tmin, max_value=_scm_tmax, value=_scm_tmax - 2, step=1, key="scm_cut")
        if st.button("🚀 运行 SCM", type="primary", key="scm_btn"):
            try:
                with st.spinner("求解合成控制权重..."):
                    _scm_cols = [_scm_y] + [c for c in _scm_pred if c != _scm_y]
                    _df_scm = _df_panel.dropna(subset=_scm_cols)
                    if len(_df_scm) < len(_df_panel):
                        st.warning(f"已剔除含缺失值的 {len(_df_panel) - len(_df_scm)} 行（结果变量/预测变量缺失）。")
                    _piv = _df_scm.pivot_table(index=_col_year, columns=_col_id, values=_scm_y)
                    # 结果变量的个体轨迹必须完整：剔除任一时期缺失的公司（避免非平衡面板缺口）
                    _piv = _piv.dropna(axis=1)
                    if str(_scm_tu) not in [str(c) for c in _piv.columns]:
                        st.error(f"处理单元「{_scm_tu}」的结果变量存在缺失，已被剔除，请换一个处理单元或先清洗数据。")
                        st.stop()
                    _pre = _piv[_piv.index < _scm_cut]
                    _post = _piv[_piv.index >= _scm_cut]
                    _donors = [u for u in _piv.columns if str(u) != str(_scm_tu)]
                    _pre_y_vec = _pre[_scm_tu].values
                    _X0 = _pre[_donors].values
                    if _scm_pred:
                        _extra = [c for c in _scm_pred if c != _scm_y]
                        if _extra:
                            _cov_pre = _df_panel[_extra].groupby(level=0).mean(numeric_only=True).reindex(_donors)
                            _cov_treat = _df_panel[_extra].groupby(level=0).mean(numeric_only=True).reindex([_scm_tu])
                            _X0 = np.vstack([_X0, _cov_pre.values.T])
                            _pre_y_vec = np.concatenate([_pre_y_vec, _cov_treat.values.flatten()])
                    from scipy.optimize import minimize
                    def _obj(w):
                        return np.sum((_pre_y_vec - _X0.dot(w)) ** 2)
                    _cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
                    _bnds = [(0.0, 1.0)] * len(_donors)
                    _res = minimize(_obj, np.ones(len(_donors)) / len(_donors), method="SLSQP", bounds=_bnds, constraints=_cons)
                    _w = _res.x
                    _syn_pre_y = _pre[_donors].values.dot(_w)
                    _pre_treat_y = _pre[_scm_tu].values
                    _pre_rmspe = np.sqrt(np.mean((_pre_treat_y - _syn_pre_y) ** 2))
                    _real_post = _post[_scm_tu].values
                    _syn_post = _post[_donors].values.dot(_w)
                    _gap = _real_post - _syn_post
                    _post_rmspe = np.sqrt(np.mean(_gap ** 2))
                    _wdf = pd.DataFrame({"单元": _donors, "权重": np.round(_w, 4)})
                    _wdf = _wdf[_wdf["权重"] > 0.001].sort_values("权重", ascending=False).reset_index(drop=True)
                    st.subheader("📊 合成控制权重")
                    _show_table(_wdf, "scm_weights.xlsx", "Weights")
                    _rmspe_df = pd.DataFrame([
                        {"统计量": "预处理 RMSPE（越小越好）", "数值": f"{_pre_rmspe:.4f}"},
                        {"统计量": "后处理 RMSPE", "数值": f"{_post_rmspe:.4f}"},
                        {"统计量": "后/前 RMSPE 比（>1 表明显著处理效应）", "数值": f"{_post_rmspe / _pre_rmspe:.4f}"},
                    ])
                    st.subheader("📈 拟合优度（RMSPE）")
                    _show_table(_rmspe_df, "scm_rmspe.xlsx", "RMSPE")
                    _gap_df = pd.DataFrame({"年份": _post.index.astype(int), "真实值": _real_post, "合成值": _syn_post, "差距": _gap})
                    st.subheader("📉 处理组 vs 合成控制（后处理差距）")
                    _show_table(_gap_df, "scm_gap.xlsx", "Gap")
                    _chart = _gap_df.set_index("年份")[["真实值", "合成值"]]
                    st.line_chart(_chart)
                    st.info(f"政策后平均处理效应≈ {_gap.mean():.4f}；RMSPE 比 = {_post_rmspe / _pre_rmspe:.2f}（远大于1 支持处理效应存在）。")
            except Exception as _e:
                st.error(f"SCM 失败：{_e}"); st.code(traceback.format_exc())

    # ===================== 系统 GMM =====================
    with _tab_x3:
        st.caption("系统 GMM（动态面板 Arellano-Bond / Blundell-Bond）：以滞后因变量作为内生项、用其更深滞后作为工具，2 步最优 GMM 估计；可叠加水平方程（系统 GMM）。报告 AR(1)/AR(2) 序列相关与 Hansen 过度识别检验。")
        _gmm_y = st.selectbox("被解释变量 Y（水平值）", options=_num_cols, key="gmm_y")
        _gmm_x = st.multiselect("外生/前定 regressor（可空，仅估计 AR(1)）", options=[c for c in _num_cols if c != _gmm_y], key="gmm_x")
        _gmm_gmax = st.number_input("工具变量最大滞后深度", min_value=2, max_value=6, value=4, step=1, key="gmm_gmax")
        _gmm_sys = st.checkbox("系统 GMM（叠加水平方程）", value=True, key="gmm_sys")
        if _gmm_y:
            if st.button("🚀 运行 系统 GMM", type="primary", key="gmm_btn"):
                try:
                    with st.spinner("估计动态面板 GMM..."):
                        _gd = _df_panel[[_gmm_y] + _gmm_x].dropna()
                        _yv = _gd[_gmm_y].astype(float).values
                        _xv = _gd[_gmm_x].astype(float).values if _gmm_x else np.zeros((len(_gd), 0))
                        _idv = _gd.index.get_level_values(0).values
                        _tv = _gd.index.get_level_values(1).values
                        _beta, _se, _dg = _ab_gmm(_yv, _xv, _idv, _tv, _arl=1, _gmax=int(_gmm_gmax), _sys=_gmm_sys)
                        _labels = [_gmm_y + " AR(1)"] + list(_gmm_x)
                        _rows = []
                        for _i, _lab in enumerate(_labels):
                            _star = "***" if _dg and _se[_i] != 0 and abs(_beta[_i] / _se[_i]) > 2.58 else ("**" if _se[_i] != 0 and abs(_beta[_i] / _se[_i]) > 1.96 else ("*" if _se[_i] != 0 and abs(_beta[_i] / _se[_i]) > 1.645 else ""))
                            _rows.append({"变量": _lab, "系数": f"{_beta[_i]:.4f}{_star}", "标准误": f"{_se[_i]:.4f}"})
                        _disp = pd.DataFrame(_rows)
                        st.subheader("📊 系统 GMM 系数")
                        _show_table(_disp, "sysgmm_coef.xlsx", "GMM")
                        _diag = pd.DataFrame([
                            {"检验": "AR(1) 自相关", "统计量": f"{_dg['ar1']:.4f}", "p值": f"{_dg['ar1_p']:.4f}（应显著）"},
                            {"检验": "AR(2) 自相关", "统计量": f"{_dg['ar2']:.4f}", "p值": f"{_dg['ar2_p']:.4f}（应不显著）"},
                            {"检验": "Hansen J 过度识别", "统计量": f"{_dg['hansen']:.4f}", "p值": f"{_dg['hansen_p']:.4f}（应>0.05）"},
                            {"检验": "观测数", "统计量": str(_dg["nobs"]), "p值": ""},
                            {"检验": "工具变量数", "统计量": str(_dg["n_instr"]), "p值": ""},
                        ])
                        st.subheader("🩺 诊断检验")
                        _show_table(_diag, "sysgmm_diag.xlsx", "Diag")
                        st.info("差分 GMM 应 AR(1) 显著、AR(2) 不显著；Hansen J p>0.05 表明工具变量整体有效。系统 GMM 可降低有限样本偏差。")
                except Exception as _e:
                    st.error(f"系统 GMM 失败：{_e}"); st.code(traceback.format_exc())

    # ===================== Heckman 两步法 =====================
    with _tab_x4:
        st.caption("Heckman 两步法纠正样本选择性偏差：第一步 Probit 选择方程估计选择概率，计算逆米尔斯比(IMR)；第二步在结果方程中加入 IMR，若 IMR 系数(λ)显著则说明存在选择性偏差。")
        _hk_y = st.selectbox("结果变量 Y（连续）", options=_num_cols, key="hk_y")
        _hk_d = st.selectbox("选择变量 D（0/1，是否进入样本）", options=_num_cols, key="hk_d")
        _hk_x = st.multiselect("结果方程变量 X", options=[c for c in _num_cols if c not in [_hk_y, _hk_d]], key="hk_x")
        _hk_z = st.multiselect("选择方程变量 Z（建议含至少一个排他变量）", options=[c for c in _num_cols if c not in [_hk_y]], default=[c for c in _num_cols if c not in [_hk_y]][:min(3, len(_num_cols) - 1)], key="hk_z")
        if _hk_y and _hk_d and _hk_z:
            if st.button("🚀 运行 Heckman 两步法", type="primary", key="hk_btn"):
                try:
                    _dd = _df_panel[[_hk_y, _hk_d] + _hk_x + _hk_z].dropna()
                    _D = _dd[_hk_d].astype(float)
                    if set(_D.unique()).issubset({0.0, 1.0}) and _D.nunique() == 2:
                        _Z = sm.add_constant(_dd[_hk_z].astype(float))
                        _sel = sm.Probit(_D, _Z).fit(disp=0)
                        _phi = norm.pdf(_Z @ _sel.params)
                        _Phi = norm.cdf(_Z @ _sel.params)
                        _imr = np.where(_D.values == 1, _phi / _Phi, -_phi / (1 - _Phi))
                        _out = _dd[[_hk_y] + _hk_x].astype(float).copy()
                        _out["IMR"] = _imr
                        _Xo = sm.add_constant(_out[[c for c in _out.columns if c != _hk_y]])
                        _res = sm.OLS(_out[_hk_y], _Xo).fit(cov_type="HC1")
                        _rows = []
                        for _n in _Xo.columns:
                            _rows.append({"变量": _n, "系数": _fmt_coef(_res.params[_n], _res.bse[_n], _res.pvalues[_n]), "标准误": f"{_res.bse[_n]:.4f}", "P>|t|": f"{_res.pvalues[_n]:.4f}"})
                        _hdf = pd.DataFrame(_rows)
                        st.subheader("📊 Heckman 第二步结果方程（含 IMR）")
                        _show_table(_hdf, "heckman_step2.xlsx", "Heckman")
                        _lam = _res.params.get("IMR", np.nan)
                        _lam_p = _res.pvalues.get("IMR", np.nan)
                        if _lam_p < 0.05:
                            st.success(f"✅ IMR(λ) 系数 = {_lam:.4f}，p = {_lam_p:.4f} < 0.05，存在显著选择性偏差，Heckman 修正有效。")
                        else:
                            st.info(f"IMR(λ) 系数 = {_lam:.4f}，p = {_lam_p:.4f} ≥ 0.05，未检测到显著选择性偏差（原 OLS 可能已无偏）。")
                    else:
                        st.warning("选择变量 D 必须是 0/1 二值变量。")
                except Exception as _e:
                    st.error(f"Heckman 失败：{_e}"); st.code(traceback.format_exc())

    # ===================== DDD 三重差分 =====================
    with _tab_x5:
        st.caption("三重差分(DDD)在 DID 基础上引入第二重分组，通过三重交互项识别政策效应，可排除由该第二分组维度随时间变化带来的混杂。核心关注 treat×post×group2 的系数。")
        _ddd_y = st.selectbox("结果变量 Y", options=_num_cols, key="ddd_y")
        _ddd_t = st.selectbox("处理变量 treat（0/1）", options=_num_cols, key="ddd_t")
        _ddd_g = st.selectbox("第二分组 group2（0/1）", options=_num_cols, key="ddd_g")
        _ddd_p = st.selectbox("政策后 post（0/1）", options=_num_cols, key="ddd_p")
        _ddd_c = st.multiselect("控制变量", options=[c for c in _num_cols if c not in [_ddd_y, _ddd_t, _ddd_g, _ddd_p]], key="ddd_c")
        if _ddd_y and _ddd_t and _ddd_g and _ddd_p:
            if st.button("🚀 运行 DDD 三重差分", type="primary", key="ddd_btn"):
                try:
                    _dd = _df_panel[[_ddd_y, _ddd_t, _ddd_g, _ddd_p] + _ddd_c].dropna().astype(float)
                    _dd["T_P"] = _dd[_ddd_t] * _dd[_ddd_p]
                    _dd["T_G"] = _dd[_ddd_t] * _dd[_ddd_g]
                    _dd["P_G"] = _dd[_ddd_p] * _dd[_ddd_g]
                    _dd["T_P_G"] = _dd[_ddd_t] * _dd[_ddd_p] * _dd[_ddd_g]
                    _X = sm.add_constant(_dd[[_ddd_t, _ddd_p, _ddd_g, "T_P", "T_G", "P_G", "T_P_G"] + _ddd_c])
                    _m = sm.OLS(_dd[_ddd_y], _X).fit(cov_type="HC1")
                    _rows = []
                    for _n in _X.columns:
                        _rows.append({"变量": _n, "系数": _fmt_coef(_m.params[_n], _m.bse[_n], _m.pvalues[_n]), "标准误": f"{_m.bse[_n]:.4f}", "P>|t|": f"{_m.pvalues[_n]:.4f}"})
                    _ddf = pd.DataFrame(_rows)
                    st.subheader("📊 DDD 三重差分系数表")
                    _show_table(_ddf, "ddd_results.xlsx", "DDD")
                    _coef = _m.params.get("T_P_G", np.nan); _p = _m.pvalues.get("T_P_G", np.nan)
                    _star = "***" if _p < 0.01 else "**" if _p < 0.05 else "*" if _p < 0.1 else ""
                    st.success(f"✅ 三重交互项 T×P×G 系数 = {_coef:.4f}{_star}，p = {_p:.4f}（DDD 政策效应估计量）")
                except Exception as _e:
                    st.error(f"DDD 失败：{_e}"); st.code(traceback.format_exc())

    # ---------- 四个优先级 tab ----------
    _tab_p1, _tab_p2, _tab_p3, _tab_p4 = st.tabs([
        "🟥 第一优先级：核心计量模型",
        "🟧 第二优先级：准实验方法",
        "🟨 第三优先级：前沿扩展",
        "🟩 第四优先级：稳健性辅助",
    ])

    # ================================================================
    # 🟥 第一优先级：核心计量模型
    # ================================================================
    with _tab_p1:

        # ---------- 1. 工具变量法 IV/2SLS ----------
        with st.expander("1️⃣ 工具变量法 (IV/2SLS)", expanded=False):
            st.markdown(
                "**选择指导**：当你怀疑某个解释变量与误差项相关（如反向因果、遗漏变量），"
                "且能找到与该解释变量相关但与误差项无关的变量（工具变量）时使用。"
                "输出第一阶段 F（弱工具变量检验）、2SLS 第二阶段、DWH 检验、"
                "Hansen J 过度识别检验、OLS vs IV 对比表。"
            )
            _instr_cand = [
                c for c in _num_cols
                if c not in [_y_var] + _x_vars + _control_vars
            ]
            _instruments = st.multiselect(
                "工具变量 Z（外生，须与 X 不同）", options=_instr_cand, key="iv_z"
            )
            if st.button("▶️ 运行 IV/2SLS", key="run_iv"):
                if not _x_vars or not _instruments:
                    st.warning("请至少选择一个核心解释变量和一个工具变量。")
                else:
                    try:
                        _dp = _clean_panel([_y_var] + _x_vars + _control_vars + _instruments)
                        _y = _dp[_y_var]
                        _exog = sm.add_constant(_dp[_control_vars]) if _control_vars \
                            else pd.DataFrame({"const": 1.0}, index=_dp.index)
                        _endog = _dp[_x_vars]
                        _instr = _dp[_instruments]
                        _clu = pd.DataFrame({"entity": _dp.index.get_level_values(0)}, index=_dp.index)

                        _ols = PanelOLS(
                            _y, sm.add_constant(_dp[_x_vars + _control_vars]),
                            entity_effects=False, time_effects=False,
                        ).fit(cov_type="clustered", clusters=_clu)

                        _iv = IV2SLS(_y, _exog, _endog, _instr).fit(
                            cov_type="clustered", clusters=_clu
                        )

                        # 第一阶段弱工具变量 F
                        _fs_full = sm.OLS(
                            _dp[_x_vars[0]],
                            sm.add_constant(_dp[_control_vars + _instruments]),
                        ).fit()
                        _fs_res = sm.OLS(
                            _dp[_x_vars[0]],
                            sm.add_constant(_dp[_control_vars]) if _control_vars
                            else pd.DataFrame({"const": 1.0}, index=_dp.index),
                        ).fit()
                        _q = len(_instruments)
                        _n_fs = int(_fs_full.nobs)
                        _rss_r = ((_fs_res.resid) ** 2).sum()
                        _rss_u = ((_fs_full.resid) ** 2).sum()
                        _k_u = _fs_full.df_model + 1
                        _weak_f = (
                            ((_rss_r - _rss_u) / _q)
                            / (_rss_u / (_n_fs - _k_u))
                        )
                        _weak_p = 1 - stats.f.cdf(_weak_f, _q, _n_fs - _k_u)

                        # DWH 检验
                        try:
                            _common = [p for p in _iv.params.index if p in _ols.params.index]
                            _diff = _iv.params[_common].values - _ols.params[_common].values
                            _v = _iv.cov.loc[_common, _common].values - _ols.cov.loc[_common, _common].values
                            _dwh = float(_diff @ np.linalg.pinv(_v) @ _diff)
                            _dwh_p = 1 - stats.chi2.cdf(_dwh, len(_common))
                        except Exception:
                            _dwh = _dwh_p = np.nan

                        # Hansen J / Sargan（IV2SLS 用 sargan，IVGMM 用 j_stat）
                        try:
                            if hasattr(_iv, "sargan") and _iv.sargan is not None:
                                _j_stat = _iv.sargan.stat
                                _j_p = _iv.sargan.pval
                            elif hasattr(_iv, "j_stat") and _iv.j_stat is not None:
                                _j_stat = _iv.j_stat.stat
                                _j_p = _iv.j_stat.pval
                            else:
                                _j_stat = _j_p = np.nan
                        except Exception:
                            _j_stat = _j_p = np.nan

                        _names = ["OLS", "IV/2SLS"]
                        _rows = {}
                        for _v in _x_vars + _control_vars + ["const"]:
                            _rows[_v] = {"变量": _v}
                            for _mn in _names:
                                _m = _ols if _mn == "OLS" else _iv
                                if _v in _m.params:
                                    _rows[_v][_mn] = _fmt_coef(_m.params[_v], _get_se(_m, _v), _m.pvalues[_v])
                                else:
                                    _rows[_v][_mn] = ""
                        _rows["弱工具变量F"] = {"变量": "第一阶段 F (Stock-Yogo)", "OLS": "", "IV/2SLS": f"{_weak_f:.4f}(p={_weak_p:.4f})"}
                        _rows["DWH检验"] = {"变量": "DWH χ²", "OLS": "", "IV/2SLS": f"{_dwh:.4f}(p={_dwh_p:.4f})"}
                        _rows["Hansen J"] = {"变量": "Hansen J (过度识别)", "OLS": "", "IV/2SLS": f"{_j_stat:.4f}(p={_j_p:.4f})"}
                        _rows["观测数"] = {"变量": "观测数", "OLS": int(_ols.nobs), "IV/2SLS": int(_iv.nobs)}
                        _order = _x_vars + _control_vars + ["const"] + ["弱工具变量F", "DWH检验", "Hansen J", "观测数"]
                        _disp = pd.DataFrame([_rows[k] for k in _order])
                        _disp = _disp[["变量"] + _names]
                        st.markdown("##### OLS vs IV/2SLS 对比")
                        _show_table(_disp, "iv_2sls_results.xlsx", "IV2SLS")
                        st.info(
                            f"弱工具变量 F = {_weak_f:.2f}（>10 为强工具）；"
                            f"DWH p = {_dwh_p:.4f}（<0.05 说明 X 内生，应选 IV）；"
                            f"Hansen J p = {_j_p:.4f}（>0.10 说明工具外生有效）。"
                        )
                    except Exception as _e:
                        st.error(f"IV/2SLS 出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 2. 控制函数法 ----------
        with st.expander("2️⃣ 控制函数法 (Control Function)", expanded=False):
            st.markdown(
                "**选择指导**：当你有一个内生变量，但不想用传统 IV（或 IV 难以获得）时，"
                "可通过第一阶段回归得到残差，再将残差作为控制变量放入第二阶段，以吸收内生性偏误。"
                "适用于非线性模型或内生变量为离散变量时。"
            )
            _endo_cf = st.selectbox("内生解释变量（从 X 中选一个）", options=_x_vars, key="cf_endo")
            _cf_instr = st.multiselect("工具变量 Z（用于第一阶段）", options=_instr_cand, key="cf_z")
            if st.button("▶️ 运行控制函数法", key="run_cf"):
                if not _endo_cf or not _cf_instr:
                    st.warning("请选择内生变量和至少一个工具变量。")
                else:
                    try:
                        _dp = _clean_panel([_y_var] + _x_vars + _control_vars + _cf_instr)
                        _fs = sm.OLS(_dp[_endo_cf], sm.add_constant(_dp[_control_vars + _cf_instr])).fit()
                        _resid = _fs.resid.rename("cf_resid")
                        _dp2 = _dp.copy()
                        _dp2["cf_resid"] = _resid
                        _exog2 = sm.add_constant(_dp2[_x_vars + _control_vars + ["cf_resid"]])
                        _ols_2nd = PanelOLS(_dp2[_y_var], _exog2, entity_effects=False).fit(cov_type="clustered", clusters=_fe_clusters(_dp2))
                        _rows = {}
                        for _v in _x_vars + _control_vars + ["cf_resid", "const"]:
                            if _v in _ols_2nd.params:
                                _rows[_v] = {"变量": _v, "控制函数 (第二阶段)": _fmt_coef(_ols_2nd.params[_v], _get_se(_ols_2nd, _v), _ols_2nd.pvalues[_v])}
                        _rows["第一阶段F"] = {"变量": "第一阶段 F (工具联合)", "控制函数 (第二阶段)": f"{_fs.fvalue:.4f}(p={_fs.f_pvalue:.4f})"}
                        _rows["观测数"] = {"变量": "观测数", "控制函数 (第二阶段)": int(_ols_2nd.nobs)}
                        _order = _x_vars + _control_vars + ["cf_resid", "const"] + ["第一阶段F", "观测数"]
                        _disp = pd.DataFrame([_rows[k] for k in _order])
                        _disp = _disp[["变量", "控制函数 (第二阶段)"]]
                        st.markdown("##### 控制函数法结果")
                        _show_table(_disp, "control_function.xlsx", "CF")
                        _rp = _ols_2nd.pvalues.get("cf_resid", np.nan)
                        st.info(f"残差项 cf_resid 的 p = {_rp:.4f}（<0.05 说明 {_endo_cf} 内生，控制函数法有效）。")
                    except Exception as _e:
                        st.error(f"控制函数法出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 3. 系统/差分 GMM ----------
        with st.expander("3️⃣ 系统/差分 GMM（动态面板）", expanded=False):
            st.markdown(
                "**选择指导**：当模型包含被解释变量的滞后项作为解释变量（动态面板），"
                "且存在个体固定效应和内生性时使用。系统 GMM 比差分 GMM 更有效率。"
                "输出 AR(1)、AR(2) 自相关检验和 Hansen 过度识别检验。"
            )
            _gmm_lags = st.slider("工具变量最大滞后期", 2, 5, 3, key="gmm_lag")
            if st.button("▶️ 运行差分 GMM", key="run_gmm"):
                if not _x_vars:
                    st.warning("请先选择核心解释变量。")
                else:
                    try:
                        _dp = _df_panel[[_y_var] + _x_vars + _control_vars].copy()
                        for _c in [_y_var] + _x_vars + _control_vars:
                            _dp[_c] = pd.to_numeric(_dp[_c], errors="coerce")
                        _dp["L_y"] = _dp.groupby(level=0)[_y_var].shift(1)
                        _dp["D_y"] = _dp.groupby(level=0)[_y_var].diff()
                        _dp["D_Ly"] = _dp.groupby(level=0)["L_y"].diff()
                        for _c in _x_vars:
                            _dp[f"D_{_c}"] = _dp.groupby(level=0)[_c].diff()
                        for _c in _control_vars:
                            _dp[f"D_{_c}"] = _dp.groupby(level=0)[_c].diff()
                        for _L in range(2, _gmm_lags + 1):
                            _dp[f"y_L{_L}"] = _dp.groupby(level=0)[_y_var].shift(_L)
                            for _c in _x_vars:
                                _dp[f"{_c}_L{_L}"] = _dp.groupby(level=0)[_c].shift(_L)
                        _dp = _dp.dropna()
                        _d_endog = _dp[["D_Ly"] + [f"D_{_c}" for _c in _x_vars]]
                        _d_exog = sm.add_constant(_dp[[f"D_{_c}" for _c in _control_vars]]) if _control_vars else pd.DataFrame({"const": 1.0}, index=_dp.index)
                        _instr_cols = []
                        for _L in range(2, _gmm_lags + 1):
                            _instr_cols.append(f"y_L{_L}")
                            for _c in _x_vars:
                                _instr_cols.append(f"{_c}_L{_L}")
                        _d_instr = _dp[_instr_cols]
                        _clu = pd.DataFrame({"entity": _dp.index.get_level_values(0)}, index=_dp.index)
                        _gmm = IVGMM(_dp["D_y"], _d_exog, _d_endog, _d_instr).fit(cov_type="clustered", clusters=_clu)
                        _resid = _gmm.resids
                        _resid_df = _resid.to_frame("e") if hasattr(_resid, "to_frame") else pd.DataFrame({"e": _resid}, index=_dp.index)
                        _resid_df["e_L1"] = _resid_df.groupby(level=0)["e"].shift(1)
                        _resid_df["e_L2"] = _resid_df.groupby(level=0)["e"].shift(2)
                        _ar1 = sm.OLS(_resid_df["e"], sm.add_constant(_resid_df["e_L1"]), missing="drop").fit()
                        _ar2 = sm.OLS(_resid_df["e"], sm.add_constant(_resid_df["e_L2"]), missing="drop").fit()
                        try:
                            _j_stat = _gmm.j_stat.stat if _gmm.j_stat is not None else np.nan
                            _j_p = _gmm.j_stat.pval if _gmm.j_stat is not None else np.nan
                        except Exception:
                            _j_stat = _j_p = np.nan
                        _rows = {}
                        _vars_show = ["D_Ly"] + [f"D_{_c}" for _c in _x_vars] + [f"D_{_c}" for _c in _control_vars] + ["const"]
                        for _v in _vars_show:
                            if _v in _gmm.params:
                                _rows[_v] = {"变量": _v, "差分 GMM": _fmt_coef(_gmm.params[_v], _get_se(_gmm, _v), _gmm.pvalues[_v])}
                        _rows["AR(1)"] = {"变量": "AR(1) p值", "差分 GMM": f"{_ar1.pvalues.iloc[-1]:.4f}"}
                        _rows["AR(2)"] = {"变量": "AR(2) p值", "差分 GMM": f"{_ar2.pvalues.iloc[-1]:.4f}"}
                        _rows["Hansen J"] = {"变量": "Hansen J (过度识别)", "差分 GMM": f"{_j_stat:.4f}(p={_j_p:.4f})"}
                        _rows["观测数"] = {"变量": "观测数", "差分 GMM": int(_gmm.nobs)}
                        _order = _vars_show + ["AR(1)", "AR(2)", "Hansen J", "观测数"]
                        _disp = pd.DataFrame([_rows[k] for k in _order if k in _rows])
                        _disp = _disp[["变量", "差分 GMM"]]
                        st.markdown("##### 差分 GMM 结果（简化版）")
                        _show_table(_disp, "gmm_results.xlsx", "GMM")
                        st.info("AR(1) 通常显著；AR(2) p 应 >0.10（无二阶自相关）；Hansen J p 应 >0.10（工具整体外生）。")
                    except Exception as _e:
                        st.error(f"GMM 出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 4. 固定效应模型 FE ----------
        with st.expander("4️⃣ 固定效应模型 (FE)", expanded=False):
            st.markdown(
                "**选择指导**：当你认为存在不随时间变化的个体层面遗漏变量（如企业文化、管理水平）时使用。"
                "通过个体虚拟变量或组内变换消除这些固定效应，是处理遗漏变量内生性的基础方法。"
                "输出 F 检验（混合 OLS vs FE）和 Hausman 检验（FE vs RE）。"
            )
            _fe_type = st.radio("固定效应类型", ["仅个体固定效应", "双向固定效应", "仅时间固定效应"], key="fe_type")
            if st.button("▶️ 运行固定效应模型", key="run_fe"):
                if not _x_vars:
                    st.warning("请先选择核心解释变量。")
                else:
                    try:
                        _dp = _clean_panel([_y_var] + _x_vars + _control_vars)
                        _y = _dp[_y_var]
                        _use_entity = "个体" in _fe_type or "双向" in _fe_type
                        _use_time = "时间" in _fe_type or "双向" in _fe_type
                        _clu = pd.DataFrame({"entity": _dp.index.get_level_values(0)}, index=_dp.index)
                        # 混合 OLS（含常数项）
                        _pooled = PanelOLS(_y, sm.add_constant(_dp[_x_vars + _control_vars]), entity_effects=False, time_effects=False).fit(cov_type="clustered", clusters=_clu)
                        # FE（常数项 / 不随 FE 维度变动的变量会被吸收，故先剔除再估计，不再 add_constant）
                        _exog_fe = _drop_absorbed(_dp, _dp[_x_vars + _control_vars], _use_entity, _use_time)
                        _fe = PanelOLS(_y, _exog_fe, entity_effects=_use_entity, time_effects=_use_time).fit(cov_type="clustered", clusters=_clu)
                        # RE（用于 Hausman）
                        _re = RandomEffects(_y, _exog).fit(cov_type="clustered", clusters=_clu)
                        try:
                            _common = [p for p in _fe.params.index if p in _re.params.index and p != "const"]
                            _diff = _fe.params[_common].values - _re.params[_common].values
                            _v = _fe.cov.loc[_common, _common].values - _re.cov.loc[_common, _common].values
                            _h = float(_diff @ np.linalg.pinv(_v) @ _diff)
                            _h_p = 1 - stats.chi2.cdf(_h, len(_common))
                        except Exception:
                            _h = _h_p = np.nan
                        # F 检验（ pooled vs FE: 用 _fe.f_pooled）
                        try:
                            _f_stat = _fe.f_pooled.stat if _fe.f_pooled is not None else np.nan
                            _f_p = _fe.f_pooled.pval if _fe.f_pooled is not None else np.nan
                        except Exception:
                            _f_stat = _f_p = np.nan
                        _names = ["混合 OLS", "固定效应 FE", "随机效应 RE"]
                        _rows = {}
                        for _v in _x_vars + _control_vars + ["const"]:
                            _rows[_v] = {"变量": _v}
                            for _mn in _names:
                                _m = {"混合 OLS": _pooled, "固定效应 FE": _fe, "随机效应 RE": _re}[_mn]
                                if _v in _m.params:
                                    _rows[_v][_mn] = _fmt_coef(_m.params[_v], _get_se(_m, _v), _m.pvalues[_v])
                                else:
                                    _rows[_v][_mn] = ""
                        _rows["F检验"] = {"变量": "F检验 (pooled vs FE)", "混合 OLS": "", "固定效应 FE": f"{_f_stat:.4f}(p={_f_p:.4f})", "随机效应 RE": ""}
                        _rows["Hausman"] = {"变量": "Hausman χ² (FE vs RE)", "混合 OLS": "", "固定效应 FE": f"{_h:.4f}(p={_h_p:.4f})", "随机效应 RE": ""}
                        _rows["观测数"] = {"变量": "观测数", "混合 OLS": int(_pooled.nobs), "固定效应 FE": int(_fe.nobs), "随机效应 RE": int(_re.nobs)}
                        _order = _x_vars + _control_vars + ["const"] + ["F检验", "Hausman", "观测数"]
                        _disp = pd.DataFrame([_rows[k] for k in _order])
                        _disp = _disp[["变量"] + _names]
                        st.markdown("##### 混合 OLS vs FE vs RE 对比")
                        _show_table(_disp, "fe_results.xlsx", "FE")
                        st.info(f"F 检验 p = {_f_p:.4f}（<0.05 说明 FE 优于混合 OLS）；Hausman p = {_h_p:.4f}（<0.05 说明 FE 优于 RE）。")
                    except Exception as _e:
                        st.error(f"固定效应模型出错：{_e}")
                        st.code(traceback.format_exc())

    # ================================================================
    # 🟧 第二优先级：准实验方法
    # ================================================================
    with _tab_p2:

        # ---------- 5. 双重差分 DID + 事件研究（已迁移至独立页面「7. DID + 事件研究法」） ----------

        # ---------- 6. 断点回归 RDD（已迁移至独立页面「8. RDD」） ----------

        # ---------- 7. 合成控制法 ----------
        with st.expander("7️⃣ 合成控制法 (Synthetic Control)", expanded=False):
            st.markdown(
                "**选择指导**：当只有一个或少数几个处理单元，需要构造一个“反事实”对照组时使用。"
                "通过对未受处理单元的加权组合来模拟处理单元在没有处理时的结果路径。"
                "适用于政策评估案例研究。"
            )
            _sc_treated = st.selectbox("处理单元（实体标识值）", options=sorted(_df_panel.index.get_level_values(0).unique().tolist()), key="sc_treated")
            _sc_treat_yr = st.number_input("处理起始年份", value=int(_df_panel.index.get_level_values(1).min()), step=1, key="sc_yr")
            if st.button("▶️ 运行合成控制法", key="run_sc"):
                if not _sc_treated:
                    st.warning("请选择处理单元。")
                else:
                    try:
                        _ents = sorted(_df_panel.index.get_level_values(0).unique())
                        _yrs = sorted(_df_panel.index.get_level_values(1).unique())
                        # 构造时间×实体矩阵
                        _mat = _df_panel[_y_var].unstack(level=0).reindex(index=_yrs, columns=_ents)
                        _mat = _mat.dropna(how="all")
                        _donors = [e for e in _ents if e != _sc_treated]
                        _pre_yrs = [y for y in _mat.index if y < _sc_treat_yr]
                        if not _pre_yrs or len(_donors) < 2:
                            st.warning("处理前年份或控制单元不足。")
                        else:
                            from scipy.optimize import minimize
                            _Y_pre = _mat.loc[_pre_yrs, _sc_treated].values
                            _X_pre = _mat.loc[_pre_yrs, _donors].values
                            _n_d = len(_donors)
                            def _loss(w):
                                return np.sum((_Y_pre - _X_pre @ w) ** 2)
                            _cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
                            _bnds = [(0.0, 1.0)] * _n_d
                            _res = minimize(_loss, np.ones(_n_d) / _n_d, method="SLSQP", bounds=_bnds, constraints=_cons)
                            _w = _res.x
                            _synth = _mat[_donors].values @ _w
                            _effect = _mat[_sc_treated].values - _synth
                            _sc_df = pd.DataFrame({"年份": _mat.index, "处理单元": _mat[_sc_treated].values, "合成对照": _synth, "处理效应": _effect})
                            st.markdown("##### 合成控制法结果")
                            _show_table(_sc_df, "synthetic_control.xlsx", "SC")
                            st.line_chart(_sc_df.set_index("年份")[["处理单元", "合成对照"]])
                            st.info(f"处理效应（处理后平均）= {_effect[_mat.index >= _sc_treat_yr].mean():.4f}。")
                    except Exception as _e:
                        st.error(f"合成控制法出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 8. 倾向得分匹配 PSM ----------
        with st.expander("8️⃣ 倾向得分匹配 (PSM)", expanded=False):
            st.markdown(
                "**选择指导**：当你想通过可观测协变量来匹配处理组和对照组，以减少选择偏差时使用。"
                "需指定处理变量（0/1）和匹配协变量。输出匹配前后平衡性检验、ATT 估计值。"
            )
            _psm_treat = st.selectbox("处理变量（0/1）", options=_all_cols, key="psm_t")
            _psm_covs = st.multiselect("匹配协变量", options=_num_cols, key="psm_covs")
            if st.button("▶️ 运行 PSM", key="run_psm"):
                if not _psm_treat or not _psm_covs:
                    st.warning("请选择处理变量与匹配协变量。")
                else:
                    try:
                        _dp = _df_panel.copy()
                        _dp[_psm_treat] = pd.to_numeric(_dp[_psm_treat], errors="coerce")
                        _use = [_psm_treat, _y_var] + _psm_covs
                        _dp = _dp[_use].apply(pd.to_numeric, errors="coerce").dropna()
                        _dp = _dp[_dp[_psm_treat].isin([0, 1])]
                        _Xl = sm.add_constant(_dp[_psm_covs])
                        _logit = sm.Logit(_dp[_psm_treat], _Xl).fit(disp=0)
                        _dp["ps"] = _logit.predict(_Xl)
                        _treated = _dp[_dp[_psm_treat] == 1].copy()
                        _control = _dp[_dp[_psm_treat] == 0].copy()
                        _matched_idx = []
                        _used = set()
                        for _, _tr in _treated.iterrows():
                            _avail = _control.loc[~_control.index.isin(_used)]
                            if _avail.empty:
                                continue
                            _dist = (_avail["ps"] - _tr["ps"]).abs()
                            _best = _dist.idxmin()
                            _matched_idx.append(_best)
                            _used.add(_best)
                        _matched_ctrl = _control.loc[[i for i in _matched_idx if i in _control.index]]
                        _att = float(_treated[_y_var].mean() - _matched_ctrl[_y_var].mean())
                        _bal_rows = []
                        for _c in _psm_covs:
                            _bt = _treated[_c].mean()
                            _bc = _control[_c].mean()
                            _bc_m = _matched_ctrl[_c].mean()
                            _sd = (_treated[_c].std() + _control[_c].std()) / 2
                            _sd_m = (_treated[_c].std() + _matched_ctrl[_c].std()) / 2
                            _bal_rows.append({"协变量": _c, "匹配前标准化差": round(abs(_bt - _bc) / _sd, 3) if _sd else np.nan, "匹配后标准化差": round(abs(_bt - _bc_m) / _sd_m, 3) if _sd_m else np.nan})
                        _bal = pd.DataFrame(_bal_rows)
                        st.markdown("##### 平衡性检验（标准化均值差，<0.1 较好）")
                        _show_table(_bal, "psm_balance.xlsx", "PSM平衡")
                        st.success(f"ATT = {_att:.4f}（处理组相比匹配后对照组的效应）")
                    except Exception as _e:
                        st.error(f"PSM 出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 9. 赫克曼两步法 ----------
        with st.expander("9️⃣ 赫克曼两步法 (Heckman)", expanded=False):
            st.markdown(
                "**选择指导**：当样本存在选择性偏差（例如只观察到工作人群的工资，但就业决策是内生的）时使用。"
                "需指定选择方程（至少包含一个排他性变量）和主方程。"
                "输出第一阶段 Probit 回归、逆米尔斯比率、第二阶段校正回归。"
            )
            _sel_dep = st.selectbox("选择方程被解释变量（0/1，是否被观测到）", options=_all_cols, key="heck_sel")
            _excl = st.selectbox("排他性变量（仅出现在选择方程）", options=_num_cols, key="heck_excl")
            if st.button("▶️ 运行 Heckman 两步法", key="run_heck"):
                if not _sel_dep or not _excl or not _y_var:
                    st.warning("请选择选择方程因变量、排他性变量与 Y。")
                else:
                    try:
                        _dp = _df_panel.copy()
                        _dp[_sel_dep] = pd.to_numeric(_dp[_sel_dep], errors="coerce")
                        _dp = _dp[[_sel_dep, _excl, _y_var] + _x_vars + _control_vars]
                        _dp = _dp.apply(pd.to_numeric, errors="coerce").dropna()
                        _dp = _dp[_dp[_sel_dep].isin([0, 1])]
                        _Xs = sm.add_constant(_dp[[_excl] + _control_vars])
                        _probit = sm.Probit(_dp[_sel_dep], _Xs).fit(disp=0)
                        _xb = _probit.predict(_Xs)
                        _imr = stats.norm.pdf(_xb) / stats.norm.cdf(_xb)
                        _dp["__imr__"] = _imr
                        _sub = _dp[_dp[_sel_dep] == 1].copy()
                        _Xo = sm.add_constant(_sub[_x_vars + _control_vars + ["__imr__"]])
                        _ols2 = sm.OLS(_sub[_y_var], _Xo).fit(cov_type="HC1")
                        _rows = []
                        for _v in _x_vars + _control_vars + ["__imr__", "const"]:
                            _label = _v.replace("__imr__", "逆米尔斯比率(IMR)")
                            _rows.append({"变量": _label, "Heckman 第二步": _fmt_coef(_ols2.params[_v], _ols2.bse[_v], _ols2.pvalues[_v])})
                        _rows.append({"变量": "观测数", "Heckman 第二步": int(_ols2.nobs)})
                        _disp = pd.DataFrame(_rows)
                        st.markdown("##### Heckman 两步法结果")
                        _show_table(_disp, "heckman_results.xlsx", "Heckman")
                        _imr_p = _ols2.pvalues["__imr__"]
                        st.info(f"IMR 的 p = {_imr_p:.4f}（<0.05 说明存在样本选择偏差，Heckman 修正有效）。")
                    except Exception as _e:
                        st.error(f"Heckman 出错：{_e}")
                        st.code(traceback.format_exc())

    # ================================================================
    # 🟨 第三优先级：前沿扩展
    # ================================================================
    with _tab_p3:

        # ---------- 10. 双重机器学习 ----------
        with st.expander("🔟 双重机器学习 (Double ML)", expanded=False):
            st.markdown(
                "**选择指导**：当你有高维控制变量（如文本特征、图像特征），且希望灵活控制非线性关系"
                "以估计处理效应时使用。利用机器学习模型（Lasso、随机森林）在第一阶段去偏，得到因果估计。"
                "需指定处理变量、结果变量和高维控制变量集合。"
            )
            _dml_treat = st.selectbox("处理变量 (D)", options=_num_cols, key="dml_d")
            _dml_ml = st.selectbox("机器学习模型", ["Lasso", "随机森林"], key="dml_ml")
            _n_folds = st.number_input("交叉验证折数", min_value=2, max_value=10, value=5, step=1, key="dml_folds")
            if st.button("▶️ 运行双重机器学习", key="run_dml"):
                if not _dml_treat or not _control_vars:
                    st.warning("请选择处理变量和控制变量。")
                else:
                    try:
                        from sklearn.linear_model import LassoCV
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.model_selection import KFold
                        _dp = _df_panel[[_y_var, _dml_treat] + _control_vars].copy()
                        _dp = _dp.apply(pd.to_numeric, errors="coerce").dropna()
                        _Yv = _dp[_y_var].values
                        _Dv = _dp[_dml_treat].values
                        _Xv = _dp[_control_vars].values
                        _n = len(_dp)
                        _Yhat = np.zeros(_n)
                        _Dhat = np.zeros(_n)
                        _kf = KFold(n_splits=int(_n_folds), shuffle=True, random_state=42)
                        for _tr_idx, _te_idx in _kf.split(_Xv):
                            if _dml_ml == "Lasso":
                                _my = LassoCV(cv=3, max_iter=5000).fit(_Xv[_tr_idx], _Yv[_tr_idx])
                                _md = LassoCV(cv=3, max_iter=5000).fit(_Xv[_tr_idx], _Dv[_tr_idx])
                            else:
                                _my = RandomForestRegressor(n_estimators=100, random_state=42).fit(_Xv[_tr_idx], _Yv[_tr_idx])
                                _md = RandomForestRegressor(n_estimators=100, random_state=42).fit(_Xv[_tr_idx], _Dv[_tr_idx])
                            _Yhat[_te_idx] = _my.predict(_Xv[_te_idx])
                            _Dhat[_te_idx] = _md.predict(_Xv[_te_idx])
                        _Yres = _Yv - _Yhat
                        _Dres = _Dv - _Dhat
                        # 第二阶段：Yres ~ Dres
                        _dml_ols = sm.OLS(_Yres, sm.add_constant(_Dres)).fit(cov_type="HC1")
                        _theta = _dml_ols.params[0]
                        _se = _dml_ols.bse[0]
                        _pval = _dml_ols.pvalues[0]
                        _ci_l = _theta - 1.96 * _se
                        _ci_u = _theta + 1.96 * _se
                        _disp = pd.DataFrame([
                            {"统计量": "处理效应 (θ)", "数值": f"{_theta:.4f}"},
                            {"统计量": "标准误", "数值": f"{_se:.4f}"},
                            {"统计量": "p 值", "数值": f"{_pval:.4f}"},
                            {"统计量": "95% CI 下限", "数值": f"{_ci_l:.4f}"},
                            {"统计量": "95% CI 上限", "数值": f"{_ci_u:.4f}"},
                            {"统计量": "观测数", "数值": str(int(_n))},
                        ])
                        st.markdown("##### 双重机器学习结果（部分线性模型）")
                        _show_table(_disp, "double_ml.xlsx", "DML")
                        st.info(f"处理效应 θ = {_theta:.4f}（p={_pval:.4f}）；θ 显著说明处理变量对 Y 有因果效应。")
                    except ImportError:
                        st.error("需要安装 scikit-learn：pip install scikit-learn")
                    except Exception as _e:
                        st.error(f"双重机器学习出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 11. 交互固定效应 IFE ----------
        with st.expander("1️⃣1️⃣ 交互固定效应 (IFE)", expanded=False):
            st.markdown(
                "**选择指导**：当普通双向固定效应不足以捕捉时变且个体差异的遗漏变量"
                "（如全球共同趋势对不同国家影响不同）时使用。通过因子结构建模未观测的异质性，"
                "适用于宏观/跨国面板数据。"
            )
            _n_factors = st.number_input("因子个数 r", min_value=1, max_value=5, value=1, step=1, key="ife_r")
            if st.button("▶️ 运行交互固定效应", key="run_ife"):
                if not _x_vars:
                    st.warning("请先选择核心解释变量。")
                else:
                    try:
                        _dp = _df_panel[[_y_var] + _x_vars + _control_vars].copy()
                        for _c in [_y_var] + _x_vars + _control_vars:
                            _dp[_c] = pd.to_numeric(_dp[_c], errors="coerce")
                        _dp = _dp.dropna()
                        _ent_counts = _dp.groupby(level=0).size()
                        _yr_counts = _dp.groupby(level=1).size()
                        _keep_e = _ent_counts[_ent_counts == _ent_counts.max()].index
                        _keep_y = _yr_counts[_yr_counts == _yr_counts.max()].index
                        _dp = _dp[_dp.index.get_level_values(0).isin(_keep_e) & _dp.index.get_level_values(1).isin(_keep_y)].copy()
                        if _dp.empty:
                            st.warning("无法构造平衡面板，请检查数据。")
                        else:
                            _ents_u = _dp.index.get_level_values(0).unique()
                            _yrs_u = _dp.index.get_level_values(1).unique()
                            _n, _t = len(_ents_u), len(_yrs_u)
                            _all_x = _x_vars + _control_vars
                            def _to_matrix(_s):
                                _m = _s.unstack(level=1)
                                _m = _m.reindex(index=_ents_u, columns=_yrs_u)
                                return _m.values
                            _Y = _to_matrix(_dp[_y_var])
                            _Xm = {c: _to_matrix(_dp[c]) for c in _all_x}
                            _K = len(_all_x)
                            _beta = np.zeros(_K)
                            _F = np.random.randn(_t, int(_n_factors))
                            for _it in range(50):
                                _pred = np.zeros_like(_Y)
                                for _k, _c in enumerate(_all_x):
                                    _pred += _Xm[_c] * _beta[_k]
                                _E = _Y - _pred
                                _U, _S, _Vt = np.linalg.svd(_E, full_matrices=False)
                                _F_new = _Vt[:int(_n_factors)].T
                                _Lam = _U[:, :int(_n_factors)] * _S[:int(_n_factors)]
                                _pred_factor = _Lam @ _F_new.T
                                _Ycf = _Y - _pred_factor
                                _A = np.column_stack([_Xm[c].ravel() for c in _all_x])
                                _beta_new, *_ = np.linalg.lstsq(_A, _Ycf.ravel(), rcond=None)
                                if np.max(np.abs(_beta_new - _beta)) < 1e-6:
                                    _beta = _beta_new
                                    _F = _F_new
                                    break
                                _beta = _beta_new
                                _F = _F_new
                            _Ycf = _Y - _Lam @ _F.T
                            _A = np.column_stack([_Xm[c].ravel() for c in _all_x])
                            _A2 = sm.add_constant(_A)
                            _ols = sm.OLS(_Ycf.ravel(), _A2).fit(cov_type="HC1")
                            _rows = []
                            for _k, _c in enumerate(_all_x):
                                _rows.append({"变量": _c, "交互固定效应": _fmt_coef(_ols.params[_k + 1], _ols.bse[_k + 1], _ols.pvalues[_k + 1])})
                            _rows.append({"变量": "常数项", "交互固定效应": _fmt_coef(_ols.params[0], _ols.bse[0], _ols.pvalues[0])})
                            _rows.append({"变量": "因子个数 r", "交互固定效应": int(_n_factors)})
                            _rows.append({"变量": "平衡面板观测数", "交互固定效应": int(_n * _t)})
                            _disp = pd.DataFrame(_rows)
                            st.markdown("##### 交互固定效应估计（迭代法）")
                            _show_table(_disp, "ife_results.xlsx", "IFE")
                            st.info("因子结构已吸收时变个体异质性；系数为去因子后的净效应。")
                    except Exception as _e:
                        st.error(f"交互固定效应出错：{_e}")
                        st.code(traceback.format_exc())

        # 注：空间计量（SLM/SEM/SDM）已独立为「10. 空间计量」页面，此处不再重复。

    # ================================================================
    # 🟩 第四优先级：稳健性辅助
    # ================================================================
    with _tab_p4:

        # ---------- 13. 敏感性分析 (Oster) ----------
        with st.expander("1️⃣3️⃣ 敏感性分析 (Oster 检验)", expanded=False):
            st.markdown(
                "**选择指导**：当你担心遗漏变量可能影响核心结论时，通过假设遗漏变量与处理变量的相关性强度，"
                "观察核心系数是否仍保持显著和符号稳定。输出敏感性参数（Oster δ、Imbens 边界）。"
            )
            _r_max = st.number_input("R² 上限 (R_max，默认 1.0)", value=1.0, step=0.05, key="oster_rmax")
            _delta_in = st.number_input("δ 假设值（默认 1）", value=1.0, step=0.1, key="oster_delta")
            if st.button("▶️ 运行敏感性分析", key="run_oster"):
                if not _x_vars:
                    st.warning("请先选择核心解释变量。")
                else:
                    try:
                        _dp = _clean_panel([_y_var] + _x_vars + _control_vars)
                        # 无控制变量回归
                        _ols0 = sm.OLS(_dp[_y_var], sm.add_constant(_dp[_x_vars])).fit()
                        # 有控制变量回归
                        _ols1 = sm.OLS(_dp[_y_var], sm.add_constant(_dp[_x_vars + _control_vars])).fit() if _control_vars else _ols0
                        _beta0 = _ols0.params[_x_vars[0]]
                        _beta1 = _ols1.params[_x_vars[0]]
                        _r0 = _ols0.rsquared
                        _r1 = _ols1.rsquared
                        # Oster δ: 使 β=0 所需的 δ
                        _numerator = _beta1 * (_r_max - _r1)
                        _denominator = _beta0 * (_r1 - _r0) if (_r1 - _r0) != 0 else np.nan
                        _delta = _numerator / _denominator if _denominator and not np.isnan(_denominator) else np.nan
                        # 识别集 [β1, β*]
                        _beta_star = _beta1 - _delta_in * (_beta0 - _beta1) if (_r1 - _r0) != 0 else _beta1
                        _disp = pd.DataFrame([
                            {"统计量": "β (无控制)", "数值": f"{_beta0:.4f}"},
                            {"统计量": "β (有控制)", "数值": f"{_beta1:.4f}"},
                            {"统计量": "R² (无控制)", "数值": f"{_r0:.4f}"},
                            {"统计量": "R² (有控制)", "数值": f"{_r1:.4f}"},
                            {"统计量": "R_max", "数值": f"{_r_max:.4f}"},
                            {"统计量": "Oster δ (使 β=0)", "数值": f"{_delta:.4f}" if not np.isnan(_delta) else "无法计算"},
                            {"统计量": f"β* (δ={_delta_in})", "数值": f"{_beta_star:.4f}"},
                            {"统计量": "识别集", "数值": f"[{_beta1:.4f}, {_beta_star:.4f}]"},
                        ])
                        st.markdown("##### Oster 敏感性分析结果")
                        _show_table(_disp, "oster_sensitivity.xlsx", "Oster")
                        st.info("|δ|>1 说明遗漏变量偏误需比观测变量更强才能消除效应，结论较稳健。识别集不含 0 则稳健。")
                    except Exception as _e:
                        st.error(f"敏感性分析出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 14. 安慰剂检验 ----------
        with st.expander("1️⃣4️⃣ 安慰剂检验 (Placebo)", expanded=False):
            st.markdown(
                "**选择指导**：通过随机打乱处理变量或虚构政策时间，观察真实估计系数是否落在随机分布的极端尾部，"
                "以排除巧合性结果。输出随机模拟系数分布图，标记真实系数位置。"
            )
            _n_iter = st.number_input("随机打乱次数", min_value=50, max_value=2000, value=300, step=50, key="placebo_n")
            if st.button("▶️ 运行安慰剂检验", key="run_placebo"):
                if not _x_vars:
                    st.warning("请先选择核心解释变量。")
                else:
                    try:
                        _dp = _clean_panel([_y_var] + _x_vars + _control_vars)
                        _Xx = sm.add_constant(_dp[_x_vars + _control_vars])
                        _true = sm.OLS(_dp[_y_var], _Xx).fit()
                        _true_coef = _true.params[_x_vars[0]]
                        _rng = np.random.default_rng(42)
                        _sim = np.empty(int(_n_iter))
                        _arr_y = _dp[_y_var].values
                        _arr_X = _Xx.values
                        _x_pos = list(_Xx.columns).index(_x_vars[0])
                        for _i in range(int(_n_iter)):
                            _perm = _rng.permutation(_arr_y)
                            _sim[_i] = sm.OLS(_perm, _arr_X).fit().params[_x_pos]
                        _p_emp = np.mean(np.abs(_sim) >= abs(_true_coef))
                        _q = np.percentile(_sim, [2.5, 50, 97.5])
                        _summary = pd.DataFrame({
                            "统计量": ["真实系数", "模拟均值", "模拟标准差", "2.5%分位", "中位数", "97.5%分位", "经验 p 值 (|sim|>=|真实|)"],
                            "数值": [f"{_true_coef:.4f}", f"{_sim.mean():.4f}", f"{_sim.std():.4f}", f"{_q[0]:.4f}", f"{_q[1]:.4f}", f"{_q[2]:.4f}", f"{_p_emp:.4f}"],
                        })
                        st.markdown("##### 安慰剂检验汇总")
                        _show_table(_summary, "placebo_test.xlsx", "Placebo")
                        _counts, _edges = np.histogram(_sim, bins=30)
                        _hist_df = pd.DataFrame({"区间": [f"{_edges[i]:.3f}~{_edges[i+1]:.3f}" for i in range(len(_counts))], "频数": _counts})
                        st.markdown("##### 系数分布直方图")
                        st.bar_chart(_hist_df.set_index("区间")["频数"])
                        st.info(f"真实系数 {_true_coef:.4f}；经验 p = {_p_emp:.4f}。若 p <0.05，真实结果显著区别于随机分布，结果稳健。")
                    except Exception as _e:
                        st.error(f"安慰剂检验出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 15. 滞后变量法 ----------
        with st.expander("1️⃣5️⃣ 滞后变量法", expanded=False):
            st.markdown(
                "**选择指导**：当你怀疑同期反向因果（Y 影响 X）时，使用 X 的滞后一期/二期作为解释变量，"
                "可缓解反向因果带来的内生性。输出滞后一期、滞后二期的回归结果，并与当期结果对比。"
            )
            if st.button("▶️ 运行滞后变量回归", key="run_lag"):
                if not _x_vars:
                    st.warning("请先选择核心解释变量。")
                else:
                    try:
                        _dp = _df_panel[[_y_var] + _x_vars + _control_vars].copy()
                        for _c in [_y_var] + _x_vars + _control_vars:
                            _dp[_c] = pd.to_numeric(_dp[_c], errors="coerce")
                        _lag_cols = []
                        for _x in _x_vars:
                            _dp[f"{_x}_L1"] = _dp.groupby(level=0)[_x].shift(1)
                            _dp[f"{_x}_L2"] = _dp.groupby(level=0)[_x].shift(2)
                            _lag_cols += [f"{_x}_L1", f"{_x}_L2"]
                        _dp = _dp.dropna()
                        _clu = pd.DataFrame({"entity": _dp.index.get_level_values(0)}, index=_dp.index)
                        # 双向 FE 下常数项会被吸收，自动剔除；不加入 const
                        _exog = _drop_absorbed(_dp, _dp[_lag_cols + _control_vars], True, True)
                        _lag = PanelOLS(_dp[_y_var], _exog, entity_effects=True, time_effects=True).fit(cov_type="clustered", clusters=_clu)
                        _rows = {}
                        for _v in _lag_cols + _control_vars + ["const"]:
                            if _v in _lag.params:
                                _rows[_v] = {"变量": _v, "滞后变量模型": _fmt_coef(_lag.params[_v], _get_se(_lag, _v), _lag.pvalues[_v])}
                        _rows["观测数"] = {"变量": "观测数", "滞后变量模型": int(_lag.nobs)}
                        _order = _lag_cols + _control_vars + ["const", "观测数"]
                        _disp = pd.DataFrame([_rows[k] for k in _order])
                        _disp = _disp[["变量", "滞后变量模型"]]
                        st.markdown("##### 滞后变量回归结果")
                        _show_table(_disp, "lagged_regressors.xlsx", "Lag")
                        st.info("用滞后 X 作为解释变量，若系数仍显著，说明结果对反向因果稳健。")
                    except Exception as _e:
                        st.error(f"滞后变量法出错：{_e}")
                        st.code(traceback.format_exc())

        # ---------- 16. 工具变量外生性检验 (Sargan/Hansen J) ----------
        with st.expander("1️⃣6️⃣ 工具变量外生性检验 (Sargan/Hansen J)", expanded=False):
            st.markdown(
                "**选择指导**：当工具变量个数多于内生变量个数时，该检验用于判断是否存在工具变量与误差项相关"
                "（即工具变量不满足外生性）。输出检验统计量和 p 值（p>0.05 表明工具变量外生）。"
            )
            _sargan_instr = st.multiselect("工具变量 Z（个数须 > 内生变量个数）", options=_instr_cand, key="sargan_z")
            _sargan_endo = st.multiselect("内生解释变量", options=_x_vars, key="sargan_endo")
            if st.button("▶️ 运行 Sargan/Hansen J 检验", key="run_sargan"):
                if not _sargan_instr or not _sargan_endo:
                    st.warning("请选择工具变量和内生解释变量。")
                elif len(_sargan_instr) <= len(_sargan_endo):
                    st.warning("过度识别检验要求工具变量个数 > 内生变量个数（即过度识别）。")
                else:
                    try:
                        _dp = _clean_panel([_y_var] + _sargan_endo + _control_vars + _sargan_instr)
                        _y = _dp[_y_var]
                        # 排除与工具变量重叠的控制变量，避免重复列名
                        _exog_ctrls = [c for c in _control_vars if c not in _sargan_instr and c not in _sargan_endo]
                        _exog = sm.add_constant(_dp[_exog_ctrls]) if _exog_ctrls else pd.DataFrame({"const": 1.0}, index=_dp.index)
                        _endog = _dp[_sargan_endo]
                        _instr = _dp[_sargan_instr]
                        _clu = pd.DataFrame({"entity": _dp.index.get_level_values(0)}, index=_dp.index)
                        _iv = IV2SLS(_y, _exog, _endog, _instr).fit(cov_type="clustered", clusters=_clu)
                        # Sargan 检验（IV2SLS 的过度识别检验属性）
                        try:
                            if hasattr(_iv, "sargan") and _iv.sargan is not None:
                                _j_stat = _iv.sargan.stat
                                _j_p = _iv.sargan.pval
                                _j_df = _iv.sargan.df
                            elif hasattr(_iv, "j_stat") and _iv.j_stat is not None:
                                _j_stat = _iv.j_stat.stat
                                _j_p = _iv.j_stat.pval
                                _j_df = _iv.j_stat.df
                            else:
                                _j_stat = _j_p = _j_df = np.nan
                        except Exception:
                            _j_stat = _j_p = _j_df = np.nan
                        _disp = pd.DataFrame([
                            {"检验": "过度识别条件", "结果": f"工具数 {len(_sargan_instr)} > 内生变量数 {len(_sargan_endo)}"},
                            {"检验": "过度识别统计量", "结果": f"{_j_stat:.4f}" if not np.isnan(_j_stat) else "N/A"},
                            {"检验": "过度识别 p 值", "结果": f"{_j_p:.4f}" if not np.isnan(_j_p) else "N/A"},
                            {"检验": "自由度", "结果": f"{_j_df}" if not np.isnan(_j_df) else "N/A"},
                            {"检验": "观测数", "结果": str(int(_iv.nobs))},
                        ])
                        st.markdown("##### Sargan / Hansen J 过度识别检验")
                        _show_table(_disp, "sargan_hansen.xlsx", "Sargan")
                        st.info("Hansen J / Sargan p > 0.05 表明工具变量整体外生（不能拒绝外生性假设）。")
                    except Exception as _e:
                        st.error(f"Sargan/Hansen J 检验出错：{_e}")
                        st.code(traceback.format_exc())

# ================================================================
#                    第八阶段：断点回归 RDD
# ================================================================
elif page == "8. RDD":
    st.header("8. 断点回归 (Regression Discontinuity Design)")
    st.markdown(
        "当处理分配完全由某个连续驱动变量是否超过某一阈值决定时使用。"
        "本页提供局部线性/二次回归、三角核/均匀核加权、断点处理效应估计、"
        "带宽敏感性分析与安慰剂检验。"
    )

    if st.session_state.get("merged_df") is None:
        st.warning("⚠️ 请先完成「1. 数据清洗」并生成 merged_df。")
        st.stop()

    _df = st.session_state.merged_df.copy()
    _col_id = st.session_state.get("col_id")
    _col_year = st.session_state.get("col_year")
    if not _col_id or not _col_year or _col_id not in _df.columns or _col_year not in _df.columns:
        st.warning("未正确设置个体 ID 或年份列，请在「1. 数据清洗」完成列映射。")
        st.stop()

    _df_panel = _df.set_index([_col_id, _col_year]).sort_index()
    _num_cols = [c for c in _df_panel.select_dtypes(include=[np.number]).columns if c not in [_col_id, _col_year]]
    _all_cols = _df_panel.columns.tolist()

    def _fmt_coef(param, se, pval):
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        return f"{param:.4f}{stars}\n({se:.4f})"

    def _show_table(display_df, fname, sheet="结果"):
        st.dataframe(display_df, use_container_width=True)
        _output = BytesIO()
        with pd.ExcelWriter(_output, engine="openpyxl") as writer:
            display_df.to_excel(writer, index=False, sheet_name=sheet)
        _output.seek(0)
        st.download_button(
            label="⬇️ 下载结果 (Excel)",
            data=_output,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{fname}",
        )

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        st.subheader("模型设定")
        _y_var = st.selectbox("结果变量 (Y)", options=_num_cols, key="rdd_y")
        _run_col = st.selectbox("驱动变量 (Running variable)", options=_num_cols, key="rdd_run")
        _cutoff = st.number_input("断点阈值 (Cutoff)", value=0.0, step=0.1, key="rdd_cutoff")
        _bw = st.number_input("带宽 (Bandwidth)", value=1.0, step=0.1, min_value=0.01, key="rdd_bw")
        _poly = st.selectbox("多项式阶数", options=["局部线性 (1阶)", "局部二次 (2阶)"], key="rdd_poly")
        _kernel = st.selectbox("核函数", options=["均匀核 (Uniform)", "三角核 (Triangular)"], key="rdd_kernel")
        _se_type = st.selectbox("标准误类型", options=["异方差稳健 (HC1)", "按个体聚类稳健"], key="rdd_se")
        _run_btn = st.button("▶️ 运行 RDD", key="run_rdd_main")

    if _run_btn:
        if not _y_var or not _run_col:
            st.warning("请选择结果变量与驱动变量。")
        else:
            try:
                _dp = _df_panel.copy()
                _dp[_run_col] = pd.to_numeric(_dp[_run_col], errors="coerce")
                _dp[_y_var] = pd.to_numeric(_dp[_y_var], errors="coerce")
                _dp = _dp.dropna(subset=[_run_col, _y_var])

                # 带宽截断
                _sub = _dp[(_dp[_run_col] >= _cutoff - _bw) & (_dp[_run_col] <= _cutoff + _bw)].copy()
                if _sub.empty:
                    st.warning("带宽内无样本，请增大带宽。")
                    st.stop()

                _sub["__treat__"] = (_sub[_run_col] >= _cutoff).astype(float)
                _sub["__run_c__"] = _sub[_run_col] - _cutoff
                _sub["__inter__"] = _sub["__treat__"] * _sub["__run_c__"]

                # 核权重
                _u = (_sub[_run_col] - _cutoff) / _bw
                if "三角核" in _kernel:
                    _sub["__w__"] = np.maximum(0.0, 1.0 - np.abs(_u))
                else:
                    _sub["__w__"] = 1.0

                # 构造 exog
                _base = ["__treat__", "__run_c__", "__inter__"]
                if "二次" in _poly:
                    _sub["__run_c2__"] = _sub["__run_c__"] ** 2
                    _sub["__inter2__"] = _sub["__treat__"] * _sub["__run_c2__"]
                    _base += ["__run_c2__", "__inter2__"]
                _exog = sm.add_constant(_sub[_base])

                # 估计：按核权重 WLS；按个体聚类需要把 entities 传入
                if "聚类" in _se_type and _col_id:
                    _clu = _sub.index.get_level_values(0).to_numpy()
                    _rdd = sm.WLS(
                        _sub[_y_var], _exog, weights=_sub["__w__"]
                    ).fit(cov_type="cluster", cov_kwds={"groups": _clu})
                else:
                    _rdd = sm.WLS(
                        _sub[_y_var], _exog, weights=_sub["__w__"]
                    ).fit(cov_type="HC1")

                _label_map = {
                    "__treat__": "处理(断点)",
                    "__run_c__": "驱动变量(中心化)",
                    "__inter__": "处理×驱动",
                    "__run_c2__": "驱动变量²",
                    "__inter2__": "处理×驱动²",
                    "const": "常数项",
                }
                _rows = []
                for _v in _exog.columns:
                    _rows.append({
                        "变量": _label_map.get(_v, _v),
                        "RDD 估计": _fmt_coef(_rdd.params[_v], _rdd.bse[_v], _rdd.pvalues[_v]),
                    })
                _rows.append({"变量": "观测数(带宽内)", "RDD 估计": int(_rdd.nobs)})
                _rows.append({"变量": "R²", "RDD 估计": f"{_rdd.rsquared:.4f}"})
                _rows.append({"变量": "带宽", "RDD 估计": f"{_bw:.4f}"})
                _rows.append({"变量": "核函数", "RDD 估计": _kernel.split("(")[1].replace(")", "")})
                _disp = pd.DataFrame(_rows)

                with col_right:
                    st.subheader("RDD 估计结果")
                    _show_table(_disp, "rdd_results.xlsx", "RDD")

                    # 图形：散点 + 两侧拟合线
                    st.markdown("##### 断点两侧拟合图")
                    _plot_df = _sub.copy()
                    _plot_df["__bin__"] = pd.cut(_plot_df[_run_col], bins=40)
                    _bin_means = _plot_df.groupby("__bin__", observed=False).agg({
                        _run_col: "mean",
                        _y_var: "mean",
                        "__w__": "sum",
                    }).dropna()
                    _bin_means = _bin_means[_bin_means["__w__"] > 0]

                    _grid_left = np.linspace(_cutoff - _bw, _cutoff, 50)
                    _grid_right = np.linspace(_cutoff, _cutoff + _bw, 50)
                    def _pred(grid, treat):
                        _tmp = pd.DataFrame({
                            "__treat__": treat,
                            "__run_c__": grid - _cutoff,
                            "__inter__": treat * (grid - _cutoff),
                        })
                        if "二次" in _poly:
                            _tmp["__run_c2__"] = _tmp["__run_c__"] ** 2
                            _tmp["__inter2__"] = _tmp["__treat__"] * _tmp["__run_c2__"]
                        _tmp = sm.add_constant(_tmp, has_constant="add")
                        return _tmp[_exog.columns] @ _rdd.params

                    _pred_left = _pred(_grid_left, 0.0)
                    _pred_right = _pred(_grid_right, 1.0)

                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.scatter(_bin_means[_run_col], _bin_means[_y_var], alpha=0.5, s=30, label="分箱均值", color="gray")
                        ax.plot(_grid_left, _pred_left, color="#1f77b4", lw=2, label="拟合线（左）")
                        ax.plot(_grid_right, _pred_right, color="#ff7f0e", lw=2, label="拟合线（右）")
                        ax.axvline(_cutoff, color="red", linestyle="--", alpha=0.6, label="断点")
                        ax.set_xlabel(_run_col)
                        ax.set_ylabel(_y_var)
                        ax.legend()
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                    except Exception as _plot_e:
                        st.caption(f"图形渲染失败：{_plot_e}")

                    # 带宽敏感性
                    st.markdown("##### 带宽敏感性分析")
                    _bw_list = [_bw * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
                    _bw_rows = []
                    for _b in _bw_list:
                        _sb = _dp[(_dp[_run_col] >= _cutoff - _b) & (_dp[_run_col] <= _cutoff + _b)].copy()
                        if len(_sb) < 10:
                            continue
                        _sb["__t__"] = (_sb[_run_col] >= _cutoff).astype(float)
                        _sb["__rc__"] = _sb[_run_col] - _cutoff
                        _sb["__it__"] = _sb["__t__"] * _sb["__rc__"]
                        _base_b = ["__t__", "__rc__", "__it__"]
                        if "二次" in _poly:
                            _sb["__rc2__"] = _sb["__rc__"] ** 2
                            _sb["__it2__"] = _sb["__t__"] * _sb["__rc2__"]
                            _base_b += ["__rc2__", "__it2__"]
                        _u_b = (_sb[_run_col] - _cutoff) / _b
                        _wb = np.maximum(0.0, 1.0 - np.abs(_u_b)) if "三角核" in _kernel else np.ones(len(_sb))
                        _eb = sm.add_constant(_sb[_base_b])
                        if "聚类" in _se_type and _col_id:
                            _clu_b = _sb.index.get_level_values(0).to_numpy()
                            _rm = sm.WLS(_sb[_y_var], _eb, weights=_wb).fit(cov_type="cluster", cov_kwds={"groups": _clu_b})
                        else:
                            _rm = sm.WLS(_sb[_y_var], _eb, weights=_wb).fit(cov_type="HC1")
                        _bw_rows.append({
                            "带宽": f"{_b:.3f}",
                            "处理效应": f"{_rm.params['__t__']:.4f}",
                            "p值": f"{_rm.pvalues['__t__']:.4f}",
                            "样本数": int(_rm.nobs),
                        })
                    if _bw_rows:
                        _bw_df = pd.DataFrame(_bw_rows)
                        st.dataframe(_bw_df, use_container_width=True)

                    st.info(
                        "处理(断点) 系数即断点处的局部平均处理效应 (LATE)。"
                        "若选择「三角核」，远离断点的观测权重会被下调；"
                        "「按个体聚类稳健」标准误适合企业/个体层面的面板数据。"
                    )
            except Exception as _e:
                st.error(f"RDD 出错：{_e}")
                st.code(traceback.format_exc())


# ================================================================
#                    第九章：机制分析（中介 / 调节 / 门槛）
# ================================================================
elif page == "9. 机制分析（中介/调节/门槛）":
    import statsmodels.api as sm
    from scipy.stats import norm
    from io import BytesIO

    st.header("机制分析：中介效应 / 调节效应 / 门槛回归")

    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先前往「1. 数据清洗」完成数据清洗！")
        st.stop()
    _df_raw = st.session_state.merged_df
    _col_id = st.session_state.get("col_id")
    _col_year = st.session_state.get("col_year")
    if _col_id is None or _col_year is None:
        st.error("请先在「数据清洗」页面选择「实体列」和「时间列」。")
        st.stop()
    _df_panel = _df_raw.dropna(subset=[_col_id, _col_year]).copy()
    _df_panel[_col_id] = _df_panel[_col_id].astype(str)
    try:
        _df_panel[_col_year] = _df_panel[_col_year].astype(int)
    except Exception:
        pass
    _df_panel = _df_panel.set_index([_col_id, _col_year]).sort_index()
    _num_cols = _df_panel.select_dtypes(include=[np.number]).columns.tolist()

    def _clean_panel(vars_used):
        need = list(dict.fromkeys([c for c in vars_used if c in _df_panel.columns]))
        return _df_panel[need].dropna()

    _mech = st.radio("机制分析方法", ["中介效应（Mediation）", "调节效应（Moderation）", "门槛回归（Hansen Threshold）"], key="mech_method")

    # ------------------- 1. 中介效应 -------------------
    if _mech == "中介效应（Mediation）":
        st.subheader("中介效应分析（Baron-Kenny + Bootstrap）")
        _dv = st.selectbox("结果变量 (Y)", _num_cols, key="med_y")
        _iv = st.selectbox("核心解释变量 (X)", [c for c in _num_cols if c != _dv], key="med_x")
        _mv = st.selectbox("中介变量 (M)", [c for c in _num_cols if c not in [_dv, _iv]], key="med_m")
        _ctr = st.multiselect("控制变量", [c for c in _num_cols if c not in [_dv, _iv, _mv]], key="med_c")
        _clu_on = st.checkbox("按个体聚类稳健标准误", value=True, key="med_clu")
        _nboot = st.number_input("Bootstrap 次数", 100, 5000, 1000, 100, key="med_boot")
        if st.button("▶️ 运行中介效应检验", key="run_med"):
            try:
                _d = _clean_panel([_dv, _iv, _mv] + _ctr)
                _clu = _d.index.get_level_values(0).to_numpy() if _clu_on else None

                def _ols(y, X):
                    _Xc = sm.add_constant(X)
                    if _clu is not None:
                        return sm.OLS(y, _Xc).fit(cov_type="cluster", cov_kwds={"groups": _clu})
                    return sm.OLS(y, _Xc).fit(cov_type="HC1")

                _m1 = _ols(_d[_dv], _d[[_iv] + _ctr])
                c, c_se, c_p = float(_m1.params[_iv]), float(_m1.bse[_iv]), float(_m1.pvalues[_iv])
                _m2 = _ols(_d[_mv], _d[[_iv] + _ctr])
                a, a_se, a_p = float(_m2.params[_iv]), float(_m2.bse[_iv]), float(_m2.pvalues[_iv])
                _m3 = _ols(_d[_dv], _d[[_iv, _mv] + _ctr])
                cp, cp_se, cp_p = float(_m3.params[_iv]), float(_m3.bse[_iv]), float(_m3.pvalues[_iv])
                b, b_se, b_p = float(_m3.params[_mv]), float(_m3.bse[_mv]), float(_m3.pvalues[_mv])
                indirect = a * b
                _sob_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
                _sob_z = indirect / _sob_se if _sob_se > 0 else np.nan
                _sob_p = 2 * (1 - norm.cdf(abs(_sob_z))) if not np.isnan(_sob_z) else np.nan
                # Bootstrap（对样本重抽样）估计间接效应 a*b 的 CI
                _rng = np.random.default_rng(20260815)
                _idx = np.arange(len(_d))
                _boot = []
                for _ in range(int(_nboot)):
                    _s = _rng.choice(_idx, size=len(_idx), replace=True)
                    _ds = _d.iloc[_s]
                    _clu_s = _clu[_s] if _clu is not None else None

                    def _ols_b(y, X):
                        _Xc = sm.add_constant(X)
                        if _clu_s is not None:
                            return sm.OLS(y, _Xc).fit(cov_type="cluster", cov_kwds={"groups": _clu_s})
                        return sm.OLS(y, _Xc).fit(cov_type="HC1")

                    _a = float(_ols_b(_ds[_mv], _ds[[_iv] + _ctr]).params[_iv])
                    _b2 = float(_ols_b(_ds[_dv], _ds[[_iv, _mv] + _ctr]).params[_mv])
                    _boot.append(_a * _b2)
                _boot = np.array(_boot)
                _ci_lo, _ci_hi = np.nanpercentile(_boot, [2.5, 97.5])
                _boot_se = _boot.std(ddof=1)
                _boot_p = 2 * (1 - norm.cdf(abs(indirect / _boot_se))) if _boot_se > 0 else np.nan
                _rows = [
                    {"路径": "X → Y（总效应 c）", "系数": _fmt_coef(c, c_se, c_p), "说明": "核心解释对结果的总影响"},
                    {"路径": "X → M（a）", "系数": _fmt_coef(a, a_se, a_p), "说明": "核心解释对中介的影响"},
                    {"路径": "M → Y（b，控制 X）", "系数": _fmt_coef(b, b_se, b_p), "说明": "中介对结果的影响"},
                    {"路径": "X → Y（直接效应 c'）", "系数": _fmt_coef(cp, cp_se, cp_p), "说明": "控制中介后的直接影响"},
                    {"路径": "间接效应 (a×b)", "系数": f"{indirect:.4f}", "说明": f"Bootstrap 95% CI [{_ci_lo:.4f}, {_ci_hi:.4f}]"},
                    {"路径": "Sobel 检验 z", "系数": f"{_sob_z:.4f}", "说明": f"p={_sob_p:.4f}"},
                ]
                _disp = pd.DataFrame(_rows)
                st.markdown("##### 中介效应结果")
                _show_table(_disp, "mediation_results.xlsx", "Mediation")
                st.info(
                    f"间接效应 a×b = {indirect:.4f}；Bootstrap 95% CI 不含 0 则中介效应显著。"
                    f"若 c' 不显著且 a×b 显著 → 完全中介；否则为部分中介。"
                )
            except Exception as _e:
                st.error(f"中介效应出错：{_e}")
                st.code(traceback.format_exc())

    # ------------------- 2. 调节效应 -------------------
    elif _mech == "调节效应（Moderation）":
        st.subheader("调节效应分析（交互项 + 简单斜率）")
        _dv = st.selectbox("结果变量 (Y)", _num_cols, key="mod_y")
        _iv = st.selectbox("核心解释变量 (X)", [c for c in _num_cols if c != _dv], key="mod_x")
        _wv = st.selectbox("调节变量 (W)", [c for c in _num_cols if c not in [_dv, _iv]], key="mod_w")
        _ctr = st.multiselect("控制变量", [c for c in _num_cols if c not in [_dv, _iv, _wv]], key="mod_c")
        _clu_on = st.checkbox("按个体聚类稳健标准误", value=True, key="mod_clu")
        if st.button("▶️ 运行调节效应检验", key="run_mod"):
            try:
                _d = _clean_panel([_dv, _iv, _wv] + _ctr)
                _clu = _d.index.get_level_values(0).to_numpy() if _clu_on else None
                _d = _d.assign(__XW__=_d[_iv] * _d[_wv])
                _Xc = sm.add_constant(_d[[_iv, _wv, "__XW__"] + _ctr])
                if _clu is not None:
                    _m = sm.OLS(_d[_dv], _Xc).fit(cov_type="cluster", cov_kwds={"groups": _clu})
                else:
                    _m = sm.OLS(_d[_dv], _Xc).fit(cov_type="HC1")
                _rows = [
                    {"变量": _iv, "系数": _fmt_coef(float(_m.params[_iv]), float(_m.bse[_iv]), float(_m.pvalues[_iv]))},
                    {"变量": _wv, "系数": _fmt_coef(float(_m.params[_wv]), float(_m.bse[_wv]), float(_m.pvalues[_wv]))},
                    {"变量": f"{_iv}×{_wv}（交互项）", "系数": _fmt_coef(float(_m.params["__XW__"]), float(_m.bse["__XW__"]), float(_m.pvalues["__XW__"]))},
                ]
                for _c in _ctr:
                    if _c in _m.params:
                        _rows.append({"变量": _c, "系数": _fmt_coef(float(_m.params[_c]), float(_m.bse[_c]), float(_m.pvalues[_c]))})
                _rows.append({"变量": "观测数", "系数": int(_m.nobs)})
                st.markdown("##### 调节效应回归结果")
                _show_table(pd.DataFrame(_rows), "moderation_results.xlsx", "Moderation")
                # 简单斜率
                _wmean, _wsd = float(_d[_wv].mean()), float(_d[_wv].std())
                _ss = []
                for _lbl, _wv_val in [("-1SD", _wmean - _wsd), ("均值", _wmean), ("+1SD", _wmean + _wsd)]:
                    _slope = float(_m.params[_iv]) + float(_m.params["__XW__"]) * _wv_val
                    _vX = _m.cov_params().loc[_iv, _iv]
                    _vXW = _m.cov_params().loc["__XW__", "__XW__"]
                    _cv = _m.cov_params().loc[_iv, "__XW__"]
                    _se_slope = np.sqrt(_vX + _wv_val**2 * _vXW + 2 * _wv_val * _cv)
                    _t = _slope / _se_slope
                    _p = 2 * (1 - norm.cdf(abs(_t)))
                    _ss.append({"W 水平": _lbl, "W 取值": f"{_wv_val:.4f}", "X 的简单斜率": f"{_slope:.4f}", "标准误": f"{_se_slope:.4f}", "p值": f"{_p:.4f}"})
                st.markdown("##### 简单斜率分析（X 对 Y 的边际效应）")
                _show_table(pd.DataFrame(_ss), "moderation_simple_slope.xlsx", "SimpleSlope")
                st.info("交互项系数显著（X 的效应随 W 变化）即存在调节效应；简单斜率给出不同 W 水平下 X 对 Y 的边际影响。")
            except Exception as _e:
                st.error(f"调节效应出错：{_e}")
                st.code(traceback.format_exc())

    # ------------------- 3. 门槛回归（Hansen 单门槛） -------------------
    else:
        st.subheader("门槛回归（Hansen 单门槛 + Bootstrap 显著性 + 95% CI）")
        _dv = st.selectbox("结果变量 (Y)", _num_cols, key="th_y")
        _iv = st.selectbox("核心解释变量 (X)", [c for c in _num_cols if c != _dv], key="th_x")
        _qv = st.selectbox("门槛变量 (Q)", [c for c in _num_cols if c not in [_dv, _iv]], key="th_q")
        _ctr = st.multiselect("控制变量", [c for c in _num_cols if c not in [_dv, _iv, _qv]], key="th_c")
        _clu_on = st.checkbox("按个体聚类稳健标准误", value=True, key="th_clu")
        _nboot = st.number_input("Bootstrap 次数（显著性）", 100, 2000, 500, 100, key="th_boot")
        if st.button("▶️ 运行门槛回归", key="run_th"):
            try:
                _d = _clean_panel([_dv, _iv, _qv] + _ctr).reset_index(drop=True)
                y = _d[_dv].to_numpy(float)
                X = _d[_iv].to_numpy(float)
                Q = _d[_qv].to_numpy(float)
                C = _d[_ctr].to_numpy(float) if _ctr else None
                n = len(y)

                def _ssr_unrestr(yy, g):
                    _Z = np.column_stack([np.ones(n), X * (Q <= g), X * (Q > g)] + ([C] if C is not None else []))
                    _b = np.linalg.lstsq(_Z, yy, rcond=None)[0]
                    _e = yy - _Z @ _b
                    return float(_e @ _e)

                def _ssr_pooled(yy):
                    _Z = np.column_stack([np.ones(n), X] + ([C] if C is not None else []))
                    _b = np.linalg.lstsq(_Z, yy, rcond=None)[0]
                    _e = yy - _Z @ _b
                    return float(_e @ _e)

                _qs = np.percentile(Q, np.linspace(10, 90, 100))
                _ssr_u_grid = [_ssr_unrestr(y, g) for g in _qs]
                _gi = int(np.argmin(_ssr_u_grid))
                _gamma = float(_qs[_gi])
                _ssr_u = float(_ssr_u_grid[_gi])
                _ssr_r = _ssr_pooled(y)
                _k = 3 + (C.shape[1] if C is not None else 0)
                _F = (_ssr_r - _ssr_u) / _ssr_u * (n - _k)
                # Bootstrap（对合并模型残差重抽样）
                _b0 = np.linalg.lstsq(np.column_stack([np.ones(n), X] + ([C] if C is not None else [])), y, rcond=None)[0]
                _Z0 = (np.column_stack([np.ones(n), X, C]) if C is not None else np.column_stack([np.ones(n), X]))
                _resid = y - _Z0 @ _b0
                _yhat = y - _resid
                _rng = np.random.default_rng(20260815)
                _Fb = []
                for _ in range(int(_nboot)):
                    _r = _rng.choice(_resid, size=n, replace=True)
                    _yb = _yhat + _r
                    _su = min(_ssr_unrestr(_yb, g) for g in _qs)
                    _Fb.append((_ssr_pooled(_yb) - _su) / _su * (n - _k))
                _Fb = np.array(_Fb)
                _p = float((_Fb >= _F).mean())
                # Hansen 95% CI：LR(γ) = n*(SSR_unrestr(γ) - SSR(γ̂)) / SSR(γ̂) ≤ 7.35
                _lr = np.array([n * (_ssr_unrestr(y, g) - _ssr_u) / _ssr_u for g in _qs])
                _ci_mask = _lr <= 7.35
                _ci_lo = float(min(_qs[_ci_mask])) if _ci_mask.any() else np.nan
                _ci_hi = float(max(_qs[_ci_mask])) if _ci_mask.any() else np.nan
                # 估计两区制系数
                _Z = (np.column_stack([np.ones(n), X * (Q <= _gamma), X * (Q > _gamma), C]) if C is not None else np.column_stack([np.ones(n), X * (Q <= _gamma), X * (Q > _gamma)]))
                _b = np.linalg.lstsq(_Z, y, rcond=None)[0]
                _beta1, _beta2 = float(_b[1]), float(_b[2])
                _rows = [
                    {"参数": "门槛值 γ", "估计值": f"{_gamma:.4f}", "说明": f"95% CI [{_ci_lo:.4f}, {_ci_hi:.4f}]"},
                    {"参数": "X (Q≤γ) 系数 β1", "估计值": f"{_beta1:.4f}", "说明": "低区制"},
                    {"参数": "X (Q>γ) 系数 β2", "估计值": f"{_beta2:.4f}", "说明": "高区制"},
                    {"参数": "F 统计量", "估计值": f"{_F:.4f}", "说明": "门槛效应存在性"},
                    {"参数": "Bootstrap p 值", "估计值": f"{_p:.4f}", "说明": "≤0.05 表明存在门槛"},
                ]
                if C is not None:
                    for _j, _c in enumerate(_ctr):
                        _rows.append({"参数": _c, "估计值": f"{float(_b[3 + _j]):.4f}", "说明": "控制变量"})
                st.markdown("##### 门槛回归结果")
                _show_table(pd.DataFrame(_rows), "threshold_results.xlsx", "Threshold")
                st.info(f"门槛值 γ={_gamma:.4f}：低区制 β1={_beta1:.4f}，高区制 β2={_beta2:.4f}；Bootstrap p={_p:.4f} 显著表明存在结构性突变。")
            except Exception as _e:
                st.error(f"门槛回归出错：{_e}")
                st.code(traceback.format_exc())


# ================================================================
#                    第十章：空间计量（SLM / SEM / SDM）
# ================================================================
elif page == "10. 空间计量（SLM/SEM/SDM）":
    import statsmodels.api as sm
    from scipy.optimize import minimize_scalar
    from io import BytesIO

    st.header("空间计量经济模型：SLM / SEM / SDM（极大似然估计）")

    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先前往「1. 数据清洗」完成数据清洗！")
        st.stop()
    _df_raw = st.session_state.merged_df
    _col_id = st.session_state.get("col_id")
    _col_year = st.session_state.get("col_year")
    if _col_id is None or _col_year is None:
        st.error("请先在「数据清洗」页面选择「实体列」和「时间列」。")
        st.stop()
    _df_panel = _df_raw.dropna(subset=[_col_id, _col_year]).copy()
    _df_panel[_col_id] = _df_panel[_col_id].astype(str)
    try:
        _df_panel[_col_year] = _df_panel[_col_year].astype(int)
    except Exception:
        pass
    _df_panel = _df_panel.set_index([_col_id, _col_year]).sort_index()
    _num_cols = _df_panel.select_dtypes(include=[np.number]).columns.tolist()

    _df_cross = _df_panel.groupby(level=0).mean(numeric_only=True)   # 面板 → 截面（按实体时间均值，仅数值列）
    _ents = _df_cross.index.tolist()
    _n = len(_ents)
    st.info(f"空间计量使用截面数据：对面板按「{_col_id}」求时间均值，得到 {_n} 个空间单元。")

    _y = st.selectbox("被解释变量 (Y)", _num_cols, key="sp_y")
    _xs = st.multiselect("解释变量 (X)", [c for c in _num_cols if c != _y], key="sp_x")
    _wtype = st.radio("空间权重构造方式", ["一阶邻接（按实体排序循环相邻）", "基于坐标的 K 近邻", "基于坐标的距离阈值"], key="sp_wt")
    _W = None
    if _wtype == "一阶邻接（按实体排序循环相邻）":
        st.caption("按实体排序，每个单元与前后相邻单元相连（循环），行标准化。适用于无坐标的面板截面。")
        _W = np.zeros((_n, _n))
        for _i in range(_n):
            _W[_i, (_i - 1) % _n] = 1.0
            _W[_i, (_i + 1) % _n] = 1.0
    else:
        _lon = st.selectbox("经度列", _num_cols, key="sp_lon")
        _lat = st.selectbox("纬度列", _num_cols, key="sp_lat")
        _coords = _df_cross[[_lon, _lat]].to_numpy(float)
        from scipy.spatial.distance import cdist
        _D = cdist(_coords, _coords)
        if _wtype == "基于坐标的 K 近邻":
            _k = int(st.number_input("近邻数 K", 1, _n - 1, min(4, _n - 1), 1, key="sp_k"))
            _W = np.zeros((_n, _n))
            for _i in range(_n):
                _nn = np.argsort(_D[_i])[1:_k + 1]
                _W[_i, _nn] = 1.0
        else:
            _band = float(st.number_input("距离阈值", min_value=0.0, value=float(np.percentile(_D, 50)), step=0.1, key="sp_band"))
            _W = (_D <= _band).astype(float)
            np.fill_diagonal(_W, 0.0)
    if _W is not None:
        _rs = _W.sum(axis=1, keepdims=True)
        _rs[_rs == 0] = 1.0
        _W = _W / _rs   # 行标准化
        st.success(f"空间权重矩阵已构造（{_n}×{_n}，行标准化）。")

    if _xs and _W is not None:
        _yv = _df_cross[_y].to_numpy(float)
        _Xv = _df_cross[_xs].to_numpy(float)
        _Xc = np.column_stack([np.ones(_n), _Xv])
        _bols = np.linalg.lstsq(_Xc, _yv, rcond=None)[0]
        _eols = _yv - _Xc @ _bols
        _s2_ols = _eols @ _eols / _n
        _ll_ols = -0.5 * _n * (1 + np.log(2 * np.pi) + np.log(_s2_ols))

        def _fit_slm(rho):
            _yl = _yv - rho * (_W @ _yv)
            _b = np.linalg.lstsq(_Xc, _yl, rcond=None)[0]
            _e = _yl - _Xc @ _b
            _s2 = _e @ _e / _n
            _sg = np.linalg.slogdet(np.eye(_n) - rho * _W)[1]
            return 0.5 * _n * np.log(_s2) - _sg, _b, _s2

        def _fit_sem(lam):
            _A = np.eye(_n) - lam * _W
            _ys = _A @ _yv
            _Xs = _A @ _Xc
            _b = np.linalg.lstsq(_Xs, _ys, rcond=None)[0]
            _e = _ys - _Xs @ _b
            _s2 = _e @ _e / _n
            _sg = np.linalg.slogdet(_A)[1]
            return 0.5 * _n * np.log(_s2) - _sg, _b, _s2

        _WX = _W @ _Xv
        _Z = np.column_stack([_Xc, _WX])

        def _fit_sdm(rho):
            _yl = _yv - rho * (_W @ _yv)
            _d = np.linalg.lstsq(_Z, _yl, rcond=None)[0]
            _e = _yl - _Z @ _d
            _s2 = _e @ _e / _n
            _sg = np.linalg.slogdet(np.eye(_n) - rho * _W)[1]
            return 0.5 * _n * np.log(_s2) - _sg, _d, _s2

        def _mle_1d(fitf):
            """一维参数网格搜索取全局最优（避免有界标量优化陷入局部极小）。"""
            _grid = np.linspace(-0.99, 0.99, 399)
            _objs = np.array([fitf(r)[0] for r in _grid])
            return float(_grid[int(np.argmin(_objs))])

        _r_slm = _mle_1d(_fit_slm)
        _r_sem = _mle_1d(_fit_sem)
        _r_sdm = _mle_1d(_fit_sdm)
        _rs_slm, _rs_sem, _rs_sdm = _fit_slm(_r_slm), _fit_sem(_r_sem), _fit_sdm(_r_sdm)
        _ll_slm, _ll_sem, _ll_sdm = -_rs_slm[0], -_rs_sem[0], -_rs_sdm[0]

        def _se_param(fitf, rhat):
            _h = 1e-4
            # fitf 返回三元组 (obj, b, s2)，这里只取目标函数值 [0] 做数值二阶导
            _f2 = (fitf(rhat + _h)[0] - 2 * fitf(rhat)[0] + fitf(rhat - _h)[0]) / _h**2
            return float(np.sqrt(1.0 / max(1e-12, _n * _f2))) if _f2 > 0 else np.nan

        _se_rho_slm = _se_param(_fit_slm, _r_slm)
        _se_lam_sem = _se_param(_fit_sem, _r_sem)
        _se_rho_sdm = _se_param(_fit_sdm, _r_sdm)

        _rows = []
        for _i, _x in enumerate(_xs):
            _rows.append({"模型": "SLM", "变量": _x, "系数": f"{_rs_slm[1][1 + _i]:.4f}"})
        _rows.append({"模型": "SLM", "变量": "ρ (空间滞后)", "系数": f"{_r_slm:.4f} (se={_se_rho_slm:.4f})"})
        _rows.append({"模型": "SLM", "变量": "log-likelihood", "系数": f"{_ll_slm:.4f}"})
        _rows.append({"模型": "SLM", "变量": "LR vs OLS (df=1)", "系数": f"{2 * (_ll_slm - _ll_ols):.4f}"})
        for _i, _x in enumerate(_xs):
            _rows.append({"模型": "SEM", "变量": _x, "系数": f"{_rs_sem[1][1 + _i]:.4f}"})
        _rows.append({"模型": "SEM", "变量": "λ (空间误差)", "系数": f"{_r_sem:.4f} (se={_se_lam_sem:.4f})"})
        _rows.append({"模型": "SEM", "变量": "log-likelihood", "系数": f"{_ll_sem:.4f}"})
        _rows.append({"模型": "SEM", "变量": "LR vs OLS (df=1)", "系数": f"{2 * (_ll_sem - _ll_ols):.4f}"})
        for _i, _x in enumerate(_xs):
            _rows.append({"模型": "SDM", "变量": f"{_x} (直接)", "系数": f"{_rs_sdm[1][1 + _i]:.4f}"})
            _rows.append({"模型": "SDM", "变量": f"W{_x} (间接)", "系数": f"{_rs_sdm[1][1 + len(_xs) + _i]:.4f}"})
        _rows.append({"模型": "SDM", "变量": "ρ (空间滞后)", "系数": f"{_r_sdm:.4f} (se={_se_rho_sdm:.4f})"})
        _rows.append({"模型": "SDM", "变量": "log-likelihood", "系数": f"{_ll_sdm:.4f}"})
        _rows.append({"模型": "SDM", "变量": f"LR vs OLS (df={len(_xs) + 1})", "系数": f"{2 * (_ll_sdm - _ll_ols):.4f}"})
        st.markdown("##### 空间计量估计结果（极大似然）")
        _show_table(pd.DataFrame(_rows), "spatial_results.xlsx", "Spatial")
        st.info(
            "SLM/SEM/SDM 采用极大似然估计；ρ/λ 接近 0 且不显著表明空间依赖弱；"
            "LR 检验显著（>临界值 3.84，df=1）支持空间模型优于 OLS。"
        )

# ================================================================
#                    第十一章：时间序列分析（ARIMA / VAR）
# ================================================================
elif page == "11. 时间序列分析":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import traceback
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.tsa.api import ARIMA, VAR as SMVAR
    from statsmodels.tsa.stattools import grangercausalitytests

    st.header("📉 第十一章：时间序列分析（ARIMA / VAR）")
    st.caption("适用于宏观/年度聚合数据或单个体的动态建模：ARIMA 单变量预测，VAR 多变量联系统与 Granger 因果、脉冲响应。")

    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先在「1. 数据清洗」完成数据清洗。")
        st.stop()
    df = st.session_state.merged_df
    year_col = st.session_state.get("col_year", None)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    _ts_tabs = st.tabs(["🔻 ARIMA 单变量建模与预测", "🔺 VAR 向量自回归"])
    with _ts_tabs[0]:
        _ar_y = st.selectbox("目标变量", options=num_cols, key="ar_y")
        _agg = st.checkbox("按年份均值聚合为年度序列", value=True, key="ar_agg") if year_col else False
        _ar_p = st.number_input("AR 阶数 p", 0, 5, 1, key="ar_p")
        _ar_d = st.number_input("差分阶数 d", 0, 2, 1, key="ar_d")
        _ar_q = st.number_input("MA 阶数 q", 0, 5, 1, key="ar_q")
        _ar_h = st.number_input("预测步长 h", 1, 20, 5, key="ar_h")
        if _ar_y:
            if st.button("🚀 估计 ARIMA 并预测", type="primary", key="ar_btn"):
                try:
                    _s = df[[_ar_y]].astype(float)
                    if _agg:
                        _s = _s.groupby(df[year_col]).mean()
                    _s = _s.dropna().iloc[:, 0]
                    _s.index = pd.to_numeric(_s.index, errors="coerce")
                    _s = _s[~_s.index.isna()]
                    _model = ARIMA(_s, order=(int(_ar_p), int(_ar_d), int(_ar_q))).fit()
                    _fc = _model.forecast(steps=int(_ar_h))
                    _ci = _model.get_forecast(int(_ar_h)).conf_int()
                    _rows = [{"变量": n, "系数": f"{_model.params[n]:.4f}", "标准误": f"{_model.bse[n]:.4f}", "P>|z|": f"{_model.pvalues[n]:.4f}"} for n in _model.params.index]
                    st.subheader("📊 ARIMA 系数")
                    _show_table(pd.DataFrame(_rows), "arima_coef.xlsx", "ARIMA")
                    st.caption(f"AIC = {_model.aic:.3f}，BIC = {_model.bic:.3f}")
                    _fut = list(range(len(_s), len(_s) + int(_ar_h)))
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(_s.values, label="观测")
                    ax.plot(_fut, _fc, label="预测", color="red")
                    ax.fill_between(_fut, _ci[:, 0], _ci[:, 1], color="red", alpha=0.2)
                    ax.legend(); ax.set_title("ARIMA 预测")
                    st.pyplot(fig)
                except Exception as _e:
                    st.error(f"ARIMA 失败：{_e}"); st.code(traceback.format_exc())

    with _ts_tabs[1]:
        _var_cols = st.multiselect("选择变量（≥2）", options=num_cols, default=num_cols[:min(3, len(num_cols))], key="var_cols")
        _var_agg = st.checkbox("按年份均值聚合", value=True, key="var_agg") if year_col else False
        _var_lag = st.number_input("滞后阶数", 1, 8, 2, key="var_lag")
        if len(_var_cols) >= 2:
            if st.button("🚀 估计 VAR", type="primary", key="var_btn"):
                try:
                    _v = df[_var_cols].astype(float)
                    if _var_agg:
                        _v = _v.groupby(df[year_col]).mean()
                    _v = _v.dropna().astype(float)
                    _v.index = pd.to_numeric(_v.index, errors="coerce")
                    _v = _v[~_v.index.isna()].astype(float)
                    _model = SMVAR(_v).fit(maxlags=int(_var_lag))
                    st.subheader("📊 VAR 系数（各方程）")
                    _rows = []
                    for i, _eq in enumerate(_model.names):
                        for _j, _nm in enumerate(_model.exog_names):
                            _rows.append({"方程": _eq, "解释项": _nm, "系数": f"{_model.params[i, _j]:.4f}"})
                    _show_table(pd.DataFrame(_rows), "var_coef.xlsx", "VAR")
                    _gr = []
                    for _c in _var_cols:
                        for _x in _var_cols:
                            if _x == _c:
                                continue
                            try:
                                _gc = grangercausalitytests(_v[[_c, _x]].values, maxlag=int(_var_lag))
                                _p = _gc[list(_gc.keys())[0]][0]["ssr_ftest"][1]
                                _gr.append({"被影响变量": _c, "Granger 原因": _x, "滞后阶": int(_var_lag), "p值": f"{_p:.4f}", "结论": "存在 Granger 因果" if _p < 0.05 else "无"})
                            except Exception:
                                pass
                    st.subheader("📊 Granger 因果检验")
                    _show_table(pd.DataFrame(_gr), "var_granger.xlsx", "Granger")
                    _irf = _model.irf(10)
                    _irf.plot(impulse=_var_cols[0], response=_var_cols, plot_stderr=False)
                    st.pyplot(plt.gcf())
                    st.info("脉冲响应：横轴为冲击后的期数，纵轴为各变量对首个变量一单位冲击的响应。")
                except Exception as _e:
                    st.error(f"VAR 失败：{_e}"); st.code(traceback.format_exc())


elif page == "12. 结构方程模型 SEM":
    from semopy import Model, calc_stats
    st.header("📐 第十二章：问卷结构方程模型（SEM / CFA）")
    st.caption("适用于量表/问卷数据：用 lavaan 风格语法定义测量模型（潜变量 =~ 指标）与结构模型（因变量 ~ 自变量），输出参数估计与拟合优度指标（CFI/TLI/RMSEA/SRMR/GFI/AGFI）。")
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先在「1. 数据清洗」完成数据清洗。")
        st.stop()
    df = st.session_state.merged_df
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    st.markdown("**模型语法（lavaan 风格）**：`潜变量 =~ 指标1 + 指标2 + ...` 定义测量模型；`结果变量 ~ 自变量` 定义结构路径；`潜变量 ~~ 潜变量` 设协方差。")
    _sem_template = """# 示例：两个潜变量 -> 一个结果（请按你的题项修改）
eta1 =~ q1 + q2 + q3
eta2 =~ q4 + q5 + q6
y ~ eta1 + eta2
"""
    _sem_desc = st.text_area("模型定义", value=_sem_template, height=180, key="sem_desc")
    _use_cols = st.multiselect("参与建模的数值变量（建议仅勾选模型涉及的指标/题项）", options=num_cols, default=num_cols, key="sem_cols")
    if st.button("🚀 拟合 SEM", type="primary", key="sem_btn"):
        try:
            _d = df[_use_cols].astype(float).dropna()
            if _d.shape[0] < 50:
                st.error("样本量过少（<50），SEM 拟合不可靠，请检查数据或变量选择。")
            else:
                with st.spinner("正在拟合结构方程模型..."):
                    mod = Model(_sem_desc)
                    mod.fit(_d)
                    ins = mod.inspect().rename(columns={
                        "lval": "左变量", "op": "算子", "rval": "右变量",
                        "Estimate": "估计值", "Std. Err": "标准误", "z-value": "z值", "p-value": "P值",
                    })
                    st.subheader("📊 路径系数 / 参数估计")
                    _show_table(ins, "sem_params.xlsx", "SEM参数")
                    s = calc_stats(mod)
                    def _g(k):
                        try:
                            return float(s[k].iloc[0])
                        except Exception:
                            return np.nan
                    _fit = pd.DataFrame({
                        "拟合指标": ["χ² (卡方)", "自由度 DoF", "CFI", "TLI", "RMSEA", "SRMR", "GFI", "AGFI", "AIC", "BIC", "LogLik"],
                        "数值": [_g("chi2"), _g("DoF"), _g("CFI"), _g("TLI"), _g("RMSEA"), _g("SRMR"), _g("GFI"), _g("AGFI"), _g("AIC"), _g("BIC"), _g("LogLik")],
                    })
                    st.subheader("📊 模型拟合优度")
                    _show_table(_fit, "sem_fit.xlsx", "拟合优度")
                    st.caption("判断标准：CFI>0.90、TLI>0.90、RMSEA<0.08、SRMR<0.08 表示拟合良好。若结构部分为空（纯 CFA）仍会给出 χ²/CFI/RMSEA 等绝对拟合指标。")
        except Exception as _e:
            st.error(f"SEM 拟合失败：{_e}（请检查语法中变量名是否与所选列一致）")
            st.code(traceback.format_exc())


elif page == "13. 双重机器学习 / 因果森林":
    from econml.dml import LinearDML, CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st.header("🌲 第十三章：双重机器学习 与 因果森林（DML / Causal Forest）")
    st.caption("适用于估计处理变量 T 对结果 Y 的因果效应且存在大量混杂 X 的场景。双重机器学习用 ML 估计干扰函数再做正交化回归得到 ATE；因果森林估计异质性处理效应（CATE，逐样本）。")
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先在「1. 数据清洗」完成数据清洗。")
        st.stop()
    df = st.session_state.merged_df
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    _y = st.selectbox("结果变量 Y", options=num_cols, key="dml_y")
    _t = st.selectbox("处理变量 T（可连续或 0/1）", options=[c for c in num_cols if c != _y], index=0, key="dml_t")
    _x = st.multiselect("协变量 X（混杂/控制变量，建议多选）", options=[c for c in num_cols if c not in (_y, _t)], default=[c for c in num_cols if c not in (_y, _t)][:min(5, len([c for c in num_cols if c not in (_y, _t)]))], key="dml_x")
    _method = st.radio("估计方法", ["线性双重机器学习 (LinearDML)", "因果森林 (CausalForestDML)"], key="dml_method")
    if st.button("🚀 估计处理效应", type="primary", key="dml_btn"):
        try:
            _cols = [_y, _t] + _x
            _d = df[_cols].dropna().astype(float)
            if _d.shape[0] < 100:
                st.error("样本量过少（<100），建议增大样本以保证 ML 干扰函数估计质量。")
            else:
                Y = _d[[_y]].values
                T = _d[[_t]].values
                X = _d[_x].values if _x else None
                _is_binary = (set(np.unique(T).astype(int))) <= {0, 1} and len(np.unique(T)) <= 2
                _mt = LogisticRegression(max_iter=200) if _is_binary else GradientBoostingRegressor(n_estimators=50)
                _my = GradientBoostingRegressor(n_estimators=50)
                if _method.startswith("线性"):
                    dml = LinearDML(model_y=_my, model_t=_mt, discrete_treatment=_is_binary)
                    dml.fit(Y, T, X=X)
                    ate = float(dml.effect(X).mean())
                    _lo, _hi = dml.effect_interval(X, alpha=0.05)
                    ate_lo, ate_hi = float(_lo.mean()), float(_hi.mean())
                    _rows = [{"处理效应 ATE": round(ate, 4), "95% 下限": round(ate_lo, 4), "95% 上限": round(ate_hi, 4),
                              "结论": "在 5% 水平显著" if (ate_lo > 0 or ate_hi < 0) else "不显著"}]
                    st.subheader("📊 平均处理效应 ATE")
                    _show_table(pd.DataFrame(_rows), "dml_ate.xlsx", "ATE")
                    st.caption(f"处理变量 T 对结果 Y 的平均因果效应约为 {ate:.4f}（控制 X 后）。显著判断依据 95% 置信区间不包含 0。")
                else:
                    cf = CausalForestDML(model_y=_my, model_t=_mt, discrete_treatment=_is_binary, n_estimators=200, inference=True)
                    cf.fit(Y, T, X=X)
                    cate = cf.effect(X)
                    ate = float(cate.mean())
                    _lo, _hi = cf.effect_interval(X, alpha=0.05)
                    cate_lo, cate_hi = float(_lo.mean()), float(_hi.mean())
                    _sum = [{"CATE 均值 (≈ATE)": round(ate, 4), "95% 下限(均值)": round(cate_lo, 4), "95% 上限(均值)": round(cate_hi, 4),
                             "CATE 中位数": round(float(np.median(cate)), 4), "CATE 标准差": round(float(cate.std()), 4),
                             "CATE 最小": round(float(cate.min()), 4), "CATE 最大": round(float(cate.max()), 4)}]
                    st.subheader("📊 异质性处理效应（CATE）汇总")
                    _show_table(pd.DataFrame(_sum), "cf_cate_summary.xlsx", "CATE汇总")
                    _unit = pd.DataFrame({"样本序号": range(len(cate)), "CATE": cate.flatten()}).sort_values("CATE", ascending=False)
                    st.subheader("📊 逐样本 CATE（按效应降序）")
                    _show_table(_unit, "cf_cate_unit.xlsx", "逐样本CATE")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(cate.flatten(), bins=30, color="#4C78A8", alpha=0.85)
                    ax.axvline(ate, color="red", linestyle="--", linewidth=1.5, label=f"ATE={ate:.3f}")
                    ax.set_title("CATE 分布"); ax.set_xlabel("处理效应"); ax.legend()
                    st.pyplot(fig)
                    st.caption("CATE 差异越大，说明处理效应在不同个体间越异质（存在调节效应）。")
        except Exception as _e:
            st.error(f"估计失败：{_e}")
            st.code(traceback.format_exc())


elif page == "14. 多层线性模型":
    from statsmodels.regression.mixed_linear_model import MixedLM
    st.header("🏗️ 第十四章：多层线性模型（混合效应 / 分层模型）")
    st.caption("适用于具有嵌套结构的数据（如学生-学校、员工-企业、个体-时间）。估计固定效应与随机效应方差，并支持随机斜率。ICC 衡量组间差异占比。")
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先在「1. 数据清洗」完成数据清洗。")
        st.stop()
    df = st.session_state.merged_df
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    _dv = st.selectbox("因变量", options=num_cols, key="ml_dv")
    _fe = st.multiselect("固定效应自变量", options=[c for c in num_cols if c != _dv], default=[c for c in num_cols if c != _dv][:min(3, len([c for c in num_cols if c != _dv]))], key="ml_fe")
    _grp = st.selectbox("分组变量（随机截距，如企业/地区/个体）", options=[c for c in df.columns if c != _dv], index=0, key="ml_grp")
    _re = st.multiselect("随机斜率变量（可选，须为数值列）", options=_fe, key="ml_re")
    if st.button("🚀 估计多层模型", type="primary", key="ml_btn"):
        try:
            if not _fe:
                st.error("请至少选择一个固定效应自变量。")
            else:
                _d = df[[_dv] + _fe + [_grp] + _re].dropna()
                endog = _d[_dv].astype(float)
                exog = sm.add_constant(_d[_fe].astype(float))
                groups = _d[_grp].astype(str)
                if _re:
                    model = MixedLM(endog, exog, groups=groups, exog_re=_d[_re].astype(float))
                else:
                    model = MixedLM(endog, exog, groups=groups)
                res = model.fit()
                _fe_rows = [{"变量": n, "估计值": round(float(v), 4), "标准误": round(float(res.bse[n]), 4), "P值": round(float(res.pvalues[n]), 4)}
                            for n, v in res.fe_params.items()]
                st.subheader("📊 固定效应")
                _show_table(pd.DataFrame(_fe_rows), "ml_fe.xlsx", "固定效应")
                # 随机效应方差组分
                _cr = res.cov_re
                _re_rows = []
                for i in range(_cr.shape[0]):
                    for j in range(_cr.shape[1]):
                        _re_rows.append({"随机效应": _cr.index[i], "×": _cr.columns[j], "协方差/方差": round(float(_cr.iloc[i, j]), 4)})
                st.subheader("📊 随机效应方差组分")
                _show_table(pd.DataFrame(_re_rows), "ml_re.xlsx", "随机效应")
                if not _re and _cr.shape[0] == 1:
                    _gv = float(_cr.iloc[0, 0]); _rv = float(res.scale)
                    _icc = _gv / (_gv + _rv) if (_gv + _rv) > 0 else np.nan
                    st.subheader("📊 组内相关系数 ICC")
                    _show_table(pd.DataFrame([{"随机截距方差": round(_gv, 4), "残差方差": round(_rv, 4), "ICC": round(_icc, 4)}]), "ml_icc.xlsx", "ICC")
                    st.caption(f"ICC = 组间方差 / (组间方差 + 残差方差) = {_icc:.4f}，表示因变量变异中由分组结构解释的比例。")
                else:
                    st.caption("已包含随机斜率，ICC 仅适用于纯随机截距模型，此处不计算。")
        except Exception as _e:
            st.error(f"多层模型拟合失败：{_e}")
            st.code(traceback.format_exc())


elif page == "15. P2 综合评价进阶（模糊/可变权/AHP）":
    st.header("🧮 第十五章：P2 综合评价进阶（模糊综合评价 / 可变权重 / AHP 分层合成）")
    st.caption("三类高级综合评价方法：模糊综合评价（FCE，隶属度聚合）、可变权重（惩罚/鼓励低值指标）、AHP 分层合成（成对比较矩阵→权重+一致性检验）。")
    if st.session_state.merged_df is None:
        st.warning("⚠️ 请先在「1. 数据清洗」完成数据清洗（AHP 可独立于数据使用）。")
        st.stop()
    df = st.session_state.merged_df
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    def _entropy_w(Xn):
        P = Xn / (Xn.sum(0) + 1e-12)
        P = np.clip(P, 1e-12, 1)
        k = 1.0 / np.log(max(Xn.shape[0], 2))
        e = -k * (P * np.log(P)).sum(0)
        d = 1 - e
        return d / (d.sum() + 1e-12)

    def _critic_w(Xn):
        std = Xn.std(0) + 1e-12
        corr = np.corrcoef(Xn.T)
        corr = np.nan_to_num(corr, nan=1.0)
        C = std * (1.0 + (1.0 - corr).sum(0))
        return C / (C.sum() + 1e-12)

    _p2_tabs = st.tabs(["🌫️ 模糊综合评价 (FCE)", "⚖️ 可变权重 (Variable Weight)", "🪜 AHP 分层合成法"])
    with _p2_tabs[0]:
        _f_cols = st.multiselect("评价指标（数值列）", options=num_cols, default=num_cols[:min(5, len(num_cols))], key="fce_cols")
        _f_wm = st.selectbox("权重方法", ["熵权法", "CRITIC", "等权重"], key="fce_wm")
        if st.button("🚀 模糊综合评价", type="primary", key="fce_btn"):
            try:
                if len(_f_cols) < 2:
                    st.error("请至少选择 2 个评价指标。")
                else:
                    X = df[_f_cols].astype(float).dropna().values
                    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)  # 正向化到 [0,1]
                    if _f_wm == "熵权法":
                        W = _entropy_w(Xn)
                    elif _f_wm == "CRITIC":
                        W = _critic_w(Xn)
                    else:
                        W = np.ones(len(_f_cols)) / len(_f_cols)
                    # 四级模糊划分：三角隶属，峰点 0.125/0.375/0.625/0.875，宽 0.25
                    peaks = np.array([0.125, 0.375, 0.625, 0.875])
                    R = np.zeros((Xn.shape[0], len(_f_cols), 4))
                    for s in range(Xn.shape[0]):
                        for i in range(len(_f_cols)):
                            u = Xn[s, i]
                            for k in range(4):
                                R[s, i, k] = max(0.0, 1.0 - abs(u - peaks[k]) / 0.25)
                            _rs = R[s, i, :].sum()
                            if _rs > 0:
                                R[s, i, :] = R[s, i, :] / _rs   # 归一化使每指标隶属度和=1
                    B = np.einsum("i,sik->sk", W, R)  # (n,4) 每样本对优/良/中/差的隶属度
                    grades = np.array([100, 75, 50, 25])
                    score = B @ grades  # 每样本综合得分 ∈ [0,100]
                    _out = pd.DataFrame({
                        "样本": list(range(X.shape[0])),
                        "模糊综合得分": np.round(score, 3),
                        "隶属-优": np.round(B[:, 0], 3), "隶属-良": np.round(B[:, 1], 3),
                        "隶属-中": np.round(B[:, 2], 3), "隶属-差": np.round(B[:, 3], 3),
                    }).sort_values("模糊综合得分", ascending=False)
                    st.subheader("📊 模糊综合评价结果")
                    _show_table(_out, "fce_result.xlsx", "FCE")
                    st.caption("采用 M(·,+) 加权聚合：隶属度经三角模糊划分（优/良/中/差），综合得分 = 隶属向量 · [100,75,50,25]。权重：" + _f_wm + "。")
            except Exception as _e:
                st.error(f"模糊综合评价失败：{_e}")
                st.code(traceback.format_exc())
    with _p2_tabs[1]:
        _v_cols = st.multiselect("评价指标（数值列）", options=num_cols, default=num_cols[:min(5, len(num_cols))], key="vw_cols")
        _v_wm = st.selectbox("基础权重方法", ["熵权法", "CRITIC", "等权重"], key="vw_wm")
        _v_type = st.radio("可变权重类型", ["惩罚型（低值指标被降权）", "鼓励型（高值指标被加权）"], key="vw_type")
        if st.button("🚀 计算可变权重得分", type="primary", key="vw_btn"):
            try:
                if len(_v_cols) < 2:
                    st.error("请至少选择 2 个评价指标。")
                else:
                    X = df[_v_cols].astype(float).dropna().values
                    Xn = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
                    if _v_wm == "熵权法":
                        base = _entropy_w(Xn)
                    elif _v_wm == "CRITIC":
                        base = _critic_w(Xn)
                    else:
                        base = np.ones(len(_v_cols)) / len(_v_cols)
                    if "惩罚" in _v_type:
                        numer = base * (1.0 - Xn)
                    else:
                        numer = base * Xn
                    vw = numer / (numer.sum(1, keepdims=True) + 1e-12)
                    fixed_score = (base * Xn).sum(1)
                    vw_score = (vw * Xn).sum(1)
                    _out = pd.DataFrame({
                        "样本": range(X.shape[0]),
                        "固定权重得分": np.round(fixed_score, 4),
                        "可变权重得分": np.round(vw_score, 4),
                        "得分变化": np.round(vw_score - fixed_score, 4),
                    })
                    for i, c in enumerate(_v_cols):
                        _out[f"变权_{c}"] = np.round(vw[:, i], 4)
                    _out = _out.sort_values("可变权重得分", ascending=False)
                    st.subheader("📊 可变权重评价结果")
                    _show_table(_out, "vw_result.xlsx", "可变权重")
                    st.caption("可变权重 = 基础权重 × 状态项 / 归一化。惩罚型状态项 = (1−x_i)，鼓励型 = x_i。得分变化反映低值/高值指标被重新加权后的影响。")
            except Exception as _e:
                st.error(f"可变权重失败：{_e}")
                st.code(traceback.format_exc())
    with _p2_tabs[2]:
        _n = int(st.number_input("判断矩阵阶数 n", min_value=2, max_value=8, value=4, step=1, key="ahp_n"))
        _labels = st.multiselect("指标/准则名称（可选，用于标注，需与阶数匹配）", options=num_cols, default=num_cols[:min(_n, len(num_cols))], key="ahp_labels")
        A = np.eye(_n)
        for i in range(_n):
            for j in range(i + 1, _n):
                val = st.number_input(f"A[{i+1},{j+1}]（{_labels[i] if i < len(_labels) else '指标'+str(i+1)} vs {_labels[j] if j < len(_labels) else '指标'+str(j+1)}）", min_value=1/9, max_value=9.0, value=1.0, step=0.5, key=f"ahp_{i}_{j}")
                A[i, j] = val
                A[j, i] = 1.0 / val
        if st.button("🚀 计算 AHP 权重", type="primary", key="ahp_btn"):
            try:
                ev, vec = np.linalg.eig(A)
                idx = int(np.argmax(ev.real))
                w = np.abs(vec[:, idx].real); w = w / w.sum()
                lam = float(ev[idx].real)
                CI = (lam - _n) / (_n - 1)
                RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
                CR = CI / RI[_n] if RI[_n] > 0 else 0.0
                _wname = _labels if len(_labels) == _n else [f"指标{i+1}" for i in range(_n)]
                _out = pd.DataFrame({"指标": _wname, "权重": np.round(w, 4)}).sort_values("权重", ascending=False)
                st.subheader("📊 AHP 权重")
                _show_table(_out, "ahp_weights.xlsx", "AHP权重")
                st.subheader("📊 一致性检验")
                _show_table(pd.DataFrame([{"λmax": round(lam, 4), "CI": round(CI, 4), "RI": RI[_n], "CR": round(CR, 4), "结论": "通过（CR<0.1）" if CR < 0.1 else "未通过，请调整判断矩阵"}]), "ahp_cr.xlsx", "一致性")
                st.caption("权重由判断矩阵最大特征根对应的特征向量归一化得到；CR<0.1 表示判断具有满意一致性。")
            except Exception as _e:
                st.error(f"AHP 计算失败：{_e}")
                st.code(traceback.format_exc())

