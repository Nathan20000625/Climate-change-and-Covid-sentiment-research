# -*- coding: utf-8 -*-
"""
气候变化发帖数量与情感对自变量的OLS回归
输入: climate_post_and_emotions.csv
9次回归: 分别以发帖量与8种情感频率为因变量，灾害/疫情/新闻等为自变量
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ============ 配置 ============
# 以脚本所在目录为基准，这样从任意位置运行都能找到同目录下的 CSV
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "climate_post_and_emotions.csv")
OUTPUT_SUMMARY_CSV = os.path.join(SCRIPT_DIR, "ols_regression_summary.csv")
OUTPUT_COEFFS_CSV = os.path.join(SCRIPT_DIR, "ols_coefficients.csv")
OUTPUT_KEY_CI_CSV = os.path.join(SCRIPT_DIR, "key_covid_effects_ci.csv")

# 因变量（共9个）
DEPENDENT_VARS = [
    "climate_post",
    "climate_joy_score_freq",
    "climate_sadness_score_freq",
    "climate_anger_score_freq",
    "climate_fear_score_freq",
    "climate_surprise_score_freq",
    "climate_disgust_score_freq",
    "climate_trust_score_freq",
    "climate_anticipation_score_freq",
]


# 自变量
INDEPENDENT_VARS = [
    "US_daily_covid_confirm",
    "debates",
    "climatenews",
    "WinterStorm",
    "Wildfire",
    "TropicalCyclone",
    "SevereStorm",
    "Flood",
    "Drought",
    "GovernmentResponseIndex_Average",
]

# 情感模型中需标准化的自变量（仅此 3 个；其余 debates、灾害等指标变量保留原始取值）
STANDARDIZE_VARS = [
    "US_daily_covid_confirm",
    "climatenews",
    "GovernmentResponseIndex_Average",
]

# 星期虚拟变量（周二到周日，周一作为基准组）
WEEKDAY_DUMMIES = [
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]



def main():
    # 常见编码依次尝试（含中文 Windows 导出的 GBK/GB18030）
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(INPUT_CSV, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("无法用 utf-8 / gb18030 / gbk 解析 CSV，请检查文件编码")
    # 日期列（取第一列）用于创建星期虚拟变量
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    # 创建星期虚拟变量（周二到周日，周一作为基准组）
    # dayofweek: 0=Monday, 1=Tuesday, ..., 6=Sunday
    weekday_num = df[date_col].dt.dayofweek
    df["Tuesday"] = (weekday_num == 1).astype(int)
    df["Wednesday"] = (weekday_num == 2).astype(int)
    df["Thursday"] = (weekday_num == 3).astype(int)
    df["Friday"] = (weekday_num == 4).astype(int)
    df["Saturday"] = (weekday_num == 5).astype(int)
    df["Sunday"] = (weekday_num == 6).astype(int)
    
    # 将星期虚拟变量添加到自变量列表
    all_independent_vars = INDEPENDENT_VARS + WEEKDAY_DUMMIES
    # 输出用自变量：同时包含 confirm 与 death，死亡模型填 death、确诊模型填 confirm
    output_independent_vars = (
        ["US_daily_covid_confirm", "US_daily_covid_death"]
        + [v for v in INDEPENDENT_VARS if v != "US_daily_covid_confirm"]
        + WEEKDAY_DUMMIES
    )
    all_vars = DEPENDENT_VARS + all_independent_vars
    
    for c in all_vars:
        if c not in df.columns:
            raise ValueError(f"列不存在: {c}")
    # 全样本：所有回归变量均无缺失，用于“先全样本标准化、再分段”
    full = df.dropna(subset=all_vars).copy()
    if len(full) < 10:
        raise ValueError(f"样本量 {len(full)} 过少，无法进行回归分析")
    
    emotion_vars = [v for v in DEPENDENT_VARS if v != "climate_post"]
    full_mean = full[emotion_vars + all_independent_vars].mean()
    full_std = full[emotion_vars + all_independent_vars].std()

    results_list = []
    summaries = []  # (因变量, model)
    climate_post_model_info = []  # (模型名, model, 自变量列表)，用于 2×2 发帖量森林图
    key_effect_rows = []  # 关键系数与95%置信区间汇总

    for y_name in DEPENDENT_VARS:
        if y_name == "climate_post":
            # 发帖量模型 1：因变量为原始发帖量，自变量中仅 US_daily_covid_confirm 按千人缩放
            y1 = full["climate_post"]
            X1 = full[all_independent_vars].copy()
            X1["US_daily_covid_confirm"] = full["US_daily_covid_confirm"] / 1000.0
            X1 = add_constant(X1)
            model1 = OLS(y1, X1).fit()
            summaries.append(("climate_post_model_1k", model1))
            climate_post_model_info.append(("climate_post_model_1k", model1, all_independent_vars.copy()))
            results_list.append({
                "因变量": "climate_post_model_1k",
                "R_squared": model1.rsquared,
                "Adj_R_squared": model1.rsquared_adj,
                "F_statistic": model1.fvalue,
                "F_pvalue": model1.f_pvalue,
                "N_obs": int(model1.nobs),
            })
            for var in all_independent_vars:
                results_list[-1][f"coef_{var}"] = model1.params.get(var, np.nan)
                results_list[-1][f"se_{var}"] = model1.bse.get(var, np.nan)
                results_list[-1][f"pval_{var}"] = model1.pvalues.get(var, np.nan)

            # 发帖量模型 2：保持原有设定，因变量 log(1+发帖量)，US_daily_covid_confirm 取 log(1+x)
            y2 = np.log1p(full["climate_post"])
            X2 = full[all_independent_vars].copy()
            X2["US_daily_covid_confirm"] = np.log1p(full["US_daily_covid_confirm"])
            X2 = add_constant(X2)
            model2 = OLS(y2, X2).fit()
            summaries.append(("climate_post_model_ln", model2))
            climate_post_model_info.append(("climate_post_model_ln", model2, all_independent_vars.copy()))
            results_list.append({
                "因变量": "climate_post_model_ln",
                "R_squared": model2.rsquared,
                "Adj_R_squared": model2.rsquared_adj,
                "F_statistic": model2.fvalue,
                "F_pvalue": model2.f_pvalue,
                "N_obs": int(model2.nobs),
            })
            for var in all_independent_vars:
                results_list[-1][f"coef_{var}"] = model2.params.get(var, np.nan)
                results_list[-1][f"se_{var}"] = model2.bse.get(var, np.nan)
                results_list[-1][f"pval_{var}"] = model2.pvalues.get(var, np.nan)

            # 发帖量模型 3：因变量为原始发帖量，自变量中仅 US_daily_covid_death 按千人缩放
            if "US_daily_covid_death" not in full.columns:
                raise ValueError("列不存在: US_daily_covid_death")
            y3 = full["climate_post"]
            # 创建新的自变量列表，将 US_daily_covid_confirm 替换为 US_daily_covid_death
            independent_vars_death = [v if v != "US_daily_covid_confirm" else "US_daily_covid_death" for v in INDEPENDENT_VARS] + WEEKDAY_DUMMIES
            X3 = full[independent_vars_death].copy()
            X3["US_daily_covid_death"] = full["US_daily_covid_death"] / 1000.0
            X3 = add_constant(X3)
            model3 = OLS(y3, X3).fit()
            summaries.append(("climate_post_model_1k_death", model3))
            climate_post_model_info.append(("climate_post_model_1k_death", model3, independent_vars_death.copy()))
            results_list.append({
                "因变量": "climate_post_model_1k_death",
                "R_squared": model3.rsquared,
                "Adj_R_squared": model3.rsquared_adj,
                "F_statistic": model3.fvalue,
                "F_pvalue": model3.f_pvalue,
                "N_obs": int(model3.nobs),
            })
            for var in all_independent_vars:
                # 如果变量是 US_daily_covid_confirm，则从模型中获取 US_daily_covid_death 的系数
                if var == "US_daily_covid_confirm":
                    results_list[-1][f"coef_{var}"] = model3.params.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"se_{var}"] = model3.bse.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"pval_{var}"] = model3.pvalues.get("US_daily_covid_death", np.nan)
                else:
                    results_list[-1][f"coef_{var}"] = model3.params.get(var, np.nan)
                    results_list[-1][f"se_{var}"] = model3.bse.get(var, np.nan)
                    results_list[-1][f"pval_{var}"] = model3.pvalues.get(var, np.nan)

            # 发帖量模型 4：因变量 log(1+发帖量)，US_daily_covid_death 取 log(1+x)
            y4 = np.log1p(full["climate_post"])
            X4 = full[independent_vars_death].copy()
            X4["US_daily_covid_death"] = np.log1p(full["US_daily_covid_death"])
            X4 = add_constant(X4)
            model4 = OLS(y4, X4).fit()
            summaries.append(("climate_post_model_ln_death", model4))
            climate_post_model_info.append(("climate_post_model_ln_death", model4, independent_vars_death.copy()))
            results_list.append({
                "因变量": "climate_post_model_ln_death",
                "R_squared": model4.rsquared,
                "Adj_R_squared": model4.rsquared_adj,
                "F_statistic": model4.fvalue,
                "F_pvalue": model4.f_pvalue,
                "N_obs": int(model4.nobs),
            })
            for var in all_independent_vars:
                # 如果变量是 US_daily_covid_confirm，则从模型中获取 US_daily_covid_death 的系数
                if var == "US_daily_covid_confirm":
                    results_list[-1][f"coef_{var}"] = model4.params.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"se_{var}"] = model4.bse.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"pval_{var}"] = model4.pvalues.get("US_daily_covid_death", np.nan)
                else:
                    results_list[-1][f"coef_{var}"] = model4.params.get(var, np.nan)
                    results_list[-1][f"se_{var}"] = model4.bse.get(var, np.nan)
                    results_list[-1][f"pval_{var}"] = model4.pvalues.get(var, np.nan)
            
            # 跳过后续统一处理
            continue
        else:
            # 用全样本 full 的均值和标准差做标准化；自变量中仅 STANDARDIZE_VARS 做标准化，其余保留原始取值
            y = (full[y_name] - full_mean[y_name]) / full_std[y_name]
            X_mat = full[all_independent_vars].copy()
            for c in all_independent_vars:
                if c in STANDARDIZE_VARS:
                    X_mat[c] = (full[c] - full_mean[c]) / full_std[c]
            y_label = y_name
        X = add_constant(X_mat)

        model = OLS(y, X).fit()
        results_list.append({
            "因变量": y_label,
            "R_squared": model.rsquared,
            "Adj_R_squared": model.rsquared_adj,
            "F_statistic": model.fvalue,
            "F_pvalue": model.f_pvalue,
            "N_obs": int(model.nobs),
        })
        for var in all_independent_vars:
            results_list[-1][f"coef_{var}"] = model.params.get(var, np.nan)
            results_list[-1][f"se_{var}"] = model.bse.get(var, np.nan)
            results_list[-1][f"pval_{var}"] = model.pvalues.get(var, np.nan)
        summaries.append((y_label, model))

    # 回归系数汇总表（每因变量一行）；含 US_daily_covid_confirm 与 US_daily_covid_death
    coeff_rows = []
    for y_name, model in summaries:
        row = {"因变量": y_name, "R_sq": model.rsquared, "Adj_R_sq": model.rsquared_adj, "N": int(model.nobs)}
        for v in output_independent_vars:
            row[v] = model.params.get(v, np.nan)
            row[f"{v}_se"] = model.bse.get(v, np.nan)
            row[f"{v}_pval"] = model.pvalues.get(v, np.nan)
        coeff_rows.append(row)

    # 额外：情感模型中，将 US_daily_covid_confirm 替换为 US_daily_covid_death 的结果也输出到 ols_coefficients.csv
    # 因变量仍是 8 个情感频率，只是自变量改为 death（其余变量不变）
    if "US_daily_covid_death" in full.columns:
        # death 版本的自变量列表（仅将 confirm 换成 death）
        independent_vars_death_for_emotion = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in INDEPENDENT_VARS
        ] + WEEKDAY_DUMMIES
        # 需要标准化的变量中同样把 confirm 换成 death
        standardize_vars_death = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in STANDARDIZE_VARS
        ]
        # death 的均值和标准差单独计算
        full_mean_death = full["US_daily_covid_death"].mean()
        full_std_death = full["US_daily_covid_death"].std()

        # 使用与绘制情感森林图相同的 8 个情感维度顺序
        emotion_order = ["sadness", "anger", "fear", "disgust", "surprise", "joy", "trust", "anticipation"]
        for emotion in emotion_order:
            y_name = f"climate_{emotion}_score_freq"
            # 因变量仍按全样本均值和标准差标准化
            y_death = (full[y_name] - full_mean[y_name]) / full_std[y_name]
            X_mat_death = full[independent_vars_death_for_emotion].copy()
            # 自变量中仅 standardize_vars_death 做标准化
            for c in independent_vars_death_for_emotion:
                if c in standardize_vars_death:
                    if c == "US_daily_covid_death":
                        X_mat_death[c] = (full[c] - full_mean_death) / full_std_death
                    else:
                        X_mat_death[c] = (full[c] - full_mean[c]) / full_std[c]
            X_death = add_constant(X_mat_death)
            model_death = OLS(y_death, X_death).fit()

            # 为每个“情感 × death 自变量”的模型，在 ols_coefficients.csv 中增加一行
            # 为避免与 confirm 版本混淆，这里在因变量名后加后缀 "_death"
            row = {
                "因变量": f"{y_name}_death",
                "R_sq": model_death.rsquared,
                "Adj_R_sq": model_death.rsquared_adj,
                "N": int(model_death.nobs),
            }
            for v in output_independent_vars:
                row[v] = model_death.params.get(v, np.nan)
                row[f"{v}_se"] = model_death.bse.get(v, np.nan)
                row[f"{v}_pval"] = model_death.pvalues.get(v, np.nan)
            coeff_rows.append(row)
    pd.DataFrame(coeff_rows).to_csv(OUTPUT_COEFFS_CSV, index=False, encoding="utf-8-sig")

    # p 值对应的显著性星号：* p≤0.05; ** p≤0.01; *** p≤0.001
    def sig_star(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        if p <= 0.001:
            return "***"
        if p <= 0.01:
            return "**"
        if p <= 0.05:
            return "*"
        return ""

    # 回归摘要输出为 CSV：系数/标准误/t/p、星号，以及模型统计量 R²、Adj.R²、F、N；含 US_daily_covid_death
    summary_rows = []
    for y_name, model in summaries:
        for var in ["const"] + output_independent_vars:
            p = model.pvalues.get(var, np.nan)
            summary_rows.append({
                "因变量": y_name,
                "变量": var,
                "系数": model.params.get(var, np.nan),
                "标准误": model.bse.get(var, np.nan),
                "t值": model.tvalues.get(var, np.nan),
                "p值": p,
                "星号": sig_star(p),
            })
        for var, coef in [("R_sq", model.rsquared), ("Adj_R_sq", model.rsquared_adj), ("F", model.fvalue), ("F_pval", model.f_pvalue), ("N", int(model.nobs))]:
            summary_rows.append({"因变量": y_name, "变量": var, "系数": coef, "标准误": np.nan, "t值": np.nan, "p值": np.nan, "星号": ""})
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"已写入: {OUTPUT_SUMMARY_CSV}, {OUTPUT_COEFFS_CSV}")

    # 森林图中文显示：优先使用系统自带的中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1 个森林图：横轴=8 个情感因变量，纵轴=US_daily_covid_confirm 系数，点估计+95% 置信区间，置信区间上标星号
    # 横轴排序：sadness, anger, fear, disgust, surprise, joy, trust, anticipation
    emotion_order = ["sadness", "anger", "fear", "disgust", "surprise", "joy", "trust", "anticipation"]
    var_focus = "US_daily_covid_confirm"
    model_by_y = {yn: m for yn, m in summaries}

    # 提取 climate_post_model_ln 与 climate_post_model_ln_death 的关键系数、p值及95%置信区间
    # 1) climate_post_model_ln 中 US_daily_covid_confirm 的系数
    if "climate_post_model_ln" in model_by_y:
        m_ln = model_by_y["climate_post_model_ln"]
        ci_ln = m_ln.conf_int(alpha=0.05)
        if var_focus in ci_ln.index:
            lo, hi = ci_ln.loc[var_focus, 0], ci_ln.loc[var_focus, 1]
        else:
            lo, hi = np.nan, np.nan
        key_effect_rows.append({
            "模型": "climate_post_model_ln",
            "因变量": "climate_post",
            "自变量": var_focus,
            "系数": m_ln.params.get(var_focus, np.nan),
            "p值": m_ln.pvalues.get(var_focus, np.nan),
            "CI_low": lo,
            "CI_high": hi,
        })
    # 2) climate_post_model_ln_death 中 US_daily_covid_death 的系数
    var_focus_death = "US_daily_covid_death"
    if "climate_post_model_ln_death" in model_by_y:
        m_ln_death = model_by_y["climate_post_model_ln_death"]
        ci_ln_death = m_ln_death.conf_int(alpha=0.05)
        if var_focus_death in ci_ln_death.index:
            lo_d, hi_d = ci_ln_death.loc[var_focus_death, 0], ci_ln_death.loc[var_focus_death, 1]
        else:
            lo_d, hi_d = np.nan, np.nan
        key_effect_rows.append({
            "模型": "climate_post_model_ln_death",
            "因变量": "climate_post",
            "自变量": var_focus_death,
            "系数": m_ln_death.params.get(var_focus_death, np.nan),
            "p值": m_ln_death.pvalues.get(var_focus_death, np.nan),
            "CI_low": lo_d,
            "CI_high": hi_d,
        })
    coefs, ci_lo, ci_hi, labels, pvals = [], [], [], [], []
    for emotion in emotion_order:
        y_name = f"climate_{emotion}_score_freq"
        m = model_by_y.get(y_name)
        if m is None:
            continue
        b = m.params.get(var_focus, np.nan)
        pv = m.pvalues.get(var_focus, np.nan)
        ci = m.conf_int(alpha=0.05)
        if var_focus in ci.index:
            lo, hi = ci.loc[var_focus, 0], ci.loc[var_focus, 1]
        else:
            lo, hi = np.nan, np.nan
        coefs.append(b)
        ci_lo.append(lo)
        ci_hi.append(hi)
        labels.append(emotion)
        pvals.append(pv)
    if coefs:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        coefs_arr = np.array(coefs)
        ci_lo_arr = np.array(ci_lo)
        ci_hi_arr = np.array(ci_hi)
        ax.errorbar(x, coefs_arr, yerr=[coefs_arr - ci_lo_arr, ci_hi_arr - coefs_arr],
                    fmt="o", capsize=4, capthick=1.5, markersize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        y_range = np.nanmax(ci_hi_arr) - np.nanmin(ci_lo_arr) if np.any(np.isfinite(ci_hi_arr)) and np.any(np.isfinite(ci_lo_arr)) else 1.0
        for i in range(len(x)):
            star = sig_star(pvals[i])
            if star:
                y_place = (ci_hi_arr[i] if np.isfinite(ci_hi_arr[i]) else coefs_arr[i]) + y_range * 0.04
                ax.text(x[i], y_place, star, ha="center", va="bottom", fontsize=11, fontweight="bold")
            # 在点右侧标记点估计值
            if np.isfinite(coefs_arr[i]):
                ax.text(x[i] + 0.1, coefs_arr[i], f"{coefs_arr[i]:.3f}", 
                       ha="left", va="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        # 缩小列间距：调整x轴范围，让左右边距更小
        ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
        ax.set_ylabel("Δclimate_emotions")
        ax.set_xlabel("")
        ax.set_title("COVID-19 cases")
        # 只保留下方和左侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        outpath = os.path.join(SCRIPT_DIR, "forest_by_dv.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {outpath}")

    # 2 个森林图：用 US_daily_covid_death 代替 US_daily_covid_confirm，标题为 COVID-19 death
    if "US_daily_covid_death" in full.columns:
        # 为情感模型创建使用 death 的版本（临时模型，仅用于画图）
        independent_vars_death_for_emotion = [v if v != "US_daily_covid_confirm" else "US_daily_covid_death" for v in INDEPENDENT_VARS] + WEEKDAY_DUMMIES
        # 更新 STANDARDIZE_VARS：将 confirm 替换为 death
        standardize_vars_death = [v if v != "US_daily_covid_confirm" else "US_daily_covid_death" for v in STANDARDIZE_VARS]
        # 计算 death 的均值和标准差（death 不在 full_mean 中，需要单独计算）
        full_mean_death = full["US_daily_covid_death"].mean()
        full_std_death = full["US_daily_covid_death"].std()
        
        coefs_death, ci_lo_death, ci_hi_death, labels_death, pvals_death = [], [], [], [], []
        # 使用相同的横轴排序：sadness, anger, fear, disgust, surprise, joy, trust, anticipation
        for emotion in emotion_order:
            y_name = f"climate_{emotion}_score_freq"
            # 为情感模型创建使用 death 的临时模型
            y_death = (full[y_name] - full_mean[y_name]) / full_std[y_name]
            X_mat_death = full[independent_vars_death_for_emotion].copy()
            for c in independent_vars_death_for_emotion:
                if c in standardize_vars_death:
                    if c == "US_daily_covid_death":
                        X_mat_death[c] = (full[c] - full_mean_death) / full_std_death
                    else:
                        X_mat_death[c] = (full[c] - full_mean[c]) / full_std[c]
            X_death = add_constant(X_mat_death)
            model_death = OLS(y_death, X_death).fit()
            
            b_death = model_death.params.get(var_focus_death, np.nan)
            pv_death = model_death.pvalues.get(var_focus_death, np.nan)
            ci_death = model_death.conf_int(alpha=0.05)
            if var_focus_death in ci_death.index:
                lo_death, hi_death = ci_death.loc[var_focus_death, 0], ci_death.loc[var_focus_death, 1]
            else:
                lo_death, hi_death = np.nan, np.nan
            # 将八个情感模型中 US_daily_covid_death 的系数与95%置信区间记录下来
            key_effect_rows.append({
                "模型": "emotion_death",
                "因变量": y_name,
                "自变量": var_focus_death,
                "系数": b_death,
                 "p值": pv_death,
                "CI_low": lo_death,
                "CI_high": hi_death,
            })
            coefs_death.append(b_death)
            ci_lo_death.append(lo_death)
            ci_hi_death.append(hi_death)
            labels_death.append(emotion)
            pvals_death.append(pv_death)
        
        if coefs_death:
            fig_death, ax_death = plt.subplots(figsize=(10, 5))
            x_death = np.arange(len(labels_death))
            coefs_arr_death = np.array(coefs_death)
            ci_lo_arr_death = np.array(ci_lo_death)
            ci_hi_arr_death = np.array(ci_hi_death)
            ax_death.errorbar(x_death, coefs_arr_death, yerr=[coefs_arr_death - ci_lo_arr_death, ci_hi_arr_death - coefs_arr_death],
                        fmt="o", capsize=4, capthick=1.5, markersize=8)
            ax_death.axhline(0, color="gray", linestyle="--", linewidth=1)
            y_range_death = np.nanmax(ci_hi_arr_death) - np.nanmin(ci_lo_arr_death) if np.any(np.isfinite(ci_hi_arr_death)) and np.any(np.isfinite(ci_lo_arr_death)) else 1.0
            for i in range(len(x_death)):
                star_death = sig_star(pvals_death[i])
                if star_death:
                    y_place_death = (ci_hi_arr_death[i] if np.isfinite(ci_hi_arr_death[i]) else coefs_arr_death[i]) + y_range_death * 0.04
                    ax_death.text(x_death[i], y_place_death, star_death, ha="center", va="bottom", fontsize=11, fontweight="bold")
                # 在点右侧标记点估计值
                if np.isfinite(coefs_arr_death[i]):
                    ax_death.text(x_death[i] + 0.1, coefs_arr_death[i], f"{coefs_arr_death[i]:.3f}", 
                           ha="left", va="center", fontsize=9)
            ax_death.set_xticks(x_death)
            ax_death.set_xticklabels(labels_death, rotation=0, ha="center")
            # 缩小列间距：调整x轴范围，让左右边距更小
            ax_death.set_xlim(x_death[0] - 0.3, x_death[-1] + 0.3)
            ax_death.set_ylabel("Δclimate_emotions")
            ax_death.set_xlabel("")
            ax_death.set_title("COVID-19 death")
            # 只保留下方和左侧边框
            ax_death.spines['top'].set_visible(False)
            ax_death.spines['right'].set_visible(False)
            fig_death.tight_layout()
            outpath_death = os.path.join(SCRIPT_DIR, "forest_by_dv_death.png")
            fig_death.savefig(outpath_death, dpi=150, bbox_inches="tight")
            plt.close(fig_death)
            print(f"  已保存: {outpath_death}")

    # ========= 生成年份类似示例表格的 CSV：前两个发帖量模型 =========
    # Model 1: 使用死亡数的对数（climate_post_model_ln_death）
    # Model 2: 使用确诊数的对数（climate_post_model_ln）
    table_rows = []
    if "climate_post_model_ln_death" in model_by_y and "climate_post_model_ln" in model_by_y:
        m1 = model_by_y["climate_post_model_ln_death"]
        m2 = model_by_y["climate_post_model_ln"]

        def sig_star(p):
            # * p≤0.05; ** p≤0.01; *** p≤0.001
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return ""
            if p <= 0.001:
                return "***"
            if p <= 0.01:
                return "**"
            if p <= 0.05:
                return "*"
            return ""

        # 变量显示名称映射（若不在此映射中，则使用原列名）
        var_label_map = {
            "US_daily_covid_death": "Log (US daily COVID-19 deaths)",
            "US_daily_covid_confirm": "Log (US daily COVID-19 confirmed cases)",
            "climatenews": "Climate news coverage",
            "GovernmentResponseIndex_Average": "OxCGRT Stringency Index",
            "debates": "Presidential Debates",
            "WinterStorm": "Winter Storm",
            "Wildfire": "Wildfire",
            "TropicalCyclone": "Tropical Cyclone",
            "SevereStorm": "Severe Storm",
            "Flood": "Flood",
            "Drought": "Drought",
            "Tuesday": "Tuesday",
            "Wednesday": "Wednesday",
            "Thursday": "Thursday",
            "Friday": "Friday",
            "Saturday": "Saturday",
            "Sunday": "Sunday",
        }

        # 按示例顺序组织自变量
        var_order_for_table = (
            ["US_daily_covid_death", "US_daily_covid_confirm"]
            + [v for v in INDEPENDENT_VARS if v not in ("US_daily_covid_confirm",)]
            + WEEKDAY_DUMMIES
        )

        def format_coef(model, var):
            coef = model.params.get(var, np.nan)
            p = model.pvalues.get(var, np.nan)
            star = sig_star(p)
            if isinstance(coef, float) and np.isnan(coef):
                return ""
            return f"{coef:.4f}{star}"

        def format_se(model, var):
            coef = model.params.get(var, np.nan)
            se = model.bse.get(var, np.nan)
            if isinstance(coef, float) and np.isnan(coef):
                return ""
            se_str = f"{se:.4f}" if (isinstance(se, (float, np.floating)) and np.isfinite(se)) else "NA"
            return f"({se_str})"

        # 自变量行：每个变量 2 行（系数行 + 标准误行）
        for v in var_order_for_table:
            label = var_label_map.get(v, v)
            row_coef = {
                "Independent Variable": label,
                "Model 1 (COVID-19 Deaths)": format_coef(m1, v),
                "Model 2 (COVID-19 Cases)": format_coef(m2, v),
            }
            row_se = {
                "Independent Variable": "",
                "Model 1 (COVID-19 Deaths)": format_se(m1, v),
                "Model 2 (COVID-19 Cases)": format_se(m2, v),
            }
            table_rows.append(row_coef)
            table_rows.append(row_se)

        # 底部统计量：N、R-squared、Adjusted R-squared
        stats_rows = [
            ("Observations (N)", int(m1.nobs), int(m2.nobs)),
            ("R-squared", f"{m1.rsquared:.4f}", f"{m2.rsquared:.4f}"),
            ("Adjusted R-squared", f"{m1.rsquared_adj:.4f}", f"{m2.rsquared_adj:.4f}"),
        ]
        for label, v1, v2 in stats_rows:
            row = {
                "Independent Variable": label,
                "Model 1 (COVID-19 Deaths)": v1,
                "Model 2 (COVID-19 Cases)": v2,
            }
            table_rows.append(row)

        table_df = pd.DataFrame(table_rows)
        output_table_csv = os.path.join(SCRIPT_DIR, "climate_post_models_1_2_table.csv")
        table_df.to_csv(output_table_csv, index=False, encoding="utf-8-sig")
        print(f"已写入: {output_table_csv}")

    # ========= 8 个情感模型：死亡数 vs 确诊数 的表格 CSV =========
    # 生成一个 CSV，结构与上述类似，但多一列区分情感因变量
    if "US_daily_covid_death" in full.columns:
        # 依然使用上面定义的变量顺序
        var_label_map = {
            "US_daily_covid_death": "Log (US daily COVID-19 deaths)",
            "US_daily_covid_confirm": "Log (US daily COVID-19 confirmed cases)",
            "climatenews": "Climate news coverage",
            "GovernmentResponseIndex_Average": "OxCGRT Stringency Index",
            "debates": "Presidential Debates",
            "WinterStorm": "Winter Storm",
            "Wildfire": "Wildfire",
            "TropicalCyclone": "Tropical Cyclone",
            "SevereStorm": "Severe Storm",
            "Flood": "Flood",
            "Drought": "Drought",
            "Tuesday": "Tuesday",
            "Wednesday": "Wednesday",
            "Thursday": "Thursday",
            "Friday": "Friday",
            "Saturday": "Saturday",
            "Sunday": "Sunday",
        }
        var_order_for_table = (
            ["US_daily_covid_death", "US_daily_covid_confirm"]
            + [v for v in INDEPENDENT_VARS if v not in ("US_daily_covid_confirm",)]
            + WEEKDAY_DUMMIES
        )

        def sig_star(p):
            # * p≤0.05; ** p≤0.01; *** p≤0.001
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return ""
            if p <= 0.001:
                return "***"
            if p <= 0.01:
                return "**"
            if p <= 0.05:
                return "*"
            return ""

        def format_coef_se(model, var):
            coef = model.params.get(var, np.nan)
            se = model.bse.get(var, np.nan)
            p = model.pvalues.get(var, np.nan)
            star = sig_star(p)
            if isinstance(coef, float) and np.isnan(coef):
                return ""
            se_str = f"{se:.4f}" if (isinstance(se, (float, np.floating)) and np.isfinite(se)) else "NA"
            # 在 CSV 中前置一个单引号，避免 Excel 将 * 误判为公式乘号
            return f"'{coef:.4f}{star}\n({se_str})"

        # death 版本的自变量列表（仅将 confirm 换成 death）
        independent_vars_death_for_emotion = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in INDEPENDENT_VARS
        ] + WEEKDAY_DUMMIES
        # 需要标准化的变量中同样把 confirm 换成 death
        standardize_vars_death = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in STANDARDIZE_VARS
        ]
        full_mean_death = full["US_daily_covid_death"].mean()
        full_std_death = full["US_daily_covid_death"].std()

        # 8 个情感因变量顺序
        emotion_order = ["sadness", "anger", "fear", "disgust", "surprise", "joy", "trust", "anticipation"]
        emotion_label_map = {
            "sadness": "Sadness",
            "anger": "Anger",
            "fear": "Fear",
            "disgust": "Disgust",
            "surprise": "Surprise",
            "joy": "Joy",
            "trust": "Trust",
            "anticipation": "Anticipation",
        }

        table_rows_emotion = []
        # 保存每个情感对应的 death 模型，后面生成“8 列情绪”的宽表
        death_models_by_emotion = {}
        for emotion in emotion_order:
            y_name = f"climate_{emotion}_score_freq"
            # 确诊数模型（已在 summaries 中）
            m_cases = model_by_y.get(y_name)
            if m_cases is None:
                continue
            # 死亡数模型：与上方画森林图时的设定一致
            y_death = (full[y_name] - full_mean[y_name]) / full_std[y_name]
            X_mat_death = full[independent_vars_death_for_emotion].copy()
            for c in independent_vars_death_for_emotion:
                if c in standardize_vars_death:
                    if c == "US_daily_covid_death":
                        X_mat_death[c] = (full[c] - full_mean_death) / full_std_death
                    else:
                        X_mat_death[c] = (full[c] - full_mean[c]) / full_std[c]
            X_death = add_constant(X_mat_death)
            m_death = OLS(y_death, X_death).fit()
            death_models_by_emotion[emotion] = m_death

            dep_label = emotion_label_map.get(emotion, emotion)
            # 自变量行
            for v in var_order_for_table:
                row = {
                    "Dependent Variable": dep_label,
                    "Independent Variable": var_label_map.get(v, v),
                    "Model 1 (COVID-19 Deaths)": format_coef_se(m_death, v),
                    "Model 2 (COVID-19 Cases)": format_coef_se(m_cases, v),
                }
                table_rows_emotion.append(row)

            # 每个情感模型后面加 3 行统计量
            stats_rows = [
                ("Observations (N)", int(m_death.nobs), int(m_cases.nobs)),
                ("R-squared", m_death.rsquared, m_cases.rsquared),
                ("Adjusted R-squared", m_death.rsquared_adj, m_cases.rsquared_adj),
            ]
            for label, v1, v2 in stats_rows:
                row = {
                    "Dependent Variable": dep_label,
                    "Independent Variable": label,
                    "Model 1 (COVID-19 Deaths)": v1,
                    "Model 2 (COVID-19 Cases)": v2,
                }
                table_rows_emotion.append(row)

        if table_rows_emotion:
            emotion_table_df = pd.DataFrame(table_rows_emotion)
            output_emotion_table_csv = os.path.join(SCRIPT_DIR, "climate_emotions_death_vs_cases_table.csv")
            emotion_table_df.to_csv(output_emotion_table_csv, index=False, encoding="utf-8-sig")
            print(f"已写入: {output_emotion_table_csv}")

        # 只使用“死亡数自变量”的 8 个情绪模型，生成一个宽表：
        # 行 = 自变量，列 = 8 个情绪（单元格为 系数+星号 换行 (SE)）
        if death_models_by_emotion:
            wide_rows = []
            # 列名顺序：Independent Variable + 8 个情绪
            wide_columns = ["Independent Variable"] + [emotion_label_map[e] for e in emotion_order]
            # 仅输出“死亡模型”里实际进入回归的自变量（不包含 confirm）
            var_order_death_only = (
                ["US_daily_covid_death"]
                + [v for v in INDEPENDENT_VARS if v not in ("US_daily_covid_confirm",)]
                + WEEKDAY_DUMMIES
            )
            for v in var_order_death_only:
                # 第一行：系数（含星号）；第二行：标准误（括号）
                row_coef = {"Independent Variable": var_label_map.get(v, v)}
                row_se = {"Independent Variable": ""}
                for e in emotion_order:
                    m_death = death_models_by_emotion.get(e)
                    if m_death is None:
                        row_coef[emotion_label_map[e]] = ""
                        row_se[emotion_label_map[e]] = ""
                    else:
                        coef = m_death.params.get(v, np.nan)
                        se = m_death.bse.get(v, np.nan)
                        p = m_death.pvalues.get(v, np.nan)
                        star = sig_star(p)
                        if isinstance(coef, float) and np.isnan(coef):
                            row_coef[emotion_label_map[e]] = ""
                            row_se[emotion_label_map[e]] = ""
                        else:
                            row_coef[emotion_label_map[e]] = f"{coef:.4f}{star}"
                            se_str = f"{se:.4f}" if (isinstance(se, (float, np.floating)) and np.isfinite(se)) else "NA"
                            row_se[emotion_label_map[e]] = f"({se_str})"
                wide_rows.append(row_coef)
                wide_rows.append(row_se)

            # 末尾增加 3 行整体统计量：N、R-squared、Adjusted R-squared（每个情绪一列）
            stats_labels = ["Observations (N)", "R-squared", "Adjusted R-squared"]
            for label in stats_labels:
                stats_row = {"Independent Variable": label}
                for e in emotion_order:
                    m_death = death_models_by_emotion.get(e)
                    if m_death is None:
                        stats_row[emotion_label_map[e]] = ""
                    else:
                        if label == "Observations (N)":
                            stats_row[emotion_label_map[e]] = int(m_death.nobs)
                        elif label == "R-squared":
                            stats_row[emotion_label_map[e]] = m_death.rsquared
                        elif label == "Adjusted R-squared":
                            stats_row[emotion_label_map[e]] = m_death.rsquared_adj
                wide_rows.append(stats_row)
            wide_df = pd.DataFrame(wide_rows, columns=wide_columns)
            output_emotion_death_only_csv = os.path.join(SCRIPT_DIR, "climate_emotions_death_only_wide.csv")
            try:
                wide_df.to_csv(output_emotion_death_only_csv, index=False, encoding="utf-8-sig")
                print(f"已写入: {output_emotion_death_only_csv}")
            except PermissionError:
                # 文件可能正被 Excel 占用，改写入一个新文件名
                alt_path = os.path.join(SCRIPT_DIR, "climate_emotions_death_only_wide_v2.csv")
                wide_df.to_csv(alt_path, index=False, encoding="utf-8-sig")
                print(f"原文件被占用，已改写入: {alt_path}")

    # 4 个发帖量模型 → 2×2 森林图，合并为一张图
    if len(climate_post_model_info) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        for idx, (mname, model, var_list) in enumerate(climate_post_model_info):
            ax = axes[idx]
            coefs, ci_lo, ci_hi, labels, pvals = [], [], [], [], []
            ci = model.conf_int(alpha=0.05)
            for v in var_list:
                b = model.params.get(v, np.nan)
                pv = model.pvalues.get(v, np.nan)
                if v in ci.index:
                    lo, hi = ci.loc[v, 0], ci.loc[v, 1]
                else:
                    lo, hi = np.nan, np.nan
                coefs.append(b)
                ci_lo.append(lo)
                ci_hi.append(hi)
                labels.append(v)
                pvals.append(pv)
            y_pos = np.arange(len(labels))
            coefs_arr = np.array(coefs)
            ci_lo_arr = np.array(ci_lo)
            ci_hi_arr = np.array(ci_hi)
            xerr = np.array([coefs_arr - ci_lo_arr, ci_hi_arr - coefs_arr])
            ax.errorbar(coefs_arr, y_pos, xerr=xerr, fmt="o", capsize=3, capthick=1.2, markersize=5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            x_min = np.nanmin(ci_lo_arr) if np.any(np.isfinite(ci_lo_arr)) else -0.5
            x_max = np.nanmax(ci_hi_arr) if np.any(np.isfinite(ci_hi_arr)) else 0.5
            x_range = max(x_max - x_min, 0.1)
            y_range = len(y_pos) if len(y_pos) > 0 else 1.0
            for i in range(len(y_pos)):
                # 在点上方标记点估计值（由于 invert_yaxis，上方是 y 值更小的位置）
                if np.isfinite(coefs_arr[i]):
                    ax.text(coefs_arr[i], y_pos[i] - y_range * 0.04, f"{coefs_arr[i]:.3f}", 
                           ha="center", va="top", fontsize=7)
                star = sig_star(pvals[i])
                if star:
                    ax.text(x_max + x_range * 0.02, y_pos[i], star, ha="left", va="center", fontsize=9, fontweight="bold")
            x_lo = min(x_min - 0.02 * x_range, 0)
            x_hi = max(x_max + 0.12 * x_range, 0)
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel("系数 (95% CI)")
            ax.set_title(mname, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        outpath_4 = os.path.join(SCRIPT_DIR, "forest_climate_post_2x2.png")
        fig.savefig(outpath_4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {outpath_4}")

    # 将关键系数与95%置信区间输出为单独的 CSV
    if key_effect_rows:
        pd.DataFrame(key_effect_rows).to_csv(OUTPUT_KEY_CI_CSV, index=False, encoding="utf-8-sig")
        print(f"已写入: {OUTPUT_KEY_CI_CSV}")

    print("各因变量与 R2 一览:")
    for y_name, model in summaries:
        print(f"  {y_name}: R2={model.rsquared:.4f}, Adj_R2={model.rsquared_adj:.4f}, N={int(model.nobs)}")


if __name__ == "__main__":
    main()
