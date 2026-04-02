# -*- coding: utf-8 -*-
"""
OLS regressions: climate post volume and emotions on exogenous variables.
Input: climate_post_and_emotions.csv
9 regressions: post volume + 8 emotion frequencies as dependent variables; disasters/COVID/news/etc. as regressors.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ============ Config ============
# Use the script directory as base so running from anywhere works
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "climate_post_and_emotions.csv")
OUTPUT_SUMMARY_CSV = os.path.join(SCRIPT_DIR, "ols_regression_summary.csv")
OUTPUT_COEFFS_CSV = os.path.join(SCRIPT_DIR, "ols_coefficients.csv")
OUTPUT_KEY_CI_CSV = os.path.join(SCRIPT_DIR, "key_covid_effects_ci.csv")

# Dependent variables (9 total)
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


# Independent variables
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

# Variables standardized in emotion models only (others kept as-is)
STANDARDIZE_VARS = [
    "US_daily_covid_confirm",
    "climatenews",
    "GovernmentResponseIndex_Average",
]

# Weekday dummies (Tue-Sun; Monday is the baseline)
WEEKDAY_DUMMIES = [
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]



def main():
    # Try common encodings (including GBK/GB18030 from some Windows exports)
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(INPUT_CSV, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Failed to decode CSV with utf-8 / gb18030 / gbk. Please check the file encoding.")
    # Use the first column as date for weekday dummies
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    # Create weekday dummies (Tue-Sun; Monday baseline)
    # dayofweek: 0=Monday, 1=Tuesday, ..., 6=Sunday
    weekday_num = df[date_col].dt.dayofweek
    df["Tuesday"] = (weekday_num == 1).astype(int)
    df["Wednesday"] = (weekday_num == 2).astype(int)
    df["Thursday"] = (weekday_num == 3).astype(int)
    df["Friday"] = (weekday_num == 4).astype(int)
    df["Saturday"] = (weekday_num == 5).astype(int)
    df["Sunday"] = (weekday_num == 6).astype(int)
    
    # Add weekday dummies to regressors
    all_independent_vars = INDEPENDENT_VARS + WEEKDAY_DUMMIES
    # Output regressors include both confirm and death (depending on model)
    output_independent_vars = (
        ["US_daily_covid_confirm", "US_daily_covid_death"]
        + [v for v in INDEPENDENT_VARS if v != "US_daily_covid_confirm"]
        + WEEKDAY_DUMMIES
    )
    all_vars = DEPENDENT_VARS + all_independent_vars
    
    for c in all_vars:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    # Full sample: no missing values on any regression variable
    full = df.dropna(subset=all_vars).copy()
    if len(full) < 10:
        raise ValueError(f"Sample size {len(full)} is too small for regression.")
    
    emotion_vars = [v for v in DEPENDENT_VARS if v != "climate_post"]
    full_mean = full[emotion_vars + all_independent_vars].mean()
    full_std = full[emotion_vars + all_independent_vars].std()

    results_list = []
    summaries = []  # (dependent_var, model)
    climate_post_model_info = []  # (model_name, model, regressor_list) for 2x2 post-volume forest plot
    key_effect_rows = []  # key coefficients and 95% CI summary

    for y_name in DEPENDENT_VARS:
        if y_name == "climate_post":
            # Post-volume model 1: DV is raw post count; scale US_daily_covid_confirm by 1,000
            y1 = full["climate_post"]
            X1 = full[all_independent_vars].copy()
            X1["US_daily_covid_confirm"] = full["US_daily_covid_confirm"] / 1000.0
            X1 = add_constant(X1)
            model1 = OLS(y1, X1).fit()
            summaries.append(("climate_post_model_1k", model1))
            climate_post_model_info.append(("climate_post_model_1k", model1, all_independent_vars.copy()))
            results_list.append({
                "dependent_var": "climate_post_model_1k",
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

            # Post-volume model 2: DV log(1+posts); regressor log(1+US_daily_covid_confirm)
            y2 = np.log1p(full["climate_post"])
            X2 = full[all_independent_vars].copy()
            X2["US_daily_covid_confirm"] = np.log1p(full["US_daily_covid_confirm"])
            X2 = add_constant(X2)
            model2 = OLS(y2, X2).fit()
            summaries.append(("climate_post_model_ln", model2))
            climate_post_model_info.append(("climate_post_model_ln", model2, all_independent_vars.copy()))
            results_list.append({
                "dependent_var": "climate_post_model_ln",
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

            # Post-volume model 3: DV is raw post count; scale US_daily_covid_death by 1,000
            if "US_daily_covid_death" not in full.columns:
                raise ValueError("Missing column: US_daily_covid_death")
            y3 = full["climate_post"]
            # Replace confirm with death in regressor list
            independent_vars_death = [v if v != "US_daily_covid_confirm" else "US_daily_covid_death" for v in INDEPENDENT_VARS] + WEEKDAY_DUMMIES
            X3 = full[independent_vars_death].copy()
            X3["US_daily_covid_death"] = full["US_daily_covid_death"] / 1000.0
            X3 = add_constant(X3)
            model3 = OLS(y3, X3).fit()
            summaries.append(("climate_post_model_1k_death", model3))
            climate_post_model_info.append(("climate_post_model_1k_death", model3, independent_vars_death.copy()))
            results_list.append({
                "dependent_var": "climate_post_model_1k_death",
                "R_squared": model3.rsquared,
                "Adj_R_squared": model3.rsquared_adj,
                "F_statistic": model3.fvalue,
                "F_pvalue": model3.f_pvalue,
                "N_obs": int(model3.nobs),
            })
            for var in all_independent_vars:
                # For output compatibility: report death coefficient under the confirm slot
                if var == "US_daily_covid_confirm":
                    results_list[-1][f"coef_{var}"] = model3.params.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"se_{var}"] = model3.bse.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"pval_{var}"] = model3.pvalues.get("US_daily_covid_death", np.nan)
                else:
                    results_list[-1][f"coef_{var}"] = model3.params.get(var, np.nan)
                    results_list[-1][f"se_{var}"] = model3.bse.get(var, np.nan)
                    results_list[-1][f"pval_{var}"] = model3.pvalues.get(var, np.nan)

            # Post-volume model 4: DV log(1+posts); regressor log(1+US_daily_covid_death)
            y4 = np.log1p(full["climate_post"])
            X4 = full[independent_vars_death].copy()
            X4["US_daily_covid_death"] = np.log1p(full["US_daily_covid_death"])
            X4 = add_constant(X4)
            model4 = OLS(y4, X4).fit()
            summaries.append(("climate_post_model_ln_death", model4))
            climate_post_model_info.append(("climate_post_model_ln_death", model4, independent_vars_death.copy()))
            results_list.append({
                "dependent_var": "climate_post_model_ln_death",
                "R_squared": model4.rsquared,
                "Adj_R_squared": model4.rsquared_adj,
                "F_statistic": model4.fvalue,
                "F_pvalue": model4.f_pvalue,
                "N_obs": int(model4.nobs),
            })
            for var in all_independent_vars:
                # For output compatibility: report death coefficient under the confirm slot
                if var == "US_daily_covid_confirm":
                    results_list[-1][f"coef_{var}"] = model4.params.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"se_{var}"] = model4.bse.get("US_daily_covid_death", np.nan)
                    results_list[-1][f"pval_{var}"] = model4.pvalues.get("US_daily_covid_death", np.nan)
                else:
                    results_list[-1][f"coef_{var}"] = model4.params.get(var, np.nan)
                    results_list[-1][f"se_{var}"] = model4.bse.get(var, np.nan)
                    results_list[-1][f"pval_{var}"] = model4.pvalues.get(var, np.nan)
            
            # Skip the unified processing below
            continue
        else:
            # Standardize DV using full-sample mean/std; only STANDARDIZE_VARS are standardized among regressors
            y = (full[y_name] - full_mean[y_name]) / full_std[y_name]
            X_mat = full[all_independent_vars].copy()
            for c in all_independent_vars:
                if c in STANDARDIZE_VARS:
                    X_mat[c] = (full[c] - full_mean[c]) / full_std[c]
            y_label = y_name
        X = add_constant(X_mat)

        model = OLS(y, X).fit()
        results_list.append({
            "dependent_var": y_label,
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

    # Coefficient summary (one row per DV); includes both confirm and death slots
    coeff_rows = []
    for y_name, model in summaries:
        row = {"dependent_var": y_name, "R_sq": model.rsquared, "Adj_R_sq": model.rsquared_adj, "N": int(model.nobs)}
        for v in output_independent_vars:
            row[v] = model.params.get(v, np.nan)
            row[f"{v}_se"] = model.bse.get(v, np.nan)
            row[f"{v}_pval"] = model.pvalues.get(v, np.nan)
        coeff_rows.append(row)

    # Extra: for emotion models, also output a "death" version (confirm -> death) into ols_coefficients.csv
    if "US_daily_covid_death" in full.columns:
        # Regressors for death version (confirm -> death)
        independent_vars_death_for_emotion = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in INDEPENDENT_VARS
        ] + WEEKDAY_DUMMIES
        # Standardized variables list (confirm -> death)
        standardize_vars_death = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in STANDARDIZE_VARS
        ]
        # Mean/std for death computed separately
        full_mean_death = full["US_daily_covid_death"].mean()
        full_std_death = full["US_daily_covid_death"].std()

        # Same emotion order as the forest plot
        emotion_order = ["sadness", "anger", "fear", "disgust", "surprise", "joy", "trust", "anticipation"]
        for emotion in emotion_order:
            y_name = f"climate_{emotion}_score_freq"
            # DV standardized using full-sample mean/std
            y_death = (full[y_name] - full_mean[y_name]) / full_std[y_name]
            X_mat_death = full[independent_vars_death_for_emotion].copy()
            # Only standardize variables in standardize_vars_death
            for c in independent_vars_death_for_emotion:
                if c in standardize_vars_death:
                    if c == "US_daily_covid_death":
                        X_mat_death[c] = (full[c] - full_mean_death) / full_std_death
                    else:
                        X_mat_death[c] = (full[c] - full_mean[c]) / full_std[c]
            X_death = add_constant(X_mat_death)
            model_death = OLS(y_death, X_death).fit()

            # Add one row per (emotion DV) in death version; suffix DV name with "_death"
            row = {
                "dependent_var": f"{y_name}_death",
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

    # Significance stars by p-value: * p≤0.05; ** p≤0.01; *** p≤0.001
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

    # Long-form regression summary: coef/se/t/p + stars; plus model stats; includes US_daily_covid_death
    summary_rows = []
    for y_name, model in summaries:
        for var in ["const"] + output_independent_vars:
            p = model.pvalues.get(var, np.nan)
            summary_rows.append({
                "dependent_var": y_name,
                "variable": var,
                "coef": model.params.get(var, np.nan),
                "std_err": model.bse.get(var, np.nan),
                "t_value": model.tvalues.get(var, np.nan),
                "p_value": p,
                "sig": sig_star(p),
            })
        for var, coef in [("R_sq", model.rsquared), ("Adj_R_sq", model.rsquared_adj), ("F", model.fvalue), ("F_pval", model.f_pvalue), ("N", int(model.nobs))]:
            summary_rows.append({"dependent_var": y_name, "variable": var, "coef": coef, "std_err": np.nan, "t_value": np.nan, "p_value": np.nan, "sig": ""})
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"Wrote: {OUTPUT_SUMMARY_CSV}, {OUTPUT_COEFFS_CSV}")

    # Font fallback list (kept; harmless even if plots are in English)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # Forest plot: x-axis = 8 emotion DVs, y-axis = coefficient of US_daily_covid_confirm (point + 95% CI + stars)
    # Emotion order: sadness, anger, fear, disgust, surprise, joy, trust, anticipation
    emotion_order = ["sadness", "anger", "fear", "disgust", "surprise", "joy", "trust", "anticipation"]
    var_focus = "US_daily_covid_confirm"
    model_by_y = {yn: m for yn, m in summaries}

    # Extract key coefficients/p-values/95% CIs for climate_post_model_ln and climate_post_model_ln_death
    # 1) US_daily_covid_confirm in climate_post_model_ln
    if "climate_post_model_ln" in model_by_y:
        m_ln = model_by_y["climate_post_model_ln"]
        ci_ln = m_ln.conf_int(alpha=0.05)
        if var_focus in ci_ln.index:
            lo, hi = ci_ln.loc[var_focus, 0], ci_ln.loc[var_focus, 1]
        else:
            lo, hi = np.nan, np.nan
        key_effect_rows.append({
            "model": "climate_post_model_ln",
            "dependent_var": "climate_post",
            "independent_var": var_focus,
            "coef": m_ln.params.get(var_focus, np.nan),
            "p_value": m_ln.pvalues.get(var_focus, np.nan),
            "CI_low": lo,
            "CI_high": hi,
        })
    # 2) US_daily_covid_death in climate_post_model_ln_death
    var_focus_death = "US_daily_covid_death"
    if "climate_post_model_ln_death" in model_by_y:
        m_ln_death = model_by_y["climate_post_model_ln_death"]
        ci_ln_death = m_ln_death.conf_int(alpha=0.05)
        if var_focus_death in ci_ln_death.index:
            lo_d, hi_d = ci_ln_death.loc[var_focus_death, 0], ci_ln_death.loc[var_focus_death, 1]
        else:
            lo_d, hi_d = np.nan, np.nan
        key_effect_rows.append({
            "model": "climate_post_model_ln_death",
            "dependent_var": "climate_post",
            "independent_var": var_focus_death,
            "coef": m_ln_death.params.get(var_focus_death, np.nan),
            "p_value": m_ln_death.pvalues.get(var_focus_death, np.nan),
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
            # Label point estimate to the right of each dot
            if np.isfinite(coefs_arr[i]):
                ax.text(x[i] + 0.1, coefs_arr[i], f"{coefs_arr[i]:.3f}", 
                       ha="left", va="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        # Reduce horizontal margins by tightening x-limits
        ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
        ax.set_ylabel("Δclimate_emotions")
        ax.set_xlabel("")
        ax.set_title("COVID-19 cases")
        # Keep only bottom and left spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        outpath = os.path.join(SCRIPT_DIR, "forest_by_dv.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")

    # Second forest plot: replace US_daily_covid_confirm with US_daily_covid_death; title "COVID-19 death"
    if "US_daily_covid_death" in full.columns:
        # Build death-based emotion models (temporary; for plotting only)
        independent_vars_death_for_emotion = [v if v != "US_daily_covid_confirm" else "US_daily_covid_death" for v in INDEPENDENT_VARS] + WEEKDAY_DUMMIES
        # Update STANDARDIZE_VARS (confirm -> death)
        standardize_vars_death = [v if v != "US_daily_covid_confirm" else "US_daily_covid_death" for v in STANDARDIZE_VARS]
        # Mean/std for death (not in full_mean; compute separately)
        full_mean_death = full["US_daily_covid_death"].mean()
        full_std_death = full["US_daily_covid_death"].std()
        
        coefs_death, ci_lo_death, ci_hi_death, labels_death, pvals_death = [], [], [], [], []
        # Same emotion order: sadness, anger, fear, disgust, surprise, joy, trust, anticipation
        for emotion in emotion_order:
            y_name = f"climate_{emotion}_score_freq"
            # Build a temporary death-based emotion model
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
        # Record US_daily_covid_death coefficient and 95% CI for each emotion model
            key_effect_rows.append({
                "model": "emotion_death",
                "dependent_var": y_name,
                "independent_var": var_focus_death,
                "coef": b_death,
                "p_value": pv_death,
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
                # Label point estimate to the right of each dot
                if np.isfinite(coefs_arr_death[i]):
                    ax_death.text(x_death[i] + 0.1, coefs_arr_death[i], f"{coefs_arr_death[i]:.3f}", 
                           ha="left", va="center", fontsize=9)
            ax_death.set_xticks(x_death)
            ax_death.set_xticklabels(labels_death, rotation=0, ha="center")
            # Reduce horizontal margins by tightening x-limits
            ax_death.set_xlim(x_death[0] - 0.3, x_death[-1] + 0.3)
            ax_death.set_ylabel("Δclimate_emotions")
            ax_death.set_xlabel("")
            ax_death.set_title("COVID-19 death")
            # Keep only bottom and left spines
            ax_death.spines['top'].set_visible(False)
            ax_death.spines['right'].set_visible(False)
            fig_death.tight_layout()
            outpath_death = os.path.join(SCRIPT_DIR, "forest_by_dv_death.png")
            fig_death.savefig(outpath_death, dpi=150, bbox_inches="tight")
            plt.close(fig_death)
            print(f"  Saved: {outpath_death}")

    # ========= Table-style CSV for the first two post-volume models =========
    # Model 1: log(US daily COVID-19 deaths) (climate_post_model_ln_death)
    # Model 2: log(US daily COVID-19 confirmed cases) (climate_post_model_ln)
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

        # Variable display-name mapping (fallback to original column name)
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

        # Regressor ordering for the table
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

        # Regressor rows: 2 lines per variable (coef row + SE row)
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

        # Bottom stats: N, R-squared, Adjusted R-squared
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
        print(f"Wrote: {output_table_csv}")

    # ========= Emotion-model tables: deaths vs cases =========
    # Create a CSV similar to above, with an extra column to identify the emotion DV
    if "US_daily_covid_death" in full.columns:
        # Reuse the same variable ordering
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
            # Prefix with a single quote to avoid Excel treating '*' as a formula operator
            return f"'{coef:.4f}{star}\n({se_str})"

        # Regressors for death version (confirm -> death)
        independent_vars_death_for_emotion = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in INDEPENDENT_VARS
        ] + WEEKDAY_DUMMIES
        # Standardized variables list (confirm -> death)
        standardize_vars_death = [
            v if v != "US_daily_covid_confirm" else "US_daily_covid_death"
            for v in STANDARDIZE_VARS
        ]
        full_mean_death = full["US_daily_covid_death"].mean()
        full_std_death = full["US_daily_covid_death"].std()

        # Emotion DV order (8 emotions)
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
        # Save each emotion's death model to build an 8-emotion wide table later
        death_models_by_emotion = {}
        for emotion in emotion_order:
            y_name = f"climate_{emotion}_score_freq"
            # Cases model (already in summaries)
            m_cases = model_by_y.get(y_name)
            if m_cases is None:
                continue
            # Deaths model: consistent with the forest-plot specification above
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
            # Regressor rows
            for v in var_order_for_table:
                row = {
                    "Dependent Variable": dep_label,
                    "Independent Variable": var_label_map.get(v, v),
                    "Model 1 (COVID-19 Deaths)": format_coef_se(m_death, v),
                    "Model 2 (COVID-19 Cases)": format_coef_se(m_cases, v),
                }
                table_rows_emotion.append(row)

            # Add 3 stats rows after each emotion model
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
            print(f"Wrote: {output_emotion_table_csv}")

        # Using only the "deaths" emotion models, generate a wide table:
        # rows = regressors; columns = 8 emotions; cells = coef+stars + newline + (SE)
        if death_models_by_emotion:
            wide_rows = []
            # Column order: Independent Variable + 8 emotions
            wide_columns = ["Independent Variable"] + [emotion_label_map[e] for e in emotion_order]
            # Only output regressors actually used in the deaths models (exclude confirm)
            var_order_death_only = (
                ["US_daily_covid_death"]
                + [v for v in INDEPENDENT_VARS if v not in ("US_daily_covid_confirm",)]
                + WEEKDAY_DUMMIES
            )
            for v in var_order_death_only:
                # First row: coefficient (with stars); second row: standard error (in parentheses)
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

            # Append 3 overall stats rows: N, R-squared, Adjusted R-squared (one column per emotion)
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
                print(f"Wrote: {output_emotion_death_only_csv}")
            except PermissionError:
                # The file may be open in Excel; write to an alternative filename
                alt_path = os.path.join(SCRIPT_DIR, "climate_emotions_death_only_wide_v2.csv")
                wide_df.to_csv(alt_path, index=False, encoding="utf-8-sig")
                print(f"File in use; wrote instead: {alt_path}")

    # 4 post-volume models -> a combined 2x2 forest plot
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
                # Label the point estimate above the dot (due to invert_yaxis, "above" means smaller y)
                if np.isfinite(coefs_arr[i]):
                    ax.text(coefs_arr[i], y_pos[i] - y_range * 0.04, f"{coefs_arr[i]:.3f}", 
                           ha="center", va="top", fontsize=7)
                star = sig_star(pvals[i])
                if star:
                    ax.text(x_max + x_range * 0.02, y_pos[i], star, ha="left", va="center", fontsize=9, fontweight="bold")
            x_lo = min(x_min - 0.02 * x_range, 0)
            x_hi = max(x_max + 0.12 * x_range, 0)
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel("Coefficient (95% CI)")
            ax.set_title(mname, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        outpath_4 = os.path.join(SCRIPT_DIR, "forest_climate_post_2x2.png")
        fig.savefig(outpath_4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath_4}")

    # Export key coefficients and 95% CIs
    if key_effect_rows:
        pd.DataFrame(key_effect_rows).to_csv(OUTPUT_KEY_CI_CSV, index=False, encoding="utf-8-sig")
        print(f"Wrote: {OUTPUT_KEY_CI_CSV}")

    print("Dependent variables and R2 summary:")
    for y_name, model in summaries:
        print(f"  {y_name}: R2={model.rsquared:.4f}, Adj_R2={model.rsquared_adj:.4f}, N={int(model.nobs)}")


if __name__ == "__main__":
    main()
