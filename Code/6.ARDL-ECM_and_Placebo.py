import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to ensure images can be saved in server/command line environment
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Configure Chinese font support
try:
    import platform
    system = platform.system()
    if system == 'Windows':
        # Windows: prefer Microsoft YaHei, then SimHei and others
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'sans-serif']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC', 'sans-serif']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
except Exception:
    # If detection fails, fall back to generic font settings
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering issue
from scipy.stats import pearsonr, norm, t, f


def discover_emotion_columns(columns: List[str]) -> List[str]:
    """
    Discover emotion columns from dataset.

    Expected naming: climate_{emotion}_score_freq and covid_{emotion}_score_freq
    Returns list of all emotion column names
    """
    climate_prefix = "climate_"
    covid_prefix = "covid_"
    suffix = "_score_freq"

    emotion_columns = []

    for col in columns:
        if (col.startswith(climate_prefix) or col.startswith(covid_prefix)) and col.endswith(suffix):
            emotion_columns.append(col)

    return sorted(emotion_columns)


def discover_exogenous_columns(columns: List[str]) -> List[str]:
    """
    Discover exogenous variable columns from dataset.
    
    Expected columns: US_daily_covid_death, debates, climatenews
    Also includes extreme weather variables (columns AI to BX) as dummy variables
    Returns list of exogenous variable names
    """
    exogenous_columns = []
    expected_exog = ["US_daily_covid_death", "debates", "climatenews", "GovernmentResponseIndex_Average"]
    
    # Add standard exogenous variables
    for col in columns:
        if col in expected_exog:
            exogenous_columns.append(col)
    
    # Add extreme weather variables (merged into 6 main categories)
    # These are dummy variables for extreme weather events
    extreme_weather_columns = [
        'WinterStorm', 'Wildfire', 'TropicalCyclone', 'SevereStorm', 'Flood', 'Drought'
    ]
    
    for col in columns:
        if col in extreme_weather_columns:
            exogenous_columns.append(col)
    
    return sorted(exogenous_columns)


def _single_level_tests(values: np.ndarray) -> Dict[str, float]:
    """Run ADF on a numeric array and return p-values and flags.

    Returns keys: adf_p
    """
    res: Dict[str, float] = {"adf_p": float("nan")}
    try:
        res["adf_p"] = float(adfuller(values, autolag="AIC")[1])
    except Exception:
        pass
    return res


def judge_stationary(pvals: Dict[str, float], adf_alpha: float = 0.05) -> bool:
    """Heuristic stationarity decision using ADF.

    Stationary if ADF rejects unit root at adf_alpha.
    Missing tests are ignored conservatively.
    """
    adf_ok = (pd.notna(pvals.get("adf_p")) and pvals["adf_p"] < adf_alpha)
    return bool(adf_ok)


def find_integration_order(series: pd.Series, series_name: str, max_d: int = 2) -> Tuple[int, pd.Series, Dict[str, object]]:
    """Determine integration order up to max_d and return (d, stationary_series, report_row).

    Perform unit root tests directly on the raw series; return the series differenced d times,
    aligned with the original index (with NaN padding at the beginning).
    """
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    report: Dict[str, object] = {"series": series_name}
    if cleaned.size < 20:
        report.update({"final_order": np.nan, "decision": "insufficient"})
        return 0, series, report

    current = cleaned.copy()
    order_found = None
    all_pvals: List[Tuple[int, Dict[str, float]]] = []
    for d in range(0, max_d + 1):
        pvals = _single_level_tests(current.values)
        all_pvals.append((d, pvals))
        if judge_stationary(pvals):
            order_found = d
            break
        # prepare next difference if not reached max_d
        if d < max_d:
            current = np.diff(current.values)
            current = pd.Series(current, index=cleaned.index[d+1:])

    # Build report
    for d, pv in all_pvals:
        for k, v in pv.items():
            report[f"{k}_d{d}"] = v
    if order_found is None:
        order_found = max_d + 1  # greater than tested range
        decision = f">I({max_d})?"
    else:
        decision = f"I({order_found})"
    report["final_order"] = int(order_found) if pd.notna(order_found) else np.nan
    report["decision"] = decision

    # Construct differenced series of length aligned to original index (with NaN padding at start)
    diffed = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if order_found >= 1 and order_found <= max_d:
        diffed = diffed.diff(order_found)
    return order_found if isinstance(order_found, int) else int(max_d), diffed, report


def remove_weekday_effect(series: pd.Series, dates: pd.Series, add_trend: bool, polynomial_degree: int = 1) -> pd.Series:
    """Regress out weekday fixed effects (and optional polynomial trend). Return residuals aligned to original index.

    Args:
        add_trend: whether to include a time trend
        polynomial_degree: degree of polynomial trend (1 = linear, 2 = quadratic, and so on)
    """
    y = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    wday = pd.to_datetime(dates, errors="coerce").dt.dayofweek
    x = pd.get_dummies(wday, prefix="wd", drop_first=True)
    if add_trend:
        t = pd.Series(np.arange(len(y)), index=y.index, name="trend")
        # Add polynomial trend terms: trend, trend^2, ..., trend^k
        cols = {"trend": t}
        try:
            deg = int(polynomial_degree) if polynomial_degree is not None else 1
        except Exception:
            deg = 1
        deg = max(1, deg)
        if deg >= 2:
            for d in range(2, deg + 1):
                cols[f"trend^{d}"] = t ** d
        x = pd.concat([x] + [cols[c] for c in cols], axis=1)
    x = sm.add_constant(x, has_constant="add")
    df_design = pd.concat([y.rename("y"), x], axis=1).dropna()
    if df_design.shape[0] < 10:
        return y  # not enough data; return original
    # Ensure numeric float arrays for statsmodels
    y_arr = df_design["y"].astype(float).values
    X_df = df_design.drop(columns=["y"]).astype(float)
    model = sm.OLS(y_arr, X_df.values)
    try:
        res = model.fit()
        resid = pd.Series(index=df_design.index, data=res.resid)
        # align back to original index with NaN where missing, then forward/back fill to keep continuity if desired
        out = resid.reindex(y.index)
        return out
    except Exception:
        return y


def build_weekday_dummies(dates: pd.Series, index: pd.Index = None, prefix: str = "wd") -> pd.DataFrame:
    """Construct weekday dummy variables (Tue–Sun), with Monday as the baseline (drop_first=True).

    Convention: pandas dayofweek Monday=0 ... Sunday=6, so output columns are typically wd_1..wd_6.
    If some weekdays do not appear in the sample, add the missing columns and fill them with 0
    to keep downstream model columns stable.
    """
    dt = pd.to_datetime(dates, errors="coerce")
    if index is not None:
        dt = dt.reindex(index)
    wday = dt.dt.dayofweek
    dummies = pd.get_dummies(wday, prefix=prefix, drop_first=True)
    # Ensure Tue..Sun (1..6) dummy columns exist
    for k in range(1, 7):
        col = f"{prefix}_{k}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"{prefix}_{k}" for k in range(1, 7)]]
    return dummies.astype(float)


def export_weekday_ols_report(series: pd.Series, dates: pd.Series, series_name: str, add_trend: bool, csv_dir: str, polynomial_degree: int = 1):
    """
    Regress a single series on weekday fixed effects and an optional time trend, and export a coefficient report.
    Columns: term, coefficient, pvalue, star; with the last two rows being r_squared and aic.
    """
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception:
        pass
    y = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    wday = pd.to_datetime(dates, errors="coerce").dt.dayofweek
    X = pd.get_dummies(wday, prefix="wd", drop_first=True)
    if add_trend:
        t = pd.Series(np.arange(len(y)), index=y.index, name="trend")
        cols = {"trend": t}
        try:
            deg = int(polynomial_degree) if polynomial_degree is not None else 1
        except Exception:
            deg = 1
        deg = max(1, deg)
        if deg >= 2:
            for d in range(2, deg + 1):
                cols[f"trend^{d}"] = t ** d
        X = pd.concat([X] + [cols[c] for c in cols], axis=1)
    X = sm.add_constant(X, has_constant="add")
    df_design = pd.concat([y.rename("y"), X], axis=1).dropna()
    if df_design.shape[0] < 10:
        return
    y_arr = df_design["y"].astype(float).values
    X_df = df_design.drop(columns=["y"]).astype(float)
    try:
        res = sm.OLS(y_arr, X_df.values).fit()
        terms = ["const"] + [c for c in X_df.columns if c != "const"]
        coefs = list(res.params)
        pvals = list(res.pvalues)
        def _star(p):
            try:
                p = float(p)
            except Exception:
                return ""
            if np.isnan(p):
                return ""
            if p < 0.01:
                return "***"
            if p < 0.05:
                return "**"
            if p < 0.10:
                return "*"
            return ""
        rows = []
        for t, b, p in zip(terms, coefs, pvals):
            rows.append({
                "term": t,
                "coefficient": float(b) if pd.notna(b) else np.nan,
                "pvalue": float(p) if pd.notna(p) else np.nan,
                "star": _star(p),
            })
        rows.append({"term": "r_squared", "coefficient": float(res.rsquared), "pvalue": np.nan, "star": ""})
        rows.append({"term": "aic", "coefficient": float(res.aic), "pvalue": np.nan, "star": ""})
        df_out = pd.DataFrame(rows, columns=["term", "coefficient", "pvalue", "star"]).copy()
        # Display format: coefficient in scientific notation with two significant digits; p-value with two decimals
        def _fmt_coef(v):
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
            except Exception:
                return ""
            return np.format_float_scientific(v, precision=2, unique=False, exp_digits=2)
        def _fmt_p(v):
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
            except Exception:
                return ""
            return f"{v:.2f}"
        df_out["coefficient"] = df_out["coefficient"].apply(_fmt_coef)
        df_out["pvalue"] = df_out["pvalue"].apply(_fmt_p)
        out_dir = os.path.join(csv_dir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        out_path = os.path.join(out_dir, f"weekday_ols_{str(series_name).replace('/', '_').replace('\\', '_')}.csv")
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def build_weekday_ols_wide(csv_dir: str, outfile: str = "weekday_ols_summary_wide.csv"):
    """Aggregate weekday_ols_* regression outputs into a wide-format table:
    - Columns: each series (20 columns)
    - Rows: one row for coefficients, next row for significance stars
    Only terms in [const, wd_*, trend] are kept; r_squared and aic are ignored.
    """
    try:
        if not os.path.exists(csv_dir):
            return
        files = []
        try:
            files = [f for f in os.listdir(csv_dir) if f.startswith("weekday_ols_") and f.endswith(".csv")]
        except Exception:
            files = []
        if not files:
            return
        series_names: List[str] = []
        series_to_df: Dict[str, pd.DataFrame] = {}
        for fn in sorted(files):
            path = os.path.join(csv_dir, fn)
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except Exception:
                continue
            # Parse series name
            base = os.path.splitext(fn)[0]  # weekday_ols_xxx
            series_name = base[len("weekday_ols_"):]
            series_names.append(series_name)
            # Keep coefficient rows only (exclude r_squared / aic)
            if "term" in df.columns:
                df = df[(df["term"].astype(str) != "r_squared") & (df["term"].astype(str) != "aic")].copy()
            else:
                continue
            # Required columns
            for col in ["coefficient", "pvalue", "star"]:
                if col not in df.columns:
                    df[col] = np.nan
            # Coefficient format: scientific notation with two significant digits; p-value with two decimals
            def _fmt_coef(v):
                try:
                    if pd.isna(v):
                        return ""
                    v = float(v)
                except Exception:
                    return ""
                return np.format_float_scientific(v, precision=2, unique=False, exp_digits=2)
            def _fmt_p(v):
                try:
                    if pd.isna(v):
                        return ""
                    v = float(v)
                except Exception:
                    return ""
                return f"{v:.2f}"
            df["coefficient"] = df["coefficient"].apply(_fmt_coef)
            df["pvalue"] = df["pvalue"].apply(_fmt_p)
            # Keep required columns only
            series_to_df[series_name] = df[["term", "coefficient", "star"]].copy()

        if not series_to_df:
            return
        # Collect all terms (deduplicate by first appearance order)
        all_terms: List[str] = []
        for s in series_names:
            df = series_to_df.get(s)
            if df is None:
                continue
            for t in df["term"].astype(str).tolist():
                if t not in all_terms:
                    all_terms.append(t)
        if not all_terms:
            return
        # Build wide table: two rows per variable (variable, variable_star)
        rows: List[Dict[str, object]] = []
        header = ["variable"] + series_names
        for term in all_terms:
            row_coef: Dict[str, object] = {"variable": term}
            row_star: Dict[str, object] = {"variable": f"{term}_star"}
            for s in series_names:
                df = series_to_df.get(s)
                if df is None:
                    row_coef[s] = ""
                    row_star[s] = ""
                    continue
                sub = df[df["term"].astype(str) == str(term)]
                if not sub.empty:
                    row_coef[s] = sub.iloc[-1]["coefficient"] if pd.notna(sub.iloc[-1]["coefficient"]) else ""
                    row_star[s] = sub.iloc[-1]["star"] if ("star" in sub.columns and pd.notna(sub.iloc[-1]["star"])) else ""
                else:
                    row_coef[s] = ""
                    row_star[s] = ""
            rows.append(row_coef)
            rows.append(row_star)
        df_wide = pd.DataFrame(rows, columns=header)
        out_path = os.path.join(csv_dir, outfile)
        try:
            df_wide.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    except Exception:
        pass

def cointegration_test(series1: pd.Series, series2: pd.Series, series1_name: str, series2_name: str, dates: pd.Series = None, weekday_ols: bool = False, weekday_trend: bool = False, weekday_poly_degree: int = 1) -> Dict[str, object]:
    """
    Run cointegration tests for two time series.

    Args:
        series1, series2: raw time series
        series1_name, series2_name: series names
        dates: date series used for optional preprocessing
        weekday_ols: whether to remove weekday effects
        weekday_trend: whether to remove a linear trend during preprocessing

    Returns:
        Dict containing cointegration test results
    """
    # Clean inputs
    s1 = pd.to_numeric(series1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s2 = pd.to_numeric(series2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # Optional preprocessing (weekday effects and trend)
    if weekday_ols and dates is not None:
        # Align dates and series
        s1_dates = dates.loc[s1.index]
        s2_dates = dates.loc[s2.index]
        
        # Remove weekday effects and trend
        s1 = remove_weekday_effect(s1, s1_dates, add_trend=weekday_trend, polynomial_degree=weekday_poly_degree)
        s2 = remove_weekday_effect(s2, s2_dates, add_trend=weekday_trend, polynomial_degree=weekday_poly_degree)
        
        # Re-clean after preprocessing
        s1 = pd.to_numeric(s1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s2 = pd.to_numeric(s2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # Align indices
    common_index = s1.index.intersection(s2.index)
    if len(common_index) < 20:
        return {
            "series1": series1_name,
            "series2": series2_name,
            "coint_stat": np.nan,
            "coint_pvalue": np.nan,
            "coint_result": "insufficient_data",
            "johansen_trace_stat": np.nan,
            "johansen_trace_pvalue": np.nan,
            "johansen_max_eigen_stat": np.nan,
            "johansen_max_eigen_pvalue": np.nan,
            "johansen_result": "insufficient_data"
        }
    
    s1_aligned = s1.loc[common_index]
    s2_aligned = s2.loc[common_index]
    
    result = {
        "series1": series1_name,
        "series2": series2_name,
        "coint_stat": np.nan,
        "coint_pvalue": np.nan,
        "coint_result": "failed",
        "johansen_trace_stat": np.nan,
        "johansen_trace_pvalue": np.nan,
        "johansen_max_eigen_stat": np.nan,
        "johansen_max_eigen_pvalue": np.nan,
        "johansen_result": "failed"
    }
    
    try:
        # Engle-Granger cointegration test
        coint_stat, coint_pvalue, _ = coint(s1_aligned, s2_aligned)
        result["coint_stat"] = coint_stat
        result["coint_pvalue"] = coint_pvalue
        result["coint_result"] = "cointegrated" if coint_pvalue < 0.05 else "not_cointegrated"
    except Exception as e:
        result["coint_result"] = f"error: {str(e)[:50]}"
    
    try:
        # Johansen cointegration test
        data = np.column_stack([s1_aligned, s2_aligned])
        johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        result["johansen_trace_stat"] = johansen_result.lr1[0]  # trace statistic
        result["johansen_trace_pvalue"] = johansen_result.cvt[0, 1]  # trace critical value (approx proxy)
        result["johansen_max_eigen_stat"] = johansen_result.lr2[0]  # max-eigen statistic
        result["johansen_max_eigen_pvalue"] = johansen_result.cve[0, 1] if hasattr(johansen_result, 'cve') else np.nan  # max-eigen critical value (approx proxy)
        
        # Decide cointegration by trace statistic vs critical value
        if johansen_result.lr1[0] > johansen_result.cvt[0, 1]:
            result["johansen_result"] = "cointegrated"
        else:
            result["johansen_result"] = "not_cointegrated"
            
    except Exception as e:
        result["johansen_result"] = f"error: {str(e)[:50]}"
    
    return result


def select_optimal_lags(data: np.ndarray, max_lags: int = 6) -> int:
    """
    Select the optimal lag order using information criteria.

    Args:
        data: time-series data matrix
        max_lags: maximum lag to consider

    Returns:
        selected lag order (fallback to 1 on failure)
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    
    try:
        model = VAR(data)
        lag_order = model.select_order(maxlags=max_lags)
        
        # Prefer AIC; fall back to BIC
        if hasattr(lag_order, 'aic') and not np.isnan(lag_order.aic):
            return lag_order.aic
        elif hasattr(lag_order, 'bic') and not np.isnan(lag_order.bic):
            return lag_order.bic
        else:
            return 1  # default
    except:
        return 1  # fallback


# VEC estimation function removed (current dataset does not contain I(1)+I(1) combinations)


def ardl_ecm_model(series1: pd.Series, series2: pd.Series, series1_name: str, series2_name: str, max_lags: int = 6, dates: pd.Series = None, weekday_ols: bool = False, weekday_trend: bool = False, exog_vars: Dict[str, pd.Series] = None, exog_significant_only: bool = False, exog_threshold: float = 0.10) -> Dict[str, object]:
    """
    Estimate an ARDL-ECM model for two time series.

    Args:
        series1, series2: raw time series
        series1_name, series2_name: series names
        max_lags: maximum lag order to consider
        dates: date series used for optional preprocessing
        weekday_ols: whether to remove weekday effects
        weekday_trend: whether to remove a linear trend during preprocessing
        exog_vars: dict of exogenous series
        exog_significant_only: keep only significant exogenous variables
        exog_threshold: significance threshold for exogenous variables

    Returns:
        Dict containing ARDL-ECM model results
    """
    # Clean inputs
    s1 = pd.to_numeric(series1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s2 = pd.to_numeric(series2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # Optional preprocessing (weekday effects and trend)
    if weekday_ols and dates is not None:
        # Align dates and series
        s1_dates = dates.loc[s1.index]
        s2_dates = dates.loc[s2.index]
        
        # Remove weekday effects and trend
        s1 = remove_weekday_effect(s1, s1_dates, add_trend=weekday_trend)
        s2 = remove_weekday_effect(s2, s2_dates, add_trend=weekday_trend)
        
        # Re-clean after preprocessing
        s1 = pd.to_numeric(s1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s2 = pd.to_numeric(s2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # Align indices
    common_index = s1.index.intersection(s2.index)
    if len(common_index) < 30:
        return {
            "series1": series1_name,
            "series2": series2_name,
            "error": "insufficient_data",
            "ecm_coefficient": np.nan,
            "ecm_pvalue": np.nan,
            "short_run_coefficients": {},
            "long_run_coefficient": np.nan,
            "long_run_pvalue": np.nan,
            "long_run_std_error": np.nan,
            "long_run_tstat": np.nan,
            "long_run_ci_lower": np.nan,
            "long_run_ci_upper": np.nan,
            "r_squared": np.nan,
            "aic": np.nan
        }
    
    s1_aligned = s1.loc[common_index]
    s2_aligned = s2.loc[common_index]
    
    result = {
        "series1": series1_name,
        "series2": series2_name,
        "error": None,
        "ecm_coefficient": np.nan,
        "ecm_pvalue": np.nan,
        "short_run_coefficients": {},
        "long_run_coefficient": np.nan,
        "long_run_pvalue": np.nan,
        "long_run_std_error": np.nan,
        "long_run_tstat": np.nan,
        "long_run_ci_lower": np.nan,
        "long_run_ci_upper": np.nan,
        "exog_coefficients": {},
        "r_squared": np.nan,
        "aic": np.nan
    }
    
    try:
        n = len(s1_aligned)
        if n < 20:
            result["error"] = "insufficient_observations"
            return result
        
        y = s1_aligned.values
        x = s2_aligned.values
        
        # Grid search ARDL-ECM(p,q) (p=1..6, q=0..6), pick best by AIC; use HAC robust SE when available
        # Improvement: prefer models with lower residual serial correlation (LBQ) when possible
        # Use higher max lags to better capture dynamics and reduce residual autocorrelation
        max_p, max_q = 6, 6
        best_aic = np.inf
        best_fit = None
        best_design = None
        best_meta = None
        dy_full = np.diff(y)
        dx_full = np.diff(x)
        # Preprocess exogenous variables into a matrix
        exog_matrix = None
        exog_names_order = []
        if exog_vars is not None and len(exog_vars) > 0:
            exog_stack = []
            for exog_name, exog_series in exog_vars.items():
                ex_aligned = exog_series.loc[common_index]
                ex_clean = pd.to_numeric(ex_aligned, errors="coerce").replace([np.inf, -np.inf], np.nan)
                exog_stack.append(ex_clean.values)
                exog_names_order.append(exog_name)
            try:
                exog_matrix = np.column_stack(exog_stack) if exog_stack else None
            except Exception:
                exog_matrix = None

        for p in range(1, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    start = max(1, p, q)
                    T = (len(y) - 1) - (start - 1)
                    if T < 10:
                        continue
                    dy = dy_full[start - 1: start - 1 + T]
                    cols = []
                    col_names = []
                    cols.append(np.ones(T)); col_names.append("const")
                    if p > 1:
                        for i in range(1, p):
                            cols.append(dy_full[start - 1 - i: start - 1 - i + T])
                            col_names.append(f"dy_lag{i}")
                    # Δx current and lags (if q=0, include current first-difference only)
                    q_use = max(1, q)
                    for j in range(0, q_use):
                        cols.append(dx_full[start - 1 - j: start - 1 - j + T])
                        col_names.append(f"dx_lag{j}")
                    # y_{t-1}, x_{t-1}
                    cols.append(y[start - 1: start - 1 + T]); col_names.append("y_lag")
                    cols.append(x[start - 1: start - 1 + T]); col_names.append("x_lag")
                    # Exogenous variables (aligned with t)
                    if exog_matrix is not None:
                        try:
                            ex_block = exog_matrix[start: start + T, :]
                            if ex_block.shape[0] == T:
                                for k in range(ex_block.shape[1]):
                                    cols.append(ex_block[:, k])
                                    col_names.append(str(exog_names_order[k]))
                        except Exception:
                            pass
                    X_try = np.column_stack(cols)
                    model_try = OLS(dy, X_try)
                    try:
                        hac_lag = max(1, min(10, int(T // 4)))
                        fit_try = model_try.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lag, 'use_correction': True})
                    except Exception:
                        fit_try = model_try.fit()
                    aic_val = float(fit_try.aic)

                    # Lag selection: AIC only (smaller is better)
                    if np.isfinite(aic_val) and (best_fit is None or aic_val < best_aic):
                        best_aic = aic_val
                        best_fit = fit_try
                        best_design = (dy, X_try)
                        best_meta = {"p": p, "q": q, "col_names": col_names}
                except Exception:
                    continue

        if best_fit is None:
            raise RuntimeError("ARDL-ECM lag selection failed")
        fitted_model = best_fit
        dy, X = best_design
        col_names = best_meta.get("col_names", [])
        
        # Extract results
        params = fitted_model.params
        pvalues = fitted_model.pvalues
        # Record selected lag orders
        result["selected_p"] = best_meta.get("p", np.nan)
        result["selected_q"] = best_meta.get("q", np.nan)
        
        # ECM coefficient (coefficient on y_{t-1}; typically negative)
        try:
            ylag_idx = col_names.index("y_lag")
        except Exception:
            ylag_idx = None
        if ylag_idx is not None and ylag_idx < len(params):
            result["ecm_coefficient"] = params[ylag_idx]
            result["ecm_pvalue"] = pvalues[ylag_idx] if ylag_idx < len(pvalues) else np.nan
        
        # Short-run coefficients: extract dy/dx lag terms
        # Get selected p and q
        selected_p = best_meta.get("p", 1)
        selected_q = best_meta.get("q", 0)
        
        # Extract dy_lag{i} coefficients (dy_lag1 .. dy_lag{p-1})
        for i in range(1, selected_p):
            lag_name = f"dy_lag{i}"
            try:
                lag_idx = col_names.index(lag_name)
                if lag_idx < len(params):
                    result["short_run_coefficients"][f"dy_lag{i}"] = params[lag_idx]
                    result["short_run_coefficients"][f"dy_lag{i}_pvalue"] = pvalues[lag_idx] if lag_idx < len(pvalues) else np.nan
            except ValueError:
                # Lag term not present; skip
                pass
        
        # Extract dx_lag{j} coefficients (dx_lag0 .. dx_lag{q-1}), where q_use = max(1, q)
        q_use = max(1, selected_q)
        for j in range(0, q_use):
            lag_name = f"dx_lag{j}"
            try:
                lag_idx = col_names.index(lag_name)
                if lag_idx < len(params):
                    if j == 0:
                        # Backward compatibility: also store dx_lag0 as delta_x
                        result["short_run_coefficients"]["delta_x"] = params[lag_idx]
                        result["short_run_coefficients"]["delta_x_pvalue"] = pvalues[lag_idx] if lag_idx < len(pvalues) else np.nan
                    result["short_run_coefficients"][f"dx_lag{j}"] = params[lag_idx]
                    result["short_run_coefficients"][f"dx_lag{j}_pvalue"] = pvalues[lag_idx] if lag_idx < len(pvalues) else np.nan
            except ValueError:
                # Lag term not present; skip
                pass
        
        # If no dx_lag terms were found, set defaults (backward compatibility)
        if "delta_x" not in result["short_run_coefficients"]:
            result["short_run_coefficients"]["delta_x"] = np.nan
            result["short_run_coefficients"]["delta_x_pvalue"] = np.nan
        
        # Long-run coefficient (computed via ECM term; Delta-method SE)
        try:
            xlag_idx = col_names.index("x_lag")
        except Exception:
            xlag_idx = None
        if ylag_idx is not None and xlag_idx is not None and ylag_idx < len(params) and xlag_idx < len(params):
            if abs(params[ylag_idx]) > 1e-10:
                # Long-run coefficient = -params[x_lag] / params[y_lag]
                long_run_coef = -params[xlag_idx] / params[ylag_idx]
                result["long_run_coefficient"] = long_run_coef
                
                # Delta-method SE for the long-run coefficient
                try:
                    # Covariance matrix
                    cov_matrix = fitted_model.cov_params()
                    
                    # Relevant parameters
                    a = params[ylag_idx]  # y_lag coefficient
                    d = params[xlag_idx]  # x_lag coefficient
                    
                    # Variances and covariance
                    var_a = cov_matrix.iloc[ylag_idx, ylag_idx] if hasattr(cov_matrix, 'iloc') else cov_matrix[ylag_idx, ylag_idx]
                    var_d = cov_matrix.iloc[xlag_idx, xlag_idx] if hasattr(cov_matrix, 'iloc') else cov_matrix[xlag_idx, xlag_idx]
                    
                    # Covariance (matrix is symmetric)
                    if hasattr(cov_matrix, 'iloc'):
                        cov_ad = cov_matrix.iloc[ylag_idx, xlag_idx]
                    else:
                        cov_ad = cov_matrix[ylag_idx, xlag_idx] if ylag_idx < cov_matrix.shape[0] and xlag_idx < cov_matrix.shape[1] else 0
                    
                    # Delta method: variance of the long-run coefficient
                    # long_run = -d/a
                    # var(long_run) = (1/a²) * var(d) + (d²/a⁴) * var(a) - 2*(d/a³) * cov(a,d)
                    var_long = (1 / (a**2)) * var_d + (d**2 / (a**4)) * var_a - 2 * (d / (a**3)) * cov_ad
                    
                    # Ensure non-negative variance
                    var_long = max(0, var_long)
                    se_long = np.sqrt(var_long)
                    
                    # t-stat and p-value (t distribution with df = n - k)
                    n_obs = len(dy)
                    k_params = len(params)
                    df = n_obs - k_params
                    
                    if se_long > 1e-10 and df > 0:
                        t_stat = long_run_coef / se_long
                        # Two-sided test
                        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
                        result["long_run_std_error"] = se_long
                        result["long_run_tstat"] = t_stat
                        result["long_run_pvalue"] = p_value
                        
                        # 95% CI
                        t_critical = t.ppf(0.975, df)
                        result["long_run_ci_lower"] = long_run_coef - t_critical * se_long
                        result["long_run_ci_upper"] = long_run_coef + t_critical * se_long
                    else:
                        # Fallback if SE is too small or df invalid
                        result["long_run_std_error"] = np.nan
                        result["long_run_tstat"] = np.nan
                        result["long_run_pvalue"] = pvalues[xlag_idx] if xlag_idx < len(pvalues) else np.nan
                        result["long_run_ci_lower"] = np.nan
                        result["long_run_ci_upper"] = np.nan
                        
                except Exception as e:
                    # Fallback if Delta method fails
                    result["long_run_std_error"] = np.nan
                    result["long_run_tstat"] = np.nan
                    result["long_run_pvalue"] = pvalues[xlag_idx] if xlag_idx < len(pvalues) else np.nan
                    result["long_run_ci_lower"] = np.nan
                    result["long_run_ci_upper"] = np.nan
        
        # Extract exogenous coefficients and p-values (from fitted results, if present)
        try:
            exog_coeffs_map = {}
            # Exogenous variables
            for name in exog_names_order:
                if name in col_names:
                    idx = col_names.index(name)
                    if idx < len(params):
                        exog_coeffs_map[str(name)] = {
                            "coefficient": float(params[idx]),
                            "pvalue": float(pvalues[idx]) if idx < len(pvalues) else float("nan"),
                        }
            # Constant (kept alongside exogenous variables in the design matrix)
            if "const" in col_names:
                cidx = col_names.index("const")
                if cidx < len(params):
                    exog_coeffs_map["const"] = {
                        "coefficient": float(params[cidx]),
                        "pvalue": float(pvalues[cidx]) if cidx < len(pvalues) else float("nan"),
                    }
            if len(exog_coeffs_map) > 0:
                result["exog_coefficients"] = exog_coeffs_map
        except Exception:
            pass

        result["r_squared"] = fitted_model.rsquared
        result["aic"] = fitted_model.aic

        # Save fitted object info for IRF/dynamic multiplier and CIs
        result["_fitted_params"] = {
            "params": params,
            "pvalues": fitted_model.pvalues,
            # Save covariance matrix and SEs for robust std_error retrieval later
            "cov": fitted_model.cov_params(),
            "bse": getattr(fitted_model, "bse", None),
            "design_cols": list(col_names),
        }
        # Minimal console output: selected lags and AIC
        try:
            p_val = int(result.get('selected_p'))
            q_val = int(result.get('selected_q'))
            aic_val = float(result.get('aic'))
            print(f"  [ECM lag selection] p={p_val}, q={q_val}, AIC={aic_val:.3f}")
        except Exception:
            pass
        
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


def check_causality_exists(ardl_result: Dict[str, object], ecm_result: Dict[str, object], alpha: float = 0.10) -> bool:
    """
    Check whether a causal relationship is indicated (heuristic).

    Criteria:
    1) ARDL bounds test passes (bounds_result == 'cointegrated') -> long-run cointegration exists
    2) ECM estimation succeeds (no 'error')
    3) ECM coefficient is significant (ecm_pvalue < alpha) -> core criterion

    Notes:
    - A significant ECM term indicates an error-correction mechanism (at least short-run adjustment)
    - A significant long-run coefficient is not strictly required here; it is auxiliary information

    Args:
        ardl_result: ARDL bounds test result
        ecm_result: ECM estimation result
        alpha: significance level (default 0.10)

    Returns:
        True if causality is indicated, else False
    """
    # Bounds test must pass
    if ardl_result.get('error') or ardl_result.get('bounds_result') != 'cointegrated':
        return False
    
    # ECM must run successfully
    if ecm_result.get('error'):
        return False
    
    # Core criterion: ECM term significance
    ecm_pvalue = ecm_result.get('ecm_pvalue', 1.0)
    if pd.isna(ecm_pvalue) or float(ecm_pvalue) >= alpha:
        return False
    
    # If ECM is significant, consider causality indicated
    return True


def format_causality_detail(ardl_result: Dict[str, object], ecm_result: Dict[str, object], alpha: float = 0.10) -> str:
    """
    Summarize short-run / long-run significance and direction based on ARDL/ECM results.

    Rules:
    - Short-run: use the contemporaneous Δx coefficient (delta_x) p-value (< alpha => present); sign gives direction
    - Long-run: use the long-run coefficient p-value (< alpha => present) and requires bounds_result == 'cointegrated'
    - Direction: coef > 0 => "positive", < 0 => "negative", == 0 => "zero"
    - Note: a significant ECM term indicates error correction, but does not guarantee both short-run and long-run direct effects are significant
    """
    try:
        # ECM significance (core criterion for "causality indicated")
        ecm_coef = ecm_result.get('ecm_coefficient', np.nan)
        ecm_p = ecm_result.get('ecm_pvalue', np.nan)
        ecm_significant = (not pd.isna(ecm_p)) and float(ecm_p) < alpha
        
        def pstars(p: float) -> str:
            try:
                pv = float(p)
            except Exception:
                return ""
            if pv < 0.01:
                return "***"
            if pv < 0.05:
                return "**"
            if pv < 0.10:
                return "*"
            return ""
        
        # Short-run
        sr_coef = None
        sr_p = None
        try:
            sr_coef = ecm_result.get('short_run_coefficients', {}).get('delta_x', np.nan)
            sr_p = ecm_result.get('short_run_coefficients', {}).get('delta_x_pvalue', np.nan)
        except Exception:
            sr_coef, sr_p = np.nan, np.nan
        sr_exist = (not pd.isna(sr_p)) and float(sr_p) < alpha
        if pd.isna(sr_coef):
            sr_dir = "unknown"
            sr_info = "unknown"
        else:
            sr_dir = "positive" if float(sr_coef) > 0 else ("negative" if float(sr_coef) < 0 else "zero")
            if pd.isna(sr_p):
                sr_info = f"{sr_dir} (missing p-value)"
            else:
                sr_info = f"{sr_dir} (p={float(sr_p):.4f}{pstars(sr_p)})"

        # Long-run (requires cointegrated)
        lr_coef = ecm_result.get('long_run_coefficient', np.nan)
        lr_p = ecm_result.get('long_run_pvalue', np.nan)
        lr_exist = (
            ardl_result.get('bounds_result') == 'cointegrated'
            and (not pd.isna(lr_p))
            and float(lr_p) < alpha
        )
        if pd.isna(lr_coef):
            lr_dir = "unknown"
            lr_info = "unknown"
        else:
            lr_dir = "positive" if float(lr_coef) > 0 else ("negative" if float(lr_coef) < 0 else "zero")
            if pd.isna(lr_p):
                lr_info = f"{lr_dir} (missing p-value)"
            else:
                lr_info = f"{lr_dir} (p={float(lr_p):.4f}{pstars(lr_p)})"

        # ECM direction (informational)
        ecm_note = ""
        if not pd.isna(ecm_coef):
            ecm_dir = "negative" if float(ecm_coef) < 0 else ("positive" if float(ecm_coef) > 0 else "zero")
            if ecm_significant:
                ecm_note = f"; ECM term significant ({ecm_dir}, p={float(ecm_p):.4f}{pstars(ecm_p)}) -> error-correction mechanism present"
            else:
                ecm_note = f"; ECM term not significant ({ecm_dir}, p={float(ecm_p):.4f}{pstars(ecm_p)})"

        return (
            f"Short-run: {'present' if sr_exist else 'absent'} ({sr_info}); "
            f"Long-run: {'present' if lr_exist else 'absent'} ({lr_info})" + ecm_note
        )
    except Exception:
        return "Short-run: unknown; Long-run: unknown"


def _print_and_export_ecm(direction: str,
                          emotion: str,
                          dep_col: str,
                          indep_col: str,
                          ecm_result: Dict[str, object],
                          ardl_result: Dict[str, object],
                          irf_out_dir: str,
                          exog_out_dir: str,
                          args) -> None:
    """
    Unified ARDL-ECM reporting/export (coefficients, exogenous terms, IRF, and details).
    direction: "forward" (covid->climate, dep=climate) or "reverse" (climate->covid, dep=covid)
    """
    try:
        # Print core coefficients
        if direction == "forward":
            print("  ARDL-ECM results:")
        else:
            print("  Reverse ARDL-ECM results:")
        if ecm_result.get('error'):
            if direction == "forward":
                print(f"    Error: {ecm_result['error']}")
            else:
                print(f"    Reverse ECM Error: {ecm_result['error']}")
        else:
            if direction == "forward":
                print(f"    ECM coefficient: {ecm_result['ecm_coefficient']:.4f} (p={ecm_result['ecm_pvalue']:.4f})")
                print(f"    Short-run coefficient: {ecm_result['short_run_coefficients'].get('delta_x', 'N/A'):.4f}")
                # Long-run details (SE, CI)
                lr_coef = ecm_result.get('long_run_coefficient', np.nan)
                lr_se = ecm_result.get('long_run_std_error', np.nan)
                lr_pval = ecm_result.get('long_run_pvalue', np.nan)
                lr_ci_lower = ecm_result.get('long_run_ci_lower', np.nan)
                lr_ci_upper = ecm_result.get('long_run_ci_upper', np.nan)
                if not pd.isna(lr_coef):
                    if not pd.isna(lr_se) and not pd.isna(lr_pval):
                        print(f"    Long-run coefficient: {lr_coef:.4f} (SE={lr_se:.4f}, p={lr_pval:.4f}, 95% CI=[{lr_ci_lower:.4f}, {lr_ci_upper:.4f}])")
                    else:
                        print(f"    Long-run coefficient: {lr_coef:.4f}")
                else:
                    print("    Long-run coefficient: N/A")
            else:
                print(f"    Reverse ECM coefficient: {ecm_result['ecm_coefficient']:.4f} (p={ecm_result['ecm_pvalue']:.4f})")
                print(f"    Reverse Short-run coefficient: {ecm_result['short_run_coefficients'].get('delta_x', 'N/A'):.4f}")
                # Long-run details (SE, CI)
                lr_coef = ecm_result.get('long_run_coefficient', np.nan)
                lr_se = ecm_result.get('long_run_std_error', np.nan)
                lr_pval = ecm_result.get('long_run_pvalue', np.nan)
                lr_ci_lower = ecm_result.get('long_run_ci_lower', np.nan)
                lr_ci_upper = ecm_result.get('long_run_ci_upper', np.nan)
                if not pd.isna(lr_coef):
                    if not pd.isna(lr_se) and not pd.isna(lr_pval):
                        print(f"    Reverse Long-run coefficient: {lr_coef:.4f} (SE={lr_se:.4f}, p={lr_pval:.4f}, 95% CI=[{lr_ci_lower:.4f}, {lr_ci_upper:.4f}])")
                    else:
                        print(f"    Reverse Long-run coefficient: {lr_coef:.4f}")
                else:
                    print(f"    Reverse Long-run coefficient: N/A")
            print(f"    R^2: {ecm_result['r_squared']:.4f}")
            # Export exogenous coefficients
            pair_key = f"{dep_col}_<-_{indep_col}"
            export_exog_coefficients(exog_out_dir, pair_key, ecm_result.get('exog_coefficients', {}))
            mode = "ECM_forward" if direction == "forward" else "ECM_reverse"
            append_exog_summary(exog_out_dir, emotion=emotion, mode=mode, pair_key=pair_key, exog_coeffs=ecm_result.get('exog_coefficients', {}))
            export_all_ardl_ecm_coefficients(exog_out_dir, emotion, direction, pair_key, ecm_result)
            # Refresh direction-wide table
            build_direction_wide_coefficients(exog_out_dir, direction="forward" if direction == "forward" else "reverse")
            # Export IRF
            try:
                coef = ecm_result.get('ecm_coefficient', np.nan)
                print(f"  Debug: args.irf = {args.irf}, ecm_coefficient = {coef}")
                if args.irf and not np.isnan(coef):
                    is_forward = direction == "forward"
                    # Level response (95% CI only)
                    irf_obj = compute_ecm_dynamic_multiplier_with_ci(ecm_result, horizon=args.irf_horizon, n_boot=args.irf_boot, return_delta=False, multi_ci=[0.95])
                    # Build DataFrame with 95% CI
                    irf_df_dict = {
                        "h": irf_obj["h"],
                        "irf": irf_obj["irf"],
                        "lower_95": irf_obj.get("lower_95", irf_obj.get("lower")),
                        "upper_95": irf_obj.get("upper_95", irf_obj.get("upper")),
                    }
                    irf_df = pd.DataFrame(irf_df_dict)
                    if is_forward:
                        irf_csv = os.path.join(irf_out_dir, f"irf_csv_{emotion}_covid_to_climate.csv")
                        direction_str = f"{indep_col}->{dep_col}"
                    else:
                        irf_csv = os.path.join(irf_out_dir, f"irf_csv_{emotion}_climate_to_covid.csv")
                        direction_str = f"{indep_col}->{dep_col}"
                    irf_df.to_csv(irf_csv, index=False, encoding="utf-8-sig")
                    print(f"  IRF curve CSV saved: {irf_csv}")
                    save_irf_plot(irf_out_dir, pair_key, direction=direction_str, irf=irf_obj, response_type="level")
                    # Differenced response (95% CI only)
                    irf_obj_delta = compute_ecm_dynamic_multiplier_with_ci(ecm_result, horizon=args.irf_horizon, n_boot=args.irf_boot, return_delta=True, multi_ci=[0.95])
                    # Build DataFrame with 95% CI
                    irf_df_delta_dict = {
                        "h": irf_obj_delta["h"],
                        "irf": irf_obj_delta["irf"],
                        "lower_95": irf_obj_delta.get("lower_95", irf_obj_delta.get("lower")),
                        "upper_95": irf_obj_delta.get("upper_95", irf_obj_delta.get("upper")),
                    }
                    irf_df_delta = pd.DataFrame(irf_df_delta_dict)
                    if is_forward:
                        irf_csv_delta = os.path.join(irf_out_dir, f"irf_csv_{emotion}_covid_to_climate_delta.csv")
                    else:
                        irf_csv_delta = os.path.join(irf_out_dir, f"irf_csv_{emotion}_climate_to_covid_delta.csv")
                    irf_df_delta.to_csv(irf_csv_delta, index=False, encoding="utf-8-sig")
                    print(f"  IRF delta curve CSV saved: {irf_csv_delta}")
                    save_irf_plot(irf_out_dir, pair_key, direction=direction_str, irf=irf_obj_delta, response_type="delta")
            except Exception:
                pass
            # Do not auto-claim "causality exists/doesn't exist"; keep numeric outputs only
    except Exception as e:
        print(f"ECM reporting/export error: {str(e)}")

def ardl_bounds_test(series1: pd.Series, series2: pd.Series, series1_name: str, series2_name: str, max_lags: int = 4, dates: pd.Series = None, weekday_ols: bool = False, weekday_trend: bool = False) -> Dict[str, object]:
    """
    ARDL bounds test (Pesaran et al., 2001).
    Applicable to mixed I(0)/I(1) combinations (including I(1)+I(1)), but not I(2) or higher.

    Args:
        series1, series2: raw time series
        series1_name, series2_name: series names
        max_lags: maximum lag order
        dates: date series used for optional preprocessing
        weekday_ols: whether to remove weekday effects
        weekday_trend: whether to remove a linear trend during preprocessing

    Returns:
        Dict containing ARDL bounds test results
    """
    # Clean inputs
    s1 = pd.to_numeric(series1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s2 = pd.to_numeric(series2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # Optional preprocessing (weekday effects and trend)
    if weekday_ols and dates is not None:
        # Align dates and series
        s1_dates = dates.loc[s1.index]
        s2_dates = dates.loc[s2.index]
        
        # Remove weekday effects and trend
        s1 = remove_weekday_effect(s1, s1_dates, add_trend=weekday_trend)
        s2 = remove_weekday_effect(s2, s2_dates, add_trend=weekday_trend)
        
        # Re-clean after preprocessing
        s1 = pd.to_numeric(s1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s2 = pd.to_numeric(s2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # Align indices
    common_index = s1.index.intersection(s2.index)
    if len(common_index) < 30:  # ARDL needs more data
        return {
            "series1": series1_name,
            "series2": series2_name,
            "f_statistic": np.nan,
            "bounds_result": "insufficient_data",
            "long_run_result": "insufficient_data",
            "error": "insufficient_data"
        }
    
    s1_aligned = s1.loc[common_index]
    s2_aligned = s2.loc[common_index]
    
    result = {
        "series1": series1_name,
        "series2": series2_name,
        "f_statistic": np.nan,
        "bounds_result": "failed",
        "long_run_result": "failed",
        "n_obs": np.nan,
        "df1": np.nan,
        "df2": np.nan,
        "p_value": np.nan,
        "bound_1pct_I0": np.nan,
        "bound_1pct_I1": np.nan,
        "bound_5pct_I0": np.nan,
        "bound_5pct_I1": np.nan,
        "error": None
    }
    
    try:
        # Simplified ARDL bounds test
        # Use a simple linear regression to probe the long-run relationship
        
        n = len(s1_aligned)
        if n < 20:
            result["error"] = "insufficient_observations"
            return result
        
        y = s1_aligned.values
        x = s2_aligned.values
        
        # Simple ARDL form: y_t = α + β1 y_{t-1} + β2 x_{t-1} + ε_t
        y_lag = y[:-1]  # y_{t-1}
        x_lag = x[:-1]  # x_{t-1}
        y_current = y[1:]  # y_t
        
        # Add constant
        X = np.column_stack([np.ones(len(y_lag)), y_lag, x_lag])
        
        # Fit model
        model = OLS(y_current, X)
        fitted_model = model.fit()
        
        # F-statistic (test β1 = β2 = 0)
        # F = (R^2/(k-1)) / ((1-R^2)/(n-k))
        r_squared = fitted_model.rsquared
        n_obs = len(y_current)
        k = 3  # constant + 2 regressors
        
        if r_squared < 0.99:  # avoid division by zero
            f_stat = (r_squared / (k-1)) / ((1-r_squared) / (n_obs - k))
        else:
            f_stat = 1000  # high-R^2 case
        
        result["f_statistic"] = f_stat
        result["n_obs"] = n_obs
        result["df1"] = k - 1
        result["df2"] = n_obs - k
        try:
            # Approximate p-value via F distribution (used for significance stars)
            result["p_value"] = float(1.0 - f.cdf(f_stat, dfn=result["df1"], dfd=result["df2"]))
        except Exception:
            result["p_value"] = np.nan
        
        # Bounds critical values (Pesaran et al., 2001; approximate, Case III / k≈1)
        # 1%: I(0)=6.84, I(1)=7.84; 5%: I(0)=3.17, I(1)=4.61
        # Decision rule below: F < I(0) => not_cointegrated; F > I(1) => cointegrated
        result["bound_1pct_I0"] = 6.84
        result["bound_1pct_I1"] = 7.84
        result["bound_5pct_I0"] = 3.17
        result["bound_5pct_I1"] = 4.61
        if f_stat > result["bound_5pct_I1"]:
            result["bounds_result"] = "cointegrated"
        elif f_stat < result["bound_5pct_I0"]:
            result["bounds_result"] = "not_cointegrated"
        else:
            result["bounds_result"] = "inconclusive"
        
        # Long-run relationship test
        if len(fitted_model.params) >= 2:
            beta1_pvalue = fitted_model.pvalues[1] if len(fitted_model.pvalues) > 1 else 1.0
            result["long_run_result"] = "significant" if beta1_pvalue < 0.05 else "not_significant"
        
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


# === ARDL-ECM dynamic multipliers / IRF and coefficient export utilities ===
def _simulate_ecm_irf(params: np.ndarray, horizon: int = 30) -> np.ndarray:
    """
    Simulate the level IRF path under a unit "permanent shock" to x (Δx=1 at h=0, then Δx=0 while x level stays +1).

    Parameters follow the simplified ECM order (consistent with ardl_ecm_model):
    const, b(Δx), a(y_{t-1}), d(x_{t-1}).

    Notes:
    - Set const=0 in the simulation to isolate the shock path.
    - Initial state: y_{-1}=0, x_{-1}=0.
    - At h=0: y_0 = y_{-1} + Δy_0 = b (short-run effect).
    - This matches the simplified ECM used in this script (no higher-order Δy/Δx lags).
    - Returns an array of length horizon+1 (h=0..horizon).
    """
    if params is None or len(params) < 4:
        return np.full(horizon + 1, np.nan)
    const = 0.0  # isolate shock
    b = float(params[1])  # contemporaneous Δx coefficient
    a = float(params[2])  # y_{t-1} coefficient (typically negative)
    d = float(params[3])  # x_{t-1} coefficient

    y_prev = 0.0  # y_{-1} = 0 (pre-shock state)
    x_prev = 0.0  # x_{-1} = 0 (pre-shock state)
    x_level = 0.0
    path_y = []  # start empty; h=0 is the shock period
    for t in range(horizon + 1):  # include h=0 => horizon+1 points
        if t == 0:
            delta_x = 1.0  # shock at h=0
        else:
            delta_x = 0.0  # after h=0, x level remains shifted (permanent shock)
        # Δy_t = const + b*Δx_t + a*y_{t-1} + d*x_{t-1}
        delta_y = const + b * delta_x + a * y_prev + d * x_prev
        y_level = y_prev + delta_y
        # Update x level
        x_level = x_prev + delta_x
        # Advance state
        y_prev = y_level
        x_prev = x_level
        path_y.append(y_level)
    return np.array(path_y)


def compute_ecm_dynamic_multiplier_with_ci(ecm_result: Dict[str, object], horizon: int = 30, n_boot: int = 500, ci: float = 0.95, return_delta: bool = False, multi_ci: List[float] = None) -> Dict[str, object]:
    """
    Build IRF/dynamic multipliers and confidence intervals using a parametric bootstrap
    (normal approximation) from fitted ECM parameters and covariance.

    Args:
        return_delta: if True, return differenced response (Δy) instead of level response.
        multi_ci: optional list of confidence levels (e.g. [0.90, 0.95]); returns lower_90/upper_90 etc.

    Returns:
        Dict like {"h": array, "irf": array, "lower": array, "upper": array, "long_run": float, ...}.
    """
    fp = ecm_result.get("_fitted_params", {})
    params = fp.get("params", None)
    cov = fp.get("cov", None)
    cols = fp.get("design_cols", [])

    # Map the full parameter vector to the 4 core parameters used by the simplified ECM:
    # [const, b (contemporaneous Δx / dx_lag0), a (y_{t-1}), d (x_{t-1})]
    def _extract_core_params(pvec: np.ndarray) -> np.ndarray:
        if pvec is None or not isinstance(pvec, (list, np.ndarray)) or len(cols) == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
        def _get(name: str, default: float = 0.0, allow_nan: bool = False) -> float:
            try:
                if name in cols:
                    v = float(pvec[cols.index(name)])
                else:
                    v = default
                if not allow_nan and (pd.isna(v) or not np.isfinite(v)):
                    return default
                return v
            except Exception:
                return default if not allow_nan else np.nan
        const = _get("const", 0.0)
        b = _get("dx_lag0", 0.0)  # contemporaneous Δx coefficient
        # a and d must exist; if missing, return NaN to propagate downstream
        a = _get("y_lag", np.nan, allow_nan=True)
        d = _get("x_lag", np.nan, allow_nan=True)
        return np.array([const, b, a, d], dtype=float)

    if params is None or cov is None:
        # Even if params are missing, return the multi-CI structure if requested
        result = {"h": np.arange(horizon + 1), "irf": np.full(horizon + 1, np.nan), "long_run": np.nan}
        if multi_ci is not None and len(multi_ci) > 0:
            for ci_level in multi_ci:
                ci_label = int(ci_level * 100)
                result[f"lower_{ci_label}"] = np.full(horizon + 1, np.nan)
                result[f"upper_{ci_label}"] = np.full(horizon + 1, np.nan)
            # Also set default lower/upper keys
            result["lower"] = np.full(horizon + 1, np.nan)
            result["upper"] = np.full(horizon + 1, np.nan)
        else:
            result["lower"] = np.full(horizon + 1, np.nan)
            result["upper"] = np.full(horizon + 1, np.nan)
        return result

    # Point-estimate path (level response) from mapped core parameters
    core_params = _extract_core_params(params)
    if np.isnan(core_params).any():
        irf_level = np.full(horizon + 1, np.nan)
    else:
        irf_level = _simulate_ecm_irf(core_params, horizon=horizon)
    
    # Compute differenced response (if requested)
    if return_delta:
        # Differenced response = current level - previous level.
        # With y_{-1}=0 and level response y_0=b at h=0, the impact response is b.
        irf_point = np.diff(irf_level, prepend=0.0)  # prepend=0 implies y_{-1}=0
        # No further correction needed: impact at h=0 equals b
    else:
        irf_point = irf_level
    
    # Long-run effect (lambda), computed from level response
    try:
        a = float(core_params[2])
        d = float(core_params[3])
        long_run = -d / a if abs(a) > 1e-12 else np.nan
    except Exception:
        long_run = np.nan

    # Parametric bootstrap
    rng = np.random.default_rng(2025)
    draws = rng.multivariate_normal(mean=params, cov=cov, size=int(n_boot), check_valid='ignore')
    irf_draws_level = []
    for pdraw in draws:
        core = _extract_core_params(pdraw)
        if np.isnan(core).any():
            irf_draws_level.append(np.full(horizon + 1, np.nan))
        else:
            irf_draws_level.append(_simulate_ecm_irf(core, horizon=horizon))
    irf_draws_level = np.array(irf_draws_level)
    
    # Confidence intervals for differenced response
    if return_delta:
        # Since level response has y_0=b, the differenced response is consistent automatically
        irf_draws = np.array([np.diff(path, prepend=0.0) for path in irf_draws_level])
    else:
        irf_draws = irf_draws_level
    
    # Build return dict
    result = {"h": np.arange(horizon + 1), "irf": irf_point, "long_run": long_run}
    
    # If multiple confidence levels are provided, compute all of them
    if multi_ci is not None and len(multi_ci) > 0:
        # Compute all requested confidence intervals
        for ci_level in multi_ci:
            lower_q = (1 - ci_level) / 2
            upper_q = 1 - lower_q
            ci_label = int(ci_level * 100)
            result[f"lower_{ci_label}"] = np.nanpercentile(irf_draws, lower_q * 100, axis=0)
            result[f"upper_{ci_label}"] = np.nanpercentile(irf_draws, upper_q * 100, axis=0)
        # For backward compatibility, also set default lower/upper (use the largest CI)
        max_ci = max(multi_ci)
        lower_q = (1 - max_ci) / 2
        upper_q = 1 - lower_q
        result["lower"] = np.nanpercentile(irf_draws, lower_q * 100, axis=0)
        result["upper"] = np.nanpercentile(irf_draws, upper_q * 100, axis=0)
    else:
        # Single CI only (backward compatibility)
        lower_q = (1 - ci) / 2
        upper_q = 1 - lower_q
        result["lower"] = np.nanpercentile(irf_draws, lower_q * 100, axis=0)
        result["upper"] = np.nanpercentile(irf_draws, upper_q * 100, axis=0)
    
    return result


def save_irf_plot(csv_dir: str, pair_key: str, direction: str, irf: Dict[str, np.ndarray], response_type: str = "level"):
    """
    Save an IRF plot to disk.

    Args:
        response_type: "level" for level response (cumulative), "delta" for differenced response (impact)
    """
    try:
        h = irf["h"]
        y = irf["irf"]
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(h, y, label="IRF", color="#1f77b4", linewidth=2)
        
        # Check whether multiple CIs exist (e.g., lower_90/upper_90, lower_95/upper_95)
        # First detect numeric-suffixed CI keys (e.g., lower_90, upper_90)
        ci_keys = []
        for key in irf.keys():
            if key.startswith("lower_") and len(key) > 6:
                suffix = key[6:]  # part after "lower_"
                if suffix.isdigit():
                    ci_keys.append(key)
            elif key.startswith("upper_") and len(key) > 6:
                suffix = key[6:]  # part after "upper_"
                if suffix.isdigit():
                    ci_keys.append(key)
        
        ci_levels = set()
        for key in ci_keys:
            if key.startswith("lower_"):
                ci_levels.add(key.replace("lower_", ""))
            elif key.startswith("upper_"):
                ci_levels.add(key.replace("upper_", ""))
        
        # If numeric CI keys are found, use multi-CI mode
        # Validate each CI level has lower/upper arrays and usable data
        valid_ci_levels = []
        for ci_level_str in ci_levels:
            try:
                ci_level = int(ci_level_str)
                lower_key = f"lower_{ci_level}"
                upper_key = f"upper_{ci_level}"
                if lower_key in irf and upper_key in irf:
                    lo = irf[lower_key]
                    hi = irf[upper_key]
                    # Ensure there is at least some valid data (not all NaN)
                    if lo is not None and hi is not None and np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                        valid_ci_levels.append(ci_level)
            except (ValueError, TypeError):
                continue
        
        # Plot 95% CI only (ignore 90% for this figure)
        if valid_ci_levels:
            # Keep 95% CI only
            filtered_ci_levels = [ci for ci in valid_ci_levels if ci == 95]
            if filtered_ci_levels:
                # Colors/alpha for 95% CI
                ci_colors = {
                    95: ("#1f77b4", 0.25),  # 95% CI: deeper blue, higher opacity
                }
                # Default style (if CI level not preset)
                default_color = ("#1f77b4", 0.2)
                
                for ci_level in filtered_ci_levels:
                    lower_key = f"lower_{ci_level}"
                    upper_key = f"upper_{ci_level}"
                    lo = irf[lower_key]
                    hi = irf[upper_key]
                    # Plot using valid data (handle NaNs)
                    valid_mask = np.isfinite(lo) & np.isfinite(hi)
                    if np.any(valid_mask):
                        color, alpha = ci_colors.get(ci_level, default_color)
                        plt.fill_between(h[valid_mask], lo[valid_mask], hi[valid_mask], color=color, alpha=alpha, label=f"{ci_level}% CI")
        else:
            # Backward compatibility: plot the default CI only
            lo = irf.get("lower")
            hi = irf.get("upper")
            if lo is not None and hi is not None:
                # Ensure there is at least some valid data
                valid_mask = np.isfinite(lo) & np.isfinite(hi)
                if np.any(valid_mask):
                    plt.fill_between(h[valid_mask], lo[valid_mask], hi[valid_mask], color="#1f77b4", alpha=0.2, label="CI")
        
        plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        plt.xlabel("Horizon")
        plt.legend(loc="best", fontsize=9)
        
        # Simplify title: extract emotion name and direction
        emotion = ""
        if "climate_" in pair_key and "covid_" in pair_key:
            # Extract emotion name
            emotion = pair_key.split("climate_")[1].split("_score_freq")[0]
        elif "VAR_" in pair_key:
            # VAR case: try extracting from pair_key
            parts = pair_key.split("_")
            for i, part in enumerate(parts):
                if part in ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]:
                    emotion = part
                    break
        
        # Simplify direction label
        if "covid" in direction and "climate" in direction:
            if "covid" in direction.split("->")[0]:
                dir_text = "COVID → Climate"
            else:
                dir_text = "Climate → COVID"
        else:
            dir_text = direction.replace("->", " → ")
        
        # Title and y-axis label
        emotion_title = emotion.title() if emotion else "VAR"
        # Include λ (long-run effect) in title for visual comparison
        try:
            lr = irf.get("long_run", np.nan)
            lr_text = "" if (pd.isna(lr) or not np.isfinite(lr)) else f" | λ={float(lr):.3f}"
        except Exception:
            lr_text = ""
        if response_type == "delta":
            plt.ylabel("Response (Δ)")
            plt.title(f"IRF: {emotion_title} ({dir_text}) [Delta Response]{lr_text}")
        else:
            plt.ylabel("Response (Level)")
            plt.title(f"IRF: {emotion_title} ({dir_text}) [Cumulative Level Response]{lr_text}")
        
        # Simplify filename
        safe_emotion = emotion.replace(':', '_').replace('/', '_').replace('<', '_').replace('>', '_') if emotion else "var"
        if "covid" in direction and "climate" in direction:
            if "covid" in direction.split("->")[0]:
                file_suffix = "covid_to_climate"
            else:
                file_suffix = "climate_to_covid"
        else:
            file_suffix = direction.replace('->', 'to').replace(':', '_').replace('/', '_').replace('$', '')
        
        # Suffix by response type to avoid filename collisions
        response_suffix = "_delta" if response_type == "delta" else "_level"
        out_path = os.path.join(csv_dir, f"irf_{safe_emotion}_{file_suffix}{response_suffix}.png")
        plt.tight_layout()
        # Ensure overwrite (some viewers cache timestamps)
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"IRF plot saved: {out_path}")
    except Exception:
        print(f"IRF plot save failed: {pair_key} | {direction}")


def save_causality_dag(out_dir: str, emotion: str, edges: List[Tuple[str, str]]):
    """Draw causal arrow diagram for climate and covid under the same emotion.

    edges: List of directed edges, e.g., [("covid", "climate")] or bidirectional edges.
    Output filename examples: dag_emotion_covid_to_climate.png or dag_emotion_bidirectional.png
    """
    try:
        # Canvas and coordinates
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Node positions
        pos = {
            'climate': (2.5, 3.0),
            'covid': (7.5, 3.0),
        }

        # Determine if bidirectional causality
        is_bidirectional = set(edges) == {('climate', 'covid'), ('covid', 'climate')}
        # Node radius: 50% larger for bidirectional
        node_radius = 1.35 if is_bidirectional else 0.9

        # Draw nodes
        for name, (x, y) in pos.items():
            circle = plt.Circle((x, y), node_radius, color="#e6f2ff", ec="#1f77b4", lw=2)
            ax.add_patch(circle)
            label = f"{name}_{emotion}"
            ax.text(x, y, label, ha='center', va='center', fontsize=10)

        # Arrows
        if is_bidirectional:
            # Use single bidirectional arrow, adjusting start/end points based on enlarged node radius
            x1, y1 = pos['climate']
            x2, y2 = pos['covid']
            ax.annotate(
                "",
                xy=(x2 - node_radius, y2),
                xytext=(x1 + node_radius, y1),
                arrowprops=dict(arrowstyle="<->", lw=2, color="#444", connectionstyle="arc3,rad=0.0"),
            )
        else:
            # Unidirectional: same logic as before
            for (src, dst) in edges:
                x1, y1 = pos[src]
                x2, y2 = pos[dst]
                ax.annotate(
                    "",
                    xy=(x2 - node_radius, y2),
                    xytext=(x1 + node_radius, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#444", connectionstyle="arc3,rad=0.0"),
                )

        # Filename
        if set(edges) == {('climate', 'covid'), ('covid', 'climate')}:
            suffix = "bidirectional"
        elif edges:
            suffix = f"{edges[0][0]}_to_{edges[0][1]}"
        else:
            suffix = "none"

        safe_emotion = str(emotion).replace(':','_').replace('/','_').replace('<','_').replace('>','_')
        out_path = os.path.join(out_dir, f"dag_{safe_emotion}_{suffix}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Causal DAG saved: {out_path}")
    except Exception as _:
        print("Causal DAG save failed")


def plot_emotion_timeseries(df: pd.DataFrame, date_col: str, emotion: str, out_dir: str):
    """
    Plot time series for a single emotion, including both climate and covid topics, with Pearson correlation coefficient annotation
    
    Args:
        df: DataFrame
        date_col: Date column name
        emotion: Emotion name
        out_dir: Output directory
    """
    try:
        climate_col = f"climate_{emotion}_score_freq"
        covid_col = f"covid_{emotion}_score_freq"
        
        # Check if columns exist
        if climate_col not in df.columns or covid_col not in df.columns:
            print(f"Warning: Columns for {emotion} emotion do not exist")
            return
        
        # Extract data
        dates = pd.to_datetime(df[date_col])
        climate_data = pd.to_numeric(df[climate_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        covid_data = pd.to_numeric(df[covid_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        ax.plot(dates, climate_data, label=f'Climate {emotion.title()}', linewidth=2, color='#2E8B57', alpha=0.8)
        ax.plot(dates, covid_data, label=f'COVID {emotion.title()}', linewidth=2, color='#DC143C', alpha=0.8)
        
        # Calculate Pearson correlation coefficient
        # Only use points where both sequences have valid values
        valid_mask = ~(climate_data.isna() | covid_data.isna())
        if valid_mask.sum() > 10:  # At least 10 valid data points required
            corr_coef, corr_pvalue = pearsonr(climate_data[valid_mask], covid_data[valid_mask])
            corr_text = f'Pearson r = {corr_coef:.3f}'
            if corr_pvalue < 0.001:
                corr_text += ' (p < 0.001)'
            elif corr_pvalue < 0.01:
                corr_text += f' (p = {corr_pvalue:.3f})'
            else:
                corr_text += f' (p = {corr_pvalue:.3f})'
        else:
            corr_text = 'Insufficient data for correlation'
            corr_coef = np.nan
        
        # Add correlation coefficient annotation
        ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Set figure properties
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Emotion Score Frequency', fontsize=12)
        ax.set_title(f'{emotion.title()} Emotion Time Series: Climate vs COVID', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        safe_emotion = str(emotion).replace(':', '_').replace('/', '_').replace('<', '_').replace('>', '_')
        out_path = os.path.join(out_dir, f"timeseries_{safe_emotion}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Time series plot saved: {out_path}")
        if not np.isnan(corr_coef):
            print(f"  {emotion} Pearson correlation coefficient: {corr_coef:.4f}")
        
    except Exception as e:
        print(f"Failed to plot {emotion} time series: {str(e)}")


def export_ardl_lag_coefficients(csv_dir: str, pair_key: str, ecm_result: Dict[str, object]):
    """Export simplified ARDL-ECM coefficients (dy/dx lags, y_{t-1}, x_{t-1}, and exogenous terms)."""
    try:
        fp = ecm_result.get("_fitted_params", {})
        params = fp.get("params", None)
        pvalues = fp.get("pvalues", None)
        cols = fp.get("design_cols", [])
        
        # p-value mapping from short_run_coefficients (backward compatibility)
        short_run_coeffs = ecm_result.get("short_run_coefficients", {})
        pvals_map = {
            "const": np.nan,
            "delta_x": short_run_coeffs.get("delta_x_pvalue", np.nan),
            "y_lag": ecm_result.get("ecm_pvalue", np.nan),
            "x_lag": ecm_result.get("long_run_pvalue", np.nan),
        }
        
        rows: List[Dict[str, object]] = []
        if params is not None and cols:
            for i, name in enumerate(cols):
                coef_val = float(params[i]) if i < len(params) else np.nan
                # Prefer pvalues array; fall back to pvals_map
                if pvalues is not None and i < len(pvalues) and not np.isnan(pvalues[i]):
                    pval_val = float(pvalues[i])
                else:
                    # For dy_lag/dx_lag terms, fall back to short_run_coefficients mapping
                    if name.startswith("dy_lag") or name.startswith("dx_lag"):
                        pval_key = f"{name}_pvalue"
                        pval_val = short_run_coeffs.get(pval_key, np.nan)
                    else:
                        pval_val = pvals_map.get(name, np.nan)
                
                rows.append({
                    "term": name,
                    "coefficient": coef_val,
                    "pvalue": pval_val
                })
        if rows:
            df_out = pd.DataFrame(rows)
            # Add significance star column
            def _star(p):
                try:
                    p = float(p)
                except Exception:
                    return ""
                if np.isnan(p):
                    return ""
                if p < 0.01:
                    return "***"
                if p < 0.05:
                    return "**"
                if p < 0.10:
                    return "*"
                return ""
            if 'pvalue' in df_out.columns:
                df_out['star'] = df_out['pvalue'].apply(_star)
            # Display formatting: coefficient in scientific notation (2 sig digits); p-value to 2 decimals
            def _fmt_coef(v):
                try:
                    if pd.isna(v):
                        return ""
                    v = float(v)
                except Exception:
                    return ""
                return np.format_float_scientific(v, precision=2, unique=False, exp_digits=2)
            def _fmt_p(v):
                try:
                    if pd.isna(v):
                        return ""
                    v = float(v)
                except Exception:
                    return ""
                return f"{v:.2f}"
            if 'coefficient' in df_out.columns:
                df_out['coefficient'] = df_out['coefficient'].apply(_fmt_coef)
            if 'pvalue' in df_out.columns:
                df_out['pvalue'] = df_out['pvalue'].apply(_fmt_p)
            out_path = os.path.join(csv_dir, f"lag_coefs_{pair_key.replace(':','_').replace('/','_')}.csv")
            df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def export_exog_coefficients(csv_dir: str, pair_key: str, exog_coeffs: Dict[str, Dict[str, object]]):
    """Export result["exog_coefficients"] to CSV (columns: variable, coefficient, pvalue)."""
    # Safe filename conversion
    def _safe_name(name: str) -> str:
        try:
            s = str(name)
        except Exception:
            s = "name"
        # Replace Windows-invalid filename characters: <>:"/\|?*
        for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
            s = s.replace(ch, '_')
        return s
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception as e:
        print(f"  Exogenous coefficients: failed to create dir {csv_dir}: {e}")
        return
    rows: List[Dict[str, object]] = []
    for var_name, stats in (exog_coeffs or {}).items():
        coef = stats.get("coefficient", np.nan)
        pval = stats.get("pvalue", np.nan)
        rows.append({
            "variable": str(var_name),
            "coefficient": float(coef) if pd.notna(coef) else np.nan,
            "pvalue": float(pval) if pd.notna(pval) else np.nan,
        })
    df_out = pd.DataFrame(rows, columns=["variable", "coefficient", "pvalue"])  # export even if empty
    # Add significance star column
    def _star(p):
        try:
            p = float(p)
        except Exception:
            return ""
        if pd.isna(p):
            return ""
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
        return ""
    if 'pvalue' in df_out.columns:
        df_out['star'] = df_out['pvalue'].apply(_star)
    # Display formatting: coefficient scientific notation (2 sig digits); p-value to 2 decimals
    def _fmt_coef(v):
        try:
            if pd.isna(v):
                return ""
            v = float(v)
        except Exception:
            return ""
        return np.format_float_scientific(v, precision=2, unique=False, exp_digits=2)
    def _fmt_p(v):
        try:
            if pd.isna(v):
                return ""
            v = float(v)
        except Exception:
            return ""
        return f"{v:.2f}"
    if 'coefficient' in df_out.columns:
        df_out['coefficient'] = df_out['coefficient'].apply(_fmt_coef)
    if 'pvalue' in df_out.columns:
        df_out['pvalue'] = df_out['pvalue'].apply(_fmt_p)
    out_path = os.path.join(csv_dir, f"exog_coefs_{_safe_name(pair_key)}.csv")
    try:
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Exogenous coefficients CSV saved: {out_path} (rows={len(df_out)})")
    except Exception as e:
        print(f"  Exogenous coefficients CSV save failed: {out_path}, error={e}")


def export_all_ardl_ecm_coefficients(csv_dir: str, emotion: str, direction: str, pair_key: str, ecm_result: Dict[str, object]):
    """
    Export all ARDL-ECM coefficients to a CSV file.

    Args:
        csv_dir: output directory
        emotion: emotion label
        direction: 'forward' or 'reverse'
        pair_key: pair key
        ecm_result: ARDL-ECM result dict
    """
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception as e:
        print(f"  Failed to create dir {csv_dir}: {e}")
        return
    
    rows = []
    
    # If an error occurred, export it as a row
    if ecm_result.get('error'):
        rows.append({
            "emotion": emotion,
            "direction": direction,
            "pair_key": pair_key,
            "coefficient_name": "error",
            "coefficient_value": ecm_result.get('error'),
            "pvalue": np.nan,
            "series1": ecm_result.get('series1', ''),
            "series2": ecm_result.get('series2', '')
        })
    else:
        # Basic info
        series1 = ecm_result.get('series1', '')
        series2 = ecm_result.get('series2', '')
        
        # Fitted parameters
        fp = ecm_result.get("_fitted_params", {})
        params = fp.get("params", None)
        pvalues = fp.get("pvalues", None)
        design_cols = list(fp.get("design_cols", []))
        cov = fp.get("cov", None)
        bse = fp.get("bse", None)

        # Export coefficients for all design columns (also record SE for wide-table display)
        if params is not None and design_cols:
            for i, name in enumerate(design_cols):
                coef_val = float(params[i]) if i < len(params) and not np.isnan(params[i]) else np.nan
                pval_val = float(pvalues[i]) if (pvalues is not None and i < len(pvalues) and not np.isnan(pvalues[i])) else np.nan
                # SE from covariance diagonal; fall back to bse array
                se_val = np.nan
                if cov is not None and isinstance(cov, (np.ndarray, list)) and np.shape(cov)[0] > i and np.shape(cov)[1] > i:
                    try:
                        se_val = float(np.sqrt(cov[i][i]))
                    except Exception:
                        se_val = np.nan
                # If covariance is unavailable/NaN, try bse
                if (pd.isna(se_val) or not np.isfinite(se_val)) and bse is not None:
                    try:
                        if len(bse) > i and not np.isnan(bse[i]):
                            se_val = float(bse[i])
                    except Exception:
                        pass
                rows.append({
                    "emotion": emotion,
                    "direction": direction,
                    "pair_key": pair_key,
                    "coefficient_name": str(name),
                    "coefficient_value": coef_val,
                    "pvalue": pval_val,
                    "std_error": se_val,
                    "series1": series1,
                    "series2": series2
                })

        # Long-run coefficient (computed from ECM; Delta method)
        long_run_coef = ecm_result.get("long_run_coefficient", np.nan)
        long_run_pval = ecm_result.get("long_run_pvalue", np.nan)
        long_run_se = ecm_result.get("long_run_std_error", np.nan)
        long_run_tstat = ecm_result.get("long_run_tstat", np.nan)
        long_run_ci_lower = ecm_result.get("long_run_ci_lower", np.nan)
        long_run_ci_upper = ecm_result.get("long_run_ci_upper", np.nan)
        rows.append({
            "emotion": emotion,
            "direction": direction,
            "pair_key": pair_key,
            "coefficient_name": "long_run_coefficient",
            "coefficient_value": float(long_run_coef) if not np.isnan(long_run_coef) else np.nan,
            "pvalue": float(long_run_pval) if not np.isnan(long_run_pval) else np.nan,
            "std_error": float(long_run_se) if not np.isnan(long_run_se) else np.nan,
            "t_statistic": float(long_run_tstat) if not np.isnan(long_run_tstat) else np.nan,
            "ci_lower_95": float(long_run_ci_lower) if not np.isnan(long_run_ci_lower) else np.nan,
            "ci_upper_95": float(long_run_ci_upper) if not np.isnan(long_run_ci_upper) else np.nan,
            "series1": series1,
            "series2": series2
        })

        # Exogenous terms are already included in design_cols; do not append duplicates here
        
        # Model statistics
        r_squared = ecm_result.get("r_squared", np.nan)
        aic = ecm_result.get("aic", np.nan)
        rows.append({
            "emotion": emotion,
            "direction": direction,
            "pair_key": pair_key,
            "coefficient_name": "r_squared",
            "coefficient_value": float(r_squared) if not np.isnan(r_squared) else np.nan,
            "pvalue": np.nan,
            "series1": series1,
            "series2": series2
        })
        rows.append({
            "emotion": emotion,
            "direction": direction,
            "pair_key": pair_key,
            "coefficient_name": "aic",
            "coefficient_value": float(aic) if not np.isnan(aic) else np.nan,
            "pvalue": np.nan,
            "series1": series1,
            "series2": series2
        })
    
    # Export to CSV
    if rows:
        df_out = pd.DataFrame(rows)
        # Add significance star column
        def _star(p):
            try:
                p = float(p)
            except Exception:
                return ""
            if pd.isna(p):
                return ""
            if p < 0.01:
                return "***"
            if p < 0.05:
                return "**"
            if p < 0.10:
                return "*"
            return ""
        if 'pvalue' in df_out.columns:
            df_out['star'] = df_out['pvalue'].apply(_star)
        # Display formatting: coefficient scientific notation (2 sig digits); p-value to 2 decimals
        def _fmt_coef(v):
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
            except Exception:
                return ""
            return np.format_float_scientific(v, precision=2, unique=False, exp_digits=2)
        def _fmt_p(v):
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
            except Exception:
                return ""
            return f"{v:.2f}"
        if 'coefficient_value' in df_out.columns:
            df_out['coefficient_value'] = df_out['coefficient_value'].apply(_fmt_coef)
        if 'pvalue' in df_out.columns:
            df_out['pvalue'] = df_out['pvalue'].apply(_fmt_p)
        summary_path = os.path.join(csv_dir, "ardl_ecm_all_coefficients.csv")
        
        # If the summary file exists, append; otherwise create a new file
        if os.path.exists(summary_path):
            try:
                df_existing = pd.read_csv(summary_path, encoding="utf-8-sig")
                df_out = pd.concat([df_existing, df_out], ignore_index=True)
            except Exception:
                pass
        
        try:
            df_out.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"  ARDL-ECM coefficients exported: {summary_path} (added {len(rows)} rows)")
        except Exception as e:
            print(f"  ARDL-ECM coefficients export failed: {summary_path}, error={e}")


def build_direction_wide_coefficients(csv_dir: str, direction: str, outfile: str = None):
    """
    Build a direction-aggregated wide table from ardl_ecm_all_coefficients.csv.

    - Columns: emotions
    - Rows: coefficient / p-value / standard error for each term; additionally include ECM term rows
    - Output: coefficients_forward_wide.csv or coefficients_reverse_wide.csv
    """
    try:
        summary_path = os.path.join(csv_dir, "ardl_ecm_all_coefficients.csv")
        if not os.path.exists(summary_path):
            return
        df = pd.read_csv(summary_path, encoding="utf-8-sig")
        if df.empty:
            return
        if 'direction' not in df.columns or 'emotion' not in df.columns:
            return
        df_dir = df[df['direction'] == direction].copy()
        if df_dir.empty:
            return
        # Keep required columns only
        for col in ["coefficient_name", "coefficient_value", "pvalue", "std_error", "star"]:
            if col not in df_dir.columns:
                df_dir[col] = np.nan
        emotions = sorted(df_dir['emotion'].dropna().astype(str).unique().tolist())
        variables = df_dir['coefficient_name'].dropna().astype(str).unique().tolist()

        # Base exogenous variable names (displayed with exog_ prefix in the wide table; coefficients come from the ECM model)
        exog_base_vars = set([
            "US_daily_covid_death", "debates", "climatenews", "GovernmentResponseIndex_Average",
            "WinterStorm", "Wildfire", "TropicalCyclone", "SevereStorm", "Flood", "Drought",
            "const"
        ] + [f"wd_{k}" for k in range(1, 7)])
        rows: List[Dict[str, object]] = []
        def _format_val(v):
            """
            Coefficient formatting: scientific notation when present; empty string if missing.
            """
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
            except Exception:
                return ""
            return np.format_float_scientific(v, precision=2, unique=False, exp_digits=2)
        def _format_p(v):
            """
            p-value formatting: 4 decimals when present; empty string if missing.
            """
            try:
                if pd.isna(v):
                    return ""
                v = float(v)
            except Exception:
                return ""
            return f"{v:.4f}"

        for var in variables:
            sub = df_dir[df_dir['coefficient_name'].astype(str) == str(var)]
            # Display name: add exog_ prefix for exogenous variables; keep original for others
            display_var = f"exog_{var}" if str(var) in exog_base_vars else str(var)

            # For r_squared and aic, keep coefficient row only (no p-value or SE rows)
            if var in ["r_squared", "aic"]:
                row_coef: Dict[str, object] = {"variable": display_var}
                for emo in emotions:
                    item = sub[sub['emotion'].astype(str) == emo]
                    coef_val = np.nan
                    if not item.empty:
                        last = item.iloc[-1]
                        try:
                            coef_val = float(last.get('coefficient_value', np.nan)) if pd.notna(last.get('coefficient_value', np.nan)) else np.nan
                        except Exception:
                            coef_val = np.nan
                    row_coef[emo] = _format_val(coef_val)
                rows.append(row_coef)
                continue

            # Other variables: coefficient row + p-value row + SE row
            row_coef: Dict[str, object] = {"variable": display_var}
            row_p: Dict[str, object] = {"variable": f"{display_var}_pvalue"}
            row_se: Dict[str, object] = {"variable": f"{display_var}_std_error"}
            for emo in emotions:
                item = sub[sub['emotion'].astype(str) == emo]
                coef_val = np.nan
                p_val = np.nan
                se_val = np.nan
                if not item.empty:
                    last = item.iloc[-1]
                    try:
                        coef_val = float(last.get('coefficient_value', np.nan)) if pd.notna(last.get('coefficient_value', np.nan)) else np.nan
                    except Exception:
                        coef_val = np.nan
                    try:
                        p_val = float(last.get('pvalue', np.nan)) if pd.notna(last.get('pvalue', np.nan)) else np.nan
                    except Exception:
                        p_val = np.nan
                    try:
                        se_val = float(last.get('std_error', np.nan)) if pd.notna(last.get('std_error', np.nan)) else np.nan
                    except Exception:
                        se_val = np.nan
                row_coef[emo] = _format_val(coef_val)
                row_p[emo] = _format_p(p_val)
                row_se[emo] = _format_val(se_val)
            rows.append(row_coef)
            rows.append(row_p)
            rows.append(row_se)

        # Add ECM coefficient + p-value + SE rows (based on y_lag term)
        ecm_coef_row: Dict[str, object] = {"variable": "ecm_coefficient"}
        ecm_p_row: Dict[str, object] = {"variable": "ecm_pvalue"}
        ecm_se_row: Dict[str, object] = {"variable": "ecm_std_error"}
        sub_ecm = df_dir[df_dir["coefficient_name"] == "y_lag"]
        for emo in emotions:
            item = sub_ecm[sub_ecm["emotion"].astype(str) == emo]
            if not item.empty:
                last = item.iloc[-1]
                coef_val = last.get("coefficient_value", np.nan)
                p_val = last.get("pvalue", np.nan)
                se_val = last.get("std_error", np.nan)
            else:
                coef_val = np.nan
                p_val = np.nan
                se_val = np.nan
            ecm_coef_row[emo] = _format_val(coef_val)
            # Keep p-values to 4 decimals for readability
            ecm_p_row[emo] = _format_p(p_val)
            ecm_se_row[emo] = _format_val(se_val)
        rows.append(ecm_coef_row)
        rows.append(ecm_p_row)
        rows.append(ecm_se_row)

        # Build wide table directly; no NOTE row (interpret NA in paper/figure notes as needed)
        df_wide = pd.DataFrame(rows, columns=["variable"] + emotions)
        out_name = outfile if outfile else ("coefficients_forward_wide.csv" if direction == "forward" else "coefficients_reverse_wide.csv")
        out_path = os.path.join(csv_dir, out_name)
        df_wide.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def append_exog_summary(csv_dir: str, emotion: str, mode: str, pair_key: str, exog_coeffs: Dict[str, Dict[str, object]]):
    """Append a single export of exogenous coefficients to the global summary CSV (exog_coefs_summary.csv).
    Columns: emotion, mode, pair_key, variable, coefficient, pvalue
    """
    # Directory and file
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception:
        return
    summary_path = os.path.join(csv_dir, "exog_coefs_summary.csv")
    # Normalization
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    rows: List[Dict[str, object]] = []
    for var_name, stats in (exog_coeffs or {}).items():
        pval_val = _to_float(stats.get("pvalue", np.nan))
        # Common econ significance stars: * p<0.10, ** p<0.05, *** p<0.01
        if pd.notna(pval_val):
            if pval_val < 0.01:
                star = "***"
            elif pval_val < 0.05:
                star = "**"
            elif pval_val < 0.10:
                star = "*"
            else:
                star = ""
        else:
            star = ""
        rows.append({
            "emotion": str(emotion),
            "mode": str(mode),
            "pair_key": str(pair_key),
            "variable": str(var_name),
            "coefficient": _to_float(stats.get("coefficient", np.nan)),
            "pvalue": pval_val,
            "star": star,
        })
    # Write header even if empty (helps validation)
    df_rows = pd.DataFrame(rows, columns=["emotion", "mode", "pair_key", "variable", "coefficient", "pvalue", "star"])
    try:
        if not os.path.exists(summary_path):
            # First write (with header)
            df_rows.to_csv(summary_path, index=False, encoding="utf-8-sig")
        else:
            # Old file may lack the star column; perform one-time migration
            try:
                existing = pd.read_csv(summary_path)
                if 'star' not in existing.columns:
                    # Backfill star column for old rows
                    def _mk_star(p):
                        try:
                            pv = float(p)
                        except Exception:
                            return ""
                        if np.isnan(pv):
                            return ""
                        if pv < 0.01:
                            return "***"
                        if pv < 0.05:
                            return "**"
                        if pv < 0.10:
                            return "*"
                        return ""
                    existing['star'] = existing.get('pvalue', pd.Series([np.nan]*len(existing))).apply(_mk_star)
                    # Rewrite with unified column order
                    cols = ["emotion", "mode", "pair_key", "variable", "coefficient", "pvalue", "star"]
                    for c in cols:
                        if c not in existing.columns:
                            existing[c] = np.nan
                    existing = existing[cols]
                    existing.to_csv(summary_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"  Exogenous coefficients summary migrate-check failed: {e}")
            # Append rows (without repeating header)
            with open(summary_path, "a", encoding="utf-8-sig", newline="") as f:
                if len(df_rows) > 0:
                    df_rows.to_csv(f, header=False, index=False)
        print(f"  Exogenous coefficients summary updated: {summary_path} (+{len(df_rows)} rows)")
    except Exception as e:
        print(f"  Exogenous coefficients summary update failed: {summary_path}, error={e}")


def export_ecm_coefficients_summary(output_dir: str, coint_results: List[Dict[str, object]]):
    """
    Summarize ECM, short-run, and long-run coefficients (with significance) for 8 emotions,
    and export two CSV files (forward and reverse).

    Args:
        output_dir: output directory
        coint_results: list of cointegration/ECM results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    
    # 8 emotions
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    
    # Helper: significance stars
    def _get_star(pval):
        try:
            if pd.isna(pval):
                return ""
            p = float(pval)
            if p < 0.01:
                return "***"
            if p < 0.05:
                return "**"
            if p < 0.10:
                return "*"
            return ""
        except Exception:
            return ""
    
    # Helper: value formatting
    def _format_value(val):
        try:
            if pd.isna(val) or val == "":
                return ""
            # If scientific notation is stored as a string, convert first
            if isinstance(val, str):
                try:
                    val = float(val)
                except:
                    return val
            return f"{float(val):.4f}"
        except Exception:
            return ""
    
    # Helper: p-value formatting
    def _format_pvalue(pval):
        try:
            if pd.isna(pval) or pval == "":
                return ""
            # If scientific notation is stored as a string, convert first
            if isinstance(pval, str):
                try:
                    pval = float(pval)
                except:
                    return pval
            return f"{float(pval):.4f}"
        except Exception:
            return ""

    # z critical value for 95% CI
    try:
        from scipy import stats

        z_critical = stats.norm.ppf(0.975)
    except Exception:
        stats = None
        z_critical = None
    
    # Build a mapping from (emotion, direction) to coint_result
    result_map = {}  # key: (emotion, direction), value: coint_result

    # Stacked long table (forward + reverse), first column is direction
    stacked_rows = []  # each row: direction, emotion, 3 coefficients + their p-values
    
    # Scan coint_results and identify emotion and direction for each entry
    for result in coint_results:
        result_series1 = str(result.get('series1', '')).lower()
        result_series2 = str(result.get('series2', '')).lower()
        
        # Identify emotion
        matched_emotion = None
        for emotion in emotions:
            emotion_lower = emotion.lower()
            if emotion_lower in result_series1 or emotion_lower in result_series2:
                matched_emotion = emotion
                break
        
        if not matched_emotion:
            continue
        
        # Identify direction:
        # forward: climate in series1, covid in series2
        # reverse: covid in series1, climate in series2
        direction = None
        if 'climate' in result_series1 and 'covid' in result_series2:
            direction = "forward"
        elif 'covid' in result_series1 and 'climate' in result_series2:
            direction = "reverse"
        
        if direction:
            key = (matched_emotion.lower(), direction)
            # If a result already exists, prefer the one with more complete coefficient info
            if key not in result_map:
                result_map[key] = result
            else:
                # If new result has coefficient info but old one doesn't, replace
                old_result = result_map[key]
                old_has_coef = 'ecm_coefficient' in old_result and not pd.isna(old_result.get('ecm_coefficient', np.nan))
                new_has_coef = 'ecm_coefficient' in result and not pd.isna(result.get('ecm_coefficient', np.nan))
                if new_has_coef and not old_has_coef:
                    result_map[key] = result
    
    # Read from coint_results directly to keep original values
    for direction in ["forward", "reverse"]:
        rows = []
        rows_sig = []
        rows_ci_low = []
        rows_ci_high = []
        
        for emotion in emotions:
            # Look up in result_map
            key = (emotion.lower(), direction)
            matched_result = result_map.get(key)
            
            if matched_result:
                # Read raw values directly from the result dict
                ecm_coef = matched_result.get('ecm_coefficient', np.nan)
                ecm_pval = matched_result.get('ecm_pvalue', np.nan)
                short_coef = matched_result.get('short_run_coefficients', {}).get('delta_x', np.nan)
                short_pval = matched_result.get('short_run_coefficients', {}).get('delta_x_pvalue', np.nan)
                long_coef = matched_result.get('long_run_coefficient', np.nan)
                long_pval = matched_result.get('long_run_pvalue', np.nan)
                
                # Prefer saved confidence intervals (Delta method)
                long_ci_lower = matched_result.get('long_run_ci_lower', np.nan)
                long_ci_upper = matched_result.get('long_run_ci_upper', np.nan)

                # 95% confidence intervals
                ecm_ci = (np.nan, np.nan)
                short_ci = (np.nan, np.nan)
                # Use saved long-run CI if available; otherwise recompute
                if pd.notna(long_ci_lower) and pd.notna(long_ci_upper):
                    long_ci = (long_ci_lower, long_ci_upper)
                else:
                    long_ci = (np.nan, np.nan)
                
                fp = matched_result.get("_fitted_params", {})
                params = fp.get("params", None)
                cov = fp.get("cov", None)
                design_cols = fp.get("design_cols", [])
                if z_critical is not None and params is not None and cov is not None:
                    try:
                        params_arr = np.asarray(params, dtype=float)
                        cov_arr = np.asarray(cov, dtype=float)
                        # ECM: y_lag
                        ylag_idx = design_cols.index("y_lag") if "y_lag" in design_cols else None
                        if ylag_idx is not None and ylag_idx < len(params_arr) and ylag_idx < cov_arr.shape[0]:
                            var = cov_arr[ylag_idx, ylag_idx]
                            if pd.notna(var):
                                se = np.sqrt(max(0, var))
                                ecm_ci = (ecm_coef - z_critical * se, ecm_coef + z_critical * se)
                        # Short-run: dx_lag0 / delta_x
                        dx_idx = design_cols.index("dx_lag0") if "dx_lag0" in design_cols else None
                        if dx_idx is not None and dx_idx < len(params_arr) and dx_idx < cov_arr.shape[0]:
                            var = cov_arr[dx_idx, dx_idx]
                            if pd.notna(var):
                                se = np.sqrt(max(0, var))
                                short_ci = (short_coef - z_critical * se, short_coef + z_critical * se)
                        # Long-run: if still missing, recompute via Delta method
                        if pd.isna(long_ci[0]) or pd.isna(long_ci[1]):
                            xlag_idx = design_cols.index("x_lag") if "x_lag" in design_cols else None
                            if (
                                ylag_idx is not None
                                and xlag_idx is not None
                                and ylag_idx < len(params_arr)
                                and xlag_idx < len(params_arr)
                                and ylag_idx < cov_arr.shape[0]
                                and xlag_idx < cov_arr.shape[1]
                            ):
                                a = params_arr[ylag_idx]
                                d = params_arr[xlag_idx]
                                if abs(a) > 1e-10:
                                    var_a = cov_arr[ylag_idx, ylag_idx]
                                    var_d = cov_arr[xlag_idx, xlag_idx]
                                cov_ad = cov_arr[ylag_idx, xlag_idx]
                                if pd.notna(var_a) and pd.notna(var_d) and pd.notna(cov_ad):
                                    var_long = (1 / (a ** 2)) * var_d + (d ** 2 / (a ** 4)) * var_a - 2 * (d / (a ** 3)) * cov_ad
                                    se_long = np.sqrt(max(0, var_long))
                                    long_ci = (long_coef - z_critical * se_long, long_coef + z_critical * se_long)
                    except Exception:
                        pass
            else:
                # Not found; use empty values
                ecm_coef = ecm_pval = np.nan
                short_coef = short_pval = np.nan
                long_coef = long_pval = np.nan
                ecm_ci = short_ci = long_ci = (np.nan, np.nan)
            
            ecm_star = _get_star(ecm_pval)
            short_star = _get_star(short_pval)
            long_star = _get_star(long_pval)

            # Append to stacked long table (keep ECM/short-run/long-run and their p-values)
            stacked_rows.append(
                {
                    "direction": direction,
                    "emotion": emotion.upper(),
                    "ecm_coefficient": _format_value(ecm_coef),
                    "ecm_pvalue": _format_pvalue(ecm_pval),
                    "short_run_coefficient": _format_value(short_coef),
                    "short_run_pvalue": _format_pvalue(short_pval),
                    "long_run_coefficient": _format_value(long_coef),
                    "long_run_pvalue": _format_pvalue(long_pval),
                }
            )
            
            # Collect coefficient values
            rows.append({
                "emotion": emotion.upper(),
                "ecm_coefficient": _format_value(ecm_coef),
                "short_run_coefficient": _format_value(short_coef),
                "long_run_coefficient": _format_value(long_coef)
            })
            
            # Collect significance stars
            rows_sig.append({
                "emotion": emotion.upper(),
                "ecm_coefficient": ecm_star,
                "short_run_coefficient": short_star,
                "long_run_coefficient": long_star
            })

            # Collect confidence intervals
            rows_ci_low.append({
                "emotion": emotion.upper(),
                "ecm_coefficient": _format_value(ecm_ci[0]),
                "short_run_coefficient": _format_value(short_ci[0]),
                "long_run_coefficient": _format_value(long_ci[0])
            })
            rows_ci_high.append({
                "emotion": emotion.upper(),
                "ecm_coefficient": _format_value(ecm_ci[1]),
                "short_run_coefficient": _format_value(short_ci[1]),
                "long_run_coefficient": _format_value(long_ci[1])
            })
        
        # Save CSV (transpose)
        df = pd.DataFrame(rows)
        df_sig = pd.DataFrame(rows_sig)
        df_ci_low = pd.DataFrame(rows_ci_low)
        df_ci_high = pd.DataFrame(rows_ci_high)
        
        # Transpose: set emotion as index, then transpose
        df_transposed = df.set_index('emotion').T
        df_sig_transposed = df_sig.set_index('emotion').T
        df_ci_low_transposed = df_ci_low.set_index('emotion').T
        df_ci_high_transposed = df_ci_high.set_index('emotion').T
        
        # Reset index so original column names become the first column
        df_transposed = df_transposed.reset_index()
        df_sig_transposed = df_sig_transposed.reset_index()
        df_ci_low_transposed = df_ci_low_transposed.reset_index()
        df_ci_high_transposed = df_ci_high_transposed.reset_index()
        
        # Rename the first column and add suffixes for significance/CI rows
        df_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_sig_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_sig_transposed['coefficient_type'] = df_sig_transposed['coefficient_type'] + '_sig'
        df_ci_low_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_ci_low_transposed['coefficient_type'] = df_ci_low_transposed['coefficient_type'] + '_ci_low'
        df_ci_high_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_ci_high_transposed['coefficient_type'] = df_ci_high_transposed['coefficient_type'] + '_ci_high'
        
        # Interleave coefficient/CI/significance rows by coefficient type
        result_rows = []
        for coef_type in ['ecm_coefficient', 'short_run_coefficient', 'long_run_coefficient']:
            # Add coefficient row
            coef_row = df_transposed[df_transposed['coefficient_type'] == coef_type].iloc[0].to_dict()
            result_rows.append(coef_row)
            # Add CI rows (lower/upper)
            ci_low_row = df_ci_low_transposed[df_ci_low_transposed['coefficient_type'] == coef_type + '_ci_low'].iloc[0].to_dict()
            result_rows.append(ci_low_row)
            ci_high_row = df_ci_high_transposed[df_ci_high_transposed['coefficient_type'] == coef_type + '_ci_high'].iloc[0].to_dict()
            result_rows.append(ci_high_row)
            # Add significance row
            sig_row = df_sig_transposed[df_sig_transposed['coefficient_type'] == coef_type + '_sig'].iloc[0].to_dict()
            result_rows.append(sig_row)
        
        df_final = pd.DataFrame(result_rows)
        filename = f"ecm_coefficients_summary_{direction}.csv"
        filepath = os.path.join(output_dir, filename)
        df_final.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  ECM coefficients summary ({direction}) saved: {filepath}")

    # Extra CSV: stack forward/reverse with direction as the first column
    if stacked_rows:
        # Fix column order to avoid drift due to missing values
        col_order = [
            "direction",
            "emotion",
            "ecm_coefficient",
            "ecm_pvalue",
            "short_run_coefficient",
            "short_run_pvalue",
            "long_run_coefficient",
            "long_run_pvalue",
        ]
        stacked_df = pd.DataFrame(stacked_rows)
        for c in col_order:
            if c not in stacked_df.columns:
                stacked_df[c] = ""
        stacked_df = stacked_df[col_order]

        stacked_filename = "ecm_coefficients_summary_stacked.csv"
        stacked_filepath = os.path.join(output_dir, stacked_filename)
        try:
            stacked_df.to_csv(stacked_filepath, index=False, encoding="utf-8-sig")
            print(f"  ECM coefficients summary (stacked, with p-values) saved: {stacked_filepath}")
        except Exception as e:
            print(f"  Failed to save stacked ECM coefficients summary: {e}")


def export_ardl_bounds_summary(output_dir: str, coint_results: List[Dict[str, object]]):
    """
    Export ARDL bounds test summary for 8 emotions and both directions to one CSV:
    emotion, direction, optimal_lags, F_statistic(with stars), 1% bounds (I0-I1), 5% bounds (I0-I1), cointegration.

    Star rule: based on whether F exceeds I(0)/I(1) critical values
    (F > 1% I(1) upper -> ***, > 5% I(1) upper -> **, > 5% I(0) lower -> *).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

    def _extract_emotion(name: str) -> str:
        if not isinstance(name, str):
            return ""
        if name.startswith("climate_"):
            return name.replace("climate_", "").replace("_score_freq", "")
        if name.startswith("covid_"):
            return name.replace("covid_", "").replace("_score_freq", "")
        return ""

    def _get_direction(s1: str, s2: str) -> str:
        if isinstance(s1, str) and isinstance(s2, str):
            if s1.startswith("climate_") and s2.startswith("covid_"):
                return "climate→covid"
            if s1.startswith("covid_") and s2.startswith("climate_"):
                return "covid→climate"
        return f"{s1}->{s2}"

    def _format_bounds(v0, v1) -> str:
        try:
            if pd.isna(v0) or pd.isna(v1):
                return ""
            return f"{float(v0):.2f} - {float(v1):.2f}"
        except Exception:
            return ""

    def _format_f_with_bounds_stars(f_val, res) -> str:
        """
        Add stars to the F-statistic based on whether it exceeds I(0)/I(1) critical values
        (Pesaran bounds-test convention):
        F > 1% I(1) upper -> ***; F > 5% I(1) upper -> **; F > 5% I(0) lower -> *.
        """
        try:
            if pd.isna(f_val):
                return ""
            f = float(f_val)
            bound_1pct_I1 = res.get("bound_1pct_I1", 7.84)
            bound_5pct_I1 = res.get("bound_5pct_I1", 4.61)
            bound_5pct_I0 = res.get("bound_5pct_I0", 3.17)
            stars = ""
            if not pd.isna(bound_1pct_I1) and f > float(bound_1pct_I1):
                stars = "***"
            elif not pd.isna(bound_5pct_I1) and f > float(bound_5pct_I1):
                stars = "**"
            elif not pd.isna(bound_5pct_I0) and f > float(bound_5pct_I0):
                stars = "*"
            return f"{f:.4f}{stars}"
        except Exception:
            return ""


    def _find_optimal_lags(s1: str, s2: str) -> str:
        """
        Find the ECM result that matches (series1, series2) and read selected_p/selected_q as optimal lags (p, q).
        """
        for res in coint_results:
            if res.get("series1") == s1 and res.get("series2") == s2 and "selected_p" in res and "selected_q" in res:
                p_val = res.get("selected_p", np.nan)
                q_val = res.get("selected_q", np.nan)
                try:
                    if pd.isna(p_val) and pd.isna(q_val):
                        return ""
                    return f"({int(p_val)}, {int(q_val)})"
                except Exception:
                    return ""
        return ""

    rows = []
    for res in coint_results:
        if "f_statistic" not in res:
            continue  # keep ARDL bounds test results only

        s1 = res.get("series1", "")
        s2 = res.get("series2", "")
        emotion = _extract_emotion(s1) or _extract_emotion(s2)
        if emotion not in emotions:
            continue

        direction = _get_direction(s1, s2)
        optimal_lags = _find_optimal_lags(s1, s2)
        f_with_stars = _format_f_with_bounds_stars(res.get("f_statistic", np.nan), res)
        bounds_1pct = _format_bounds(res.get("bound_1pct_I0", np.nan), res.get("bound_1pct_I1", np.nan))
        bounds_5pct = _format_bounds(res.get("bound_5pct_I0", np.nan), res.get("bound_5pct_I1", np.nan))
        cointegration = "yes" if res.get("bounds_result") == "cointegrated" else "no"

        rows.append(
            {
                "emotion": emotion,
                "direction": direction,
                "optimal_lags": optimal_lags,
                "F_statistic": f_with_stars,
                "1% Bounds (I0 - I1)": bounds_1pct,
                "5% Bounds (I0 - I1)": bounds_5pct,
                "cointegration": cointegration,
            }
        )

    if not rows:
        print("No ARDL bounds test results found for summary export.")
        return

    df_bounds = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "ardl_bounds_summary.csv")
    try:
        df_bounds.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"ARDL bounds test summary exported: {summary_path}")
    except Exception as e:
        print(f"ARDL bounds test summary export failed: {summary_path}, error={e}")


def plot_short_long_coefficients_forest(output_dir: str, coint_results: List[Dict[str, object]], confidence_level: float = 0.95):
    """
    Generate a two-panel forest plot for short-run and long-run coefficients (with confidence intervals).

    Panel A: short-run coefficients
    Panel B: long-run coefficients

    Args:
        output_dir: output directory
        coint_results: list of cointegration/ECM results
        confidence_level: confidence level (default 0.95)
    """
    try:
        from scipy import stats
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 8 emotions
        emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
        
        # Build result mapping
        result_map = {}
        for result in coint_results:
            result_series1 = str(result.get('series1', '')).lower()
            result_series2 = str(result.get('series2', '')).lower()
            
            matched_emotion = None
            for emotion in emotions:
                emotion_lower = emotion.lower()
                if emotion_lower in result_series1 or emotion_lower in result_series2:
                    matched_emotion = emotion
                    break
            
            if not matched_emotion:
                continue
            
            direction = None
            if 'climate' in result_series1 and 'covid' in result_series2:
                direction = "forward"
            elif 'covid' in result_series1 and 'climate' in result_series2:
                direction = "reverse"
            
            if direction:
                key = (matched_emotion.lower(), direction)
                if key not in result_map:
                    result_map[key] = result
                else:
                    old_has_coef = 'ecm_coefficient' in result_map[key] and not pd.isna(result_map[key].get('ecm_coefficient', np.nan))
                    new_has_coef = 'ecm_coefficient' in result and not pd.isna(result.get('ecm_coefficient', np.nan))
                    if new_has_coef and not old_has_coef:
                        result_map[key] = result
        
        # Significance star helper
        def _star(p):
            try:
                p = float(p)
            except Exception:
                return ""
            if np.isnan(p):
                return ""
            if p < 0.01:
                return "***"
            if p < 0.05:
                return "**"
            if p < 0.10:
                return "*"
            return ""
        
        # Generate forest plots for each direction; keep paths for the combined figure
        forest_paths = {}
        for direction in ["forward", "reverse"]:
            # Collect short-run and long-run coefficient data
            short_run_data = []  # [(emotion, coefficient, lower_ci, upper_ci, pval), ...]
            long_run_data = []   # [(emotion, coefficient, lower_ci, upper_ci, pval), ...]
            
            for emotion in emotions:
                key = (emotion.lower(), direction)
                matched_result = result_map.get(key)
                
                if matched_result:
                    # Fitted parameter info
                    fp = matched_result.get("_fitted_params", {})
                    params = fp.get("params", None)
                    cov = fp.get("cov", None)
                    design_cols = fp.get("design_cols", [])
                    
                    # Short-run coefficient (delta_x)
                    short_coef = matched_result.get('short_run_coefficients', {}).get('delta_x', np.nan)
                    short_pval = matched_result.get('short_run_coefficients', {}).get('delta_x_pvalue', np.nan)
                    
                    # CI for short-run coefficient
                    if not pd.isna(short_coef) and not pd.isna(short_pval) and params is not None and cov is not None:
                        try:
                            # Find index of delta_x / dx_lag0
                            dx_idx = None
                            if 'dx_lag0' in design_cols:
                                dx_idx = design_cols.index('dx_lag0')
                            
                            if dx_idx is not None and dx_idx < len(params) and dx_idx < cov.shape[0]:
                                se = np.sqrt(cov[dx_idx, dx_idx])
                                z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                                lower_ci = short_coef - z_critical * se
                                upper_ci = short_coef + z_critical * se
                                short_run_data.append((emotion.upper(), short_coef, lower_ci, upper_ci, short_pval))
                        except Exception:
                            # If CI computation fails, approximate SE from p-value (fallback)
                            try:
                                if short_pval > 0:
                                    z_stat = abs(stats.norm.ppf(short_pval / 2))
                                    se = abs(short_coef) / z_stat if z_stat > 0 else np.nan
                                    if not pd.isna(se):
                                        z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                                        lower_ci = short_coef - z_critical * se
                                        upper_ci = short_coef + z_critical * se
                                        short_run_data.append((emotion.upper(), short_coef, lower_ci, upper_ci, short_pval))
                            except Exception:
                                pass
                    
                    # Long-run coefficient
                    long_coef = matched_result.get('long_run_coefficient', np.nan)
                    long_pval = matched_result.get('long_run_pvalue', np.nan)
                    
                    # Prefer saved CIs (computed in ardl_ecm_model via Delta method and t distribution)
                    long_ci_lower = matched_result.get('long_run_ci_lower', np.nan)
                    long_ci_upper = matched_result.get('long_run_ci_upper', np.nan)
                    
                    if not pd.isna(long_coef):
                        # If saved CI exists, use it directly (recommended; uses t distribution)
                        if pd.notna(long_ci_lower) and pd.notna(long_ci_upper):
                            long_run_data.append((emotion.upper(), long_coef, long_ci_lower, long_ci_upper, long_pval))
                        # Otherwise, recompute (Delta method)
                        elif params is not None and cov is not None:
                            try:
                                # Long-run coefficient = -params[x_lag] / params[y_lag]
                                # Compute CI for this nonlinear function
                                ylag_idx = design_cols.index('y_lag') if 'y_lag' in design_cols else None
                                xlag_idx = design_cols.index('x_lag') if 'x_lag' in design_cols else None
                                
                                if ylag_idx is not None and xlag_idx is not None:
                                    a = params[ylag_idx]  # y_lag coefficient
                                    d = params[xlag_idx]  # x_lag coefficient
                                    
                                    if abs(a) > 1e-10:
                                        # Delta method: variance of long-run coefficient
                                        # var(long_run) ≈ (1/a²) * var(d) + (d²/a⁴) * var(a) - 2*(d/a³) * cov(a,d)
                                        # Handle covariance matrix as DataFrame or numpy array
                                        if hasattr(cov, 'iloc'):
                                            var_a = cov.iloc[ylag_idx, ylag_idx]
                                            var_d = cov.iloc[xlag_idx, xlag_idx]
                                            cov_ad = cov.iloc[ylag_idx, xlag_idx] if ylag_idx < cov.shape[0] and xlag_idx < cov.shape[1] else 0
                                        else:
                                            var_a = cov[ylag_idx, ylag_idx]
                                            var_d = cov[xlag_idx, xlag_idx]
                                            cov_ad = cov[ylag_idx, xlag_idx] if ylag_idx < cov.shape[0] and xlag_idx < cov.shape[1] else 0
                                        
                                        var_long = (1 / (a**2)) * var_d + (d**2 / (a**4)) * var_a - 2 * (d / (a**3)) * cov_ad
                                        se_long = np.sqrt(max(0, var_long))  # ensure non-negative
                                        
                                        # Try t distribution if possible; otherwise use normal approximation
                                        try:
                                            # Degrees of freedom (ideally from model results); rough fallback otherwise
                                            n_obs = len(params) + 10  # rough fallback
                                            k_params = len(params)
                                            df = max(1, n_obs - k_params)
                                            t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
                                            lower_ci = long_coef - t_critical * se_long
                                            upper_ci = long_coef + t_critical * se_long
                                        except Exception:
                                            # If t distribution fails, use normal approximation
                                            z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                                            lower_ci = long_coef - z_critical * se_long
                                            upper_ci = long_coef + z_critical * se_long
                                        
                                        long_run_data.append((emotion.upper(), long_coef, lower_ci, upper_ci, long_pval))
                            except Exception:
                                # If Delta method fails, approximate from p-value (last resort)
                                try:
                                    if not pd.isna(long_pval) and long_pval > 0:
                                        z_stat = abs(stats.norm.ppf(long_pval / 2))
                                        se = abs(long_coef) / z_stat if z_stat > 0 else np.nan
                                        if not pd.isna(se):
                                            z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                                            lower_ci = long_coef - z_critical * se
                                            upper_ci = long_coef + z_critical * se
                                            long_run_data.append((emotion.upper(), long_coef, lower_ci, upper_ci, long_pval))
                                except Exception:
                                    pass
            
            # If insufficient data, skip plotting
            if not short_run_data and not long_run_data:
                continue
            
            # Create a two-panel figure
            # Direction label
            direction_label = "COVID→Climate Change" if direction == "forward" else "Climate Change→COVID"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
            # Use direction label as the figure title
            fig.suptitle(f'{direction_label}', fontsize=16, fontweight='bold')

            # Add a small panel label at top-left (forward: a, reverse: b)
            panel_label = "a" if direction == "forward" else "b"
            try:
                fig.text(0.02, 0.98, panel_label, fontsize=16, fontweight='bold', va='top', ha='left')
            except Exception:
                pass
            
            # Panel A: short-run coefficient
            if short_run_data:
                # No panel title
                ax1.set_title('')
                emotions_short, coefs_short, lowers_short, uppers_short, pvals_short = zip(*short_run_data)
                y_pos_short = np.arange(len(emotions_short))
                
                # Plot confidence intervals
                for i, (emotion, coef, lower, upper, pval) in enumerate(short_run_data):
                    ax1.plot([lower, upper], [i, i], color='#1f77b4', linewidth=2.5, alpha=0.8)
                    ax1.plot(coef, i, 'o', color='#1f77b4', markersize=8)
                    # Label coefficient value
                    ax1.text(coef, i + 0.15, f'{coef:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # y-axis labels with significance stars
                emotion_labels_short = [f"{emotion} {_star(pval)}" for emotion, pval in zip(emotions_short, pvals_short)]
                
                ax1.set_yticks(y_pos_short)
                ax1.set_yticklabels(emotion_labels_short)
                ax1.set_xlabel('Short-run coefficient', fontsize=12)
                ax1.set_ylabel('emotions', fontsize=12)
                # Axes only, no grid; keep black dashed zero line
                ax1.grid(False)
                ax1.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                # Use white background
                ax1.set_facecolor('white')
            
            # Panel B: long-run coefficient
            if long_run_data:
                # No panel title
                ax2.set_title('')
                emotions_long, coefs_long, lowers_long, uppers_long, pvals_long = zip(*long_run_data)
                y_pos_long = np.arange(len(emotions_long))
                
                # Plot confidence intervals
                for i, (emotion, coef, lower, upper, pval) in enumerate(long_run_data):
                    ax2.plot([lower, upper], [i, i], color='#1f77b4', linewidth=2.5, alpha=0.8)
                    ax2.plot(coef, i, 'o', color='#1f77b4', markersize=8)
                    # Label coefficient value
                    ax2.text(coef, i + 0.15, f'{coef:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # y-axis labels with significance stars
                emotion_labels_long = [f"{emotion} {_star(pval)}" for emotion, pval in zip(emotions_long, pvals_long)]
                
                ax2.set_yticks(y_pos_long)
                ax2.set_yticklabels(emotion_labels_long)
                ax2.set_xlabel('Long-run coefficient', fontsize=12)
                ax2.set_ylabel('emotions', fontsize=12)
                # Axes only, no grid; keep black dashed zero line
                ax2.grid(False)
                ax2.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                # Use white background
                ax2.set_facecolor('white')
            
            # Layout
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, f"coefficients_forest_plot_{direction}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            forest_paths[direction] = output_path
            print(f"  Short-run/long-run coefficient CI plot ({direction}) saved: {output_path}")

        # Combined figure: stack forward on top of reverse
        try:
            import matplotlib.image as mpimg

            forward_path = forest_paths.get("forward")
            reverse_path = forest_paths.get("reverse")
            if forward_path and reverse_path and os.path.exists(forward_path) and os.path.exists(reverse_path):
                img_forward = mpimg.imread(forward_path)
                img_reverse = mpimg.imread(reverse_path)

                # New canvas: two subplots stacked vertically
                fig_combined, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 12))

                ax_top.imshow(img_forward)
                ax_top.axis('off')

                ax_bottom.imshow(img_reverse)
                ax_bottom.axis('off')

                plt.tight_layout()
                combined_path = os.path.join(output_dir, "coefficients_forest_plot_combined.png")
                fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
                plt.close(fig_combined)
                print(f"  Combined forest plot (forward+reverse) saved: {combined_path}")
        except Exception as _:
            # If combining fails, keep individual outputs
            pass
        
    except Exception as e:
        print(f"  Failed to generate coefficient CI plot: {str(e)}")
        import traceback
        traceback.print_exc()


## Removed: residual diagnostics summary export (DW/LBQ/normality/heteroskedasticity)


def summarize_effects(csv_dir: str, pair_key: str, ecm_result: Dict[str, object], irf: Dict[str, object], horizons: List[int]):
    """Summarize contemporaneous (h=0), cumulative-at-h, and long-run effects."""
    try:
        b = ecm_result.get("short_run_coefficients", {}).get("delta_x", np.nan)
        long_run = irf.get("long_run", np.nan)
        h_arr = irf.get("h", np.array([]))
        y_arr = irf.get("irf", np.array([]))
        rows: List[Dict[str, object]] = []
        for H in horizons:
            if len(y_arr) > H:
                cumulative = float(y_arr[H])  # y is a level response, already cumulative
            else:
                cumulative = np.nan
            rows.append({
                "horizon": H,
                "contemporaneous_delta": float(b) if pd.notna(b) else np.nan,
                "cumulative_level@H": cumulative,
                "long_run": float(long_run) if pd.notna(long_run) else np.nan,
            })
        if rows:
            df_out = pd.DataFrame(rows)
            out_path = os.path.join(csv_dir, f"effects_summary_{pair_key.replace(':','_').replace('/','_')}.csv")
            df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Extra: append a quick-check record to the overview table (prefer H=14, else max H)
        try:
            # Parse emotion and direction
            emotion = ""
            try:
                if "climate_" in pair_key:
                    emotion = pair_key.split("climate_")[1].split("_score_freq")[0]
            except Exception:
                pass
            left_right = pair_key.split("_<-_")
            direction = ""
            if len(left_right) == 2:
                left, right = left_right[0], left_right[1]
                if "climate_" in left and "covid_" in right:
                    direction = "covid_to_climate"
                elif "covid_" in left and "climate_" in right:
                    direction = "climate_to_covid"
            # Target horizon H
            H_target = 14
            if not isinstance(horizons, (list, tuple)) or (14 not in horizons):
                try:
                    H_target = int(np.max(horizons)) if horizons else 14
                except Exception:
                    H_target = 14
            level_at_H = float(y_arr[H_target]) if (len(y_arr) > H_target and np.isfinite(y_arr[H_target])) else np.nan
            ecm_coef = ecm_result.get("ecm_coefficient", np.nan)
            # Write/append
            overview_path = os.path.join(csv_dir, "irf_params_overview.csv")
            row_overview = {
                "emotion": str(emotion),
                "direction": str(direction),
                "pair_key": str(pair_key),
                "b_delta_x": float(b) if pd.notna(b) else np.nan,
                "lambda_long_run": float(long_run) if pd.notna(long_run) else np.nan,
                "ecm_coefficient(a)": float(ecm_coef) if pd.notna(ecm_coef) else np.nan,
                f"level_at_H{H_target}": level_at_H,
            }
            df_row = pd.DataFrame([row_overview])
            if not os.path.exists(overview_path):
                df_row.to_csv(overview_path, index=False, encoding="utf-8-sig")
            else:
                with open(overview_path, "a", encoding="utf-8-sig", newline="") as f:
                    df_row.to_csv(f, header=False, index=False)
        except Exception:
            pass
    except Exception:
        pass


def _derive_bounds_table_for_appendix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ARDL bounds test rows and derive emotion + direction for appendix table."""
    bounds = df[df["f_statistic"].notna()].copy()

    pat = re.compile(r"^(climate|covid)_(.+?)_score_freq$")

    def parse_series(s: object):
        m = pat.match(str(s))
        if not m:
            return None, None
        return m.group(1), m.group(2)

    left = bounds["series1"].apply(parse_series)
    right = bounds["series2"].apply(parse_series)
    bounds["side1"] = [a for a, _ in left]
    bounds["emotion1"] = [b for _, b in left]
    bounds["side2"] = [a for a, _ in right]
    bounds["emotion2"] = [b for _, b in right]
    bounds["emotion"] = bounds["emotion1"].where(bounds["emotion1"].notna(), bounds["emotion2"])

    def direction(row: pd.Series) -> str:
        if row["side1"] == "climate" and row["side2"] == "covid":
            return "covid_to_climate"
        if row["side1"] == "covid" and row["side2"] == "climate":
            return "climate_to_covid"
        return "unknown"

    bounds["direction"] = bounds.apply(direction, axis=1)

    keep = [
        "emotion",
        "direction",
        "f_statistic",
        "bounds_result",
        "long_run_result",
        "n_obs",
        "df1",
        "df2",
        "p_value",
        "bound_5pct_I0",
        "bound_5pct_I1",
        "bound_1pct_I0",
        "bound_1pct_I1",
        "series1",
        "series2",
    ]
    keep = [c for c in keep if c in bounds.columns]
    bounds = bounds[keep]
    bounds = bounds.rename(
        columns={
            "f_statistic": "F",
            "bounds_result": "bounds",
            "long_run_result": "long_run",
            "n_obs": "n",
            "df1": "df_num",
            "df2": "df_den",
            "p_value": "p",
            "bound_5pct_I0": "I0_5%",
            "bound_5pct_I1": "I1_5%",
            "bound_1pct_I0": "I0_1%",
            "bound_1pct_I1": "I1_1%",
        }
    )
    return bounds


def _build_placebo_appendix_table(base_dir: str, output_dir: str) -> None:
    """Build comparison table (real vs placebo_shuffle) for appendix."""
    try:
        real_path = os.path.join(base_dir, "stationarity_results_causality.csv")
        placebo_path = os.path.join(base_dir, "stationarity_results_placebo_shuffle_causality.csv")
        out_dir = output_dir
        out_path = os.path.join(out_dir, "appendix_placebo_bounds_table.csv")

        if not (os.path.exists(real_path) and os.path.exists(placebo_path)):
            print("Appendix placebo table skipped: causality CSVs not found.")
            return

        real = pd.read_csv(real_path)
        placebo = pd.read_csv(placebo_path)

        real_t = _derive_bounds_table_for_appendix(real).add_prefix("real_")
        placebo_t = _derive_bounds_table_for_appendix(placebo).add_prefix("placebo_")

        real_t = real_t.rename(columns={"real_emotion": "emotion", "real_direction": "direction"})
        placebo_t = placebo_t.rename(columns={"placebo_emotion": "emotion", "placebo_direction": "direction"})

        merged = pd.merge(real_t, placebo_t, on=["emotion", "direction"], how="outer")

        cols = [
            "emotion",
            "direction",
            "real_F",
            "real_bounds",
            "real_long_run",
            "real_p",
            "real_n",
            "placebo_F",
            "placebo_bounds",
            "placebo_long_run",
            "placebo_p",
            "placebo_n",
            "real_series1",
            "real_series2",
            "placebo_series1",
            "placebo_series2",
            "real_I0_5%",
            "real_I1_5%",
            "real_I0_1%",
            "real_I1_1%",
            "placebo_I0_5%",
            "placebo_I1_5%",
            "placebo_I0_1%",
            "placebo_I1_1%",
        ]
        cols = [c for c in cols if c in merged.columns]
        merged = merged[cols].sort_values(["emotion", "direction"], kind="stable")

        os.makedirs(out_dir, exist_ok=True)
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nAppendix placebo bounds table written to: {out_path} (rows={len(merged)})")
    except Exception as e:
        print(f"Failed to build appendix placebo table: {str(e)}")


def add_stars(val, pval):
    """Append significance stars to a coefficient based on p-value."""
    try:
        v = float(val)
    except Exception:
        return ""
    try:
        if pd.isna(pval):
            return f"{v:.4f}"
        p = float(pval)
    except Exception:
        return f"{v:.4f}"
    if p < 0.001:
        return f"{v:.4f}***"
    elif p < 0.01:
        return f"{v:.4f}**"
    elif p < 0.05:
        return f"{v:.4f}*"
    else:
        return f"{v:.4f}"


def get_f_stars(f_val):
    """
    Append significance stars for the ARDL bounds test based on critical values.
    Typical I(1) upper bounds: 7.84 (1%), 4.61 (5%).
    """
    try:
        f = float(f_val)
    except Exception:
        return ""
    if pd.isna(f):
        return ""
    if f > 7.84:
        return f"{f:.2f}***"
    elif f > 4.61:
        return f"{f:.2f}**"
    elif f > 3.17:  # between I(0) and I(1): often inconclusive/weak; mark as *
        return f"{f:.2f}*"
    else:
        return f"{f:.2f}"


def build_supplementary_table_6(output_dir: str) -> None:
    """
    Build Supplementary Table 6 using appendix_placebo_bounds_table.csv and ECM coefficient summaries
    (real sample vs placebo sample with shuffled time order).
    """
    bounds_path = os.path.join(output_dir, "appendix_placebo_bounds_table.csv")
    real_ecm_path = os.path.join(output_dir, "ecm_coefficients_summary_stacked.csv")
    placebo_ecm_path = os.path.join(output_dir, "placebo_placebo_shuffle", "ecm_coefficients_summary_stacked.csv")

    # 1. Load data
    try:
        bounds_df = pd.read_csv(bounds_path)
        real_ecm = pd.read_csv(real_ecm_path)
        placebo_ecm = pd.read_csv(placebo_ecm_path)
    except FileNotFoundError as e:
        print(f"Failed to build Supplementary Table 6 (file not found): {e}")
        print("Please ensure the robustness checks and placebo analysis have been fully executed.")
        return

    # 2. Process F-statistic (Bounds Test)
    if "real_F" not in bounds_df.columns or "placebo_F" not in bounds_df.columns:
        print("Failed to build Supplementary Table 6: missing real_F or placebo_F columns in appendix_placebo_bounds_table.csv.")
        return
    bounds_df["Real F-statistic"] = bounds_df["real_F"].apply(get_f_stars)
    bounds_df["Placebo F-statistic"] = bounds_df["placebo_F"].apply(get_f_stars)

    # Normalize emotion capitalization
    if "emotion" in bounds_df.columns:
        bounds_df["emotion"] = bounds_df["emotion"].astype(str).str.capitalize()

    # Normalize direction labels
    if "direction" in bounds_df.columns:
        bounds_dir_map = {
            "climate_to_covid": "Climate Change -> COVID-19",
            "covid_to_climate": "COVID-19 -> Climate Change",
        }
        bounds_df["direction"] = bounds_df["direction"].map(bounds_dir_map)

    # 3. Process ECT coefficients for real and placebo samples
    if {"ecm_coefficient", "ecm_pvalue"}.issubset(real_ecm.columns):
        real_ecm["Real ECT"] = real_ecm.apply(
            lambda row: add_stars(row["ecm_coefficient"], row["ecm_pvalue"]), axis=1
        )
    else:
        print("Real ECM summary is missing ecm_coefficient or ecm_pvalue; cannot build Real ECT.")
        return

    if {"ecm_coefficient", "ecm_pvalue"}.issubset(placebo_ecm.columns):
        placebo_ecm["Placebo ECT"] = placebo_ecm.apply(
            lambda row: add_stars(row["ecm_coefficient"], row["ecm_pvalue"]), axis=1
        )
    else:
        print("Placebo ECM summary is missing ecm_coefficient or ecm_pvalue; cannot build Placebo ECT.")
        return

    # Normalize emotion capitalization and direction labels in ECM tables
    for df in (real_ecm, placebo_ecm):
        if "emotion" in df.columns:
            df["emotion"] = df["emotion"].astype(str).str.capitalize()

    ecm_dir_map = {
        "forward": "COVID-19 -> Climate Change",
        "reverse": "Climate Change -> COVID-19",
    }
    if "direction" in real_ecm.columns:
        real_ecm["direction"] = real_ecm["direction"].map(ecm_dir_map)
    if "direction" in placebo_ecm.columns:
        placebo_ecm["direction"] = placebo_ecm["direction"].map(ecm_dir_map)

    # Select required columns
    real_sub = real_ecm[["direction", "emotion", "Real ECT"]]
    placebo_sub = placebo_ecm[["direction", "emotion", "Placebo ECT"]]

    # 4. Merge tables
    merged = bounds_df[["direction", "emotion", "Real F-statistic", "Placebo F-statistic"]].copy()
    merged = pd.merge(merged, real_sub, on=["direction", "emotion"], how="left")
    merged = pd.merge(merged, placebo_sub, on=["direction", "emotion"], how="left")

    # Reorder columns to match Supplementary Table 6 layout
    final_table = merged[
        ["direction", "emotion", "Real F-statistic", "Real ECT", "Placebo F-statistic", "Placebo ECT"]
    ].copy()

    # Sort: show COVID->Climate emotions first, then Climate->COVID emotions
    final_table = final_table.sort_values(by=["direction", "emotion"], ascending=[False, True])

    # 5. Export to CSV
    out_file = os.path.join(output_dir, "Supplementary_Table_6_Placebo_Comparison.csv")
    final_table.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"\nSuccessfully generated Supplementary Table 6 and saved to: {out_file}\n")
    try:
        print("--- Table preview ---")
        # to_markdown requires tabulate; ignore if not available
        print(final_table.to_markdown(index=False))
    except Exception:
        print(final_table.head())


def main():
    # Run complete analysis (no CLI args)
    print("=" * 60)
    print("Complete Analysis Mode - All Features Included")
    print("=" * 60)
    
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "NRC_result.csv")
    output_file = os.path.join(script_dir, "stationarity_results.csv")
    output_dir = os.path.join(script_dir, "analysis_results")
    
    # Check input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found {input_file}")
        print("Please ensure NRC_result.csv file is in the current directory")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Output directory: {output_dir}")
    print("Starting complete analysis...")
    # Dedicated exogenous-coefficient export directory
    exog_output_dir = os.path.join(output_dir, "exog_coeffs")
    try:
        os.makedirs(exog_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create exog output dir: {exog_output_dir}, error={e}")
    
    # Default parameters
    args = type('Args', (), {
        'input': input_file,
        'output': output_file,
        'date_col': None,
        # Do not pre-remove weekday effects/trend before unit-root/cointegration tests
        'weekday_ols': False,
        'weekday_trend': False,
        'weekday_poly_degree': 1,
        'compare_trend': False,
        'no_cointegration': False,
        'cointegration_preprocess': True,
        'ardl_ecm': True,
        'granger_test': True,
        'skip_models': False,
        'exog_significant_only': False,
        'exog_threshold': 0.10,
        'out_dir': output_dir,
        'irf': True,
        'irf_horizon': 14,
        'irf_boot': 500,
        'effects_horizons': "0,7,14,30",
        'timeseries_plots': False
    })()
    
    # Execute complete analysis
    try:
        # Load data
        df = pd.read_csv(input_file)
        if df.shape[1] < 3:
            raise ValueError("CSV requires at least date column and two emotion score columns")
        
        # Identify date column
        date_col = args.date_col if args.date_col else df.columns[0]
        
        # Parse date and sort
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().any():
            # Try strict format 'YYYY/M/D'
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y/%m/%d", errors="coerce")
        if df[date_col].isna().any():
            raise ValueError("Date parsing failed, please check date format is 2020/1/20")
        
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Discover emotion columns
        emotion_columns = discover_emotion_columns(df.columns.tolist())
        if not emotion_columns:
            raise ValueError("No matching emotion columns found (climate_* or covid_*).")
        
        # Discover exogenous variables
        exogenous_columns = discover_exogenous_columns(df.columns.tolist())
        try:
            print(f"Found {len(exogenous_columns)} exogenous columns: {exogenous_columns}")
        except Exception:
            pass
        
        print(f"Found {len(emotion_columns)} emotion columns")
        if exogenous_columns:
            print(f"Found {len(exogenous_columns)} exogenous variables: {', '.join(exogenous_columns)}")
        else:
            print("No exogenous variables found")

        # Export weekday-effect/trend regression reports (per current settings)
        try:
            if args.weekday_ols:
                weekday_dir = os.path.join(output_dir, "weekday_ols")
                for col in emotion_columns:
                    export_weekday_ols_report(
                        df[col],
                        df[date_col],
                        series_name=col,
                        add_trend=args.weekday_trend,
                        csv_dir=weekday_dir,
                        polynomial_degree=getattr(args, 'weekday_poly_degree', 1),
                    )
            # Export weekday OLS wide-format summary
            try:
                build_weekday_ols_wide(weekday_dir, outfile="weekday_ols_summary_wide.csv")
            except Exception:
                pass
        except Exception:
            pass
        
        # Real sample + placebo scenario (shuffle time order)
        scenarios = [
            ("real", False),
            ("placebo_shuffle", True),
        ]

        for scenario_label, shuffle_time in scenarios:
            if shuffle_time:
                print(f"\n================ Placebo scenario (shuffled time order): {scenario_label} ================")
                df_scen = df.sample(frac=1.0, replace=False, random_state=123).reset_index(drop=True)
                output_dir_scen = os.path.join(output_dir, f"placebo_{scenario_label}")
                output_file_scen = os.path.join(script_dir, f"stationarity_results_{scenario_label}.csv")
            else:
                print(f"\n================ Real-sample scenario: {scenario_label} ================")
                df_scen = df.copy()
                output_dir_scen = output_dir
                output_file_scen = output_file

            os.makedirs(output_dir_scen, exist_ok=True)
            exog_output_dir_scen = os.path.join(output_dir_scen, "exog_coeffs")
            os.makedirs(exog_output_dir_scen, exist_ok=True)

            # Run stationarity tests
            print("\n=== Running Stationarity Tests ===")
            integ_rows = []
            diffed_df = df_scen[[date_col]].copy()

            for col in emotion_columns:
                base_series = df_scen[col]
                # Run unit-root tests on raw series (no weekday/trend pre-removal)
                d_order, diffed, report = find_integration_order(base_series, col, max_d=2)
                integ_rows.append(report)

                # Save differenced column
                out_col = f"{col}_d{d_order}" if d_order > 0 else col
                diffed_df[out_col] = diffed

            # Save stationarity results
            integ_df = pd.DataFrame(integ_rows)
            front_cols = ["series", "final_order", "decision"]
            other_cols = [c for c in integ_df.columns if c not in front_cols]
            integ_df = integ_df[front_cols + sorted(other_cols)]
            integ_df.to_csv(output_file_scen, index=False, encoding="utf-8-sig")
            print(f"Stationarity test completed, saved to: {output_file_scen}")

            # Save differenced data
            diffed_path = os.path.splitext(output_file_scen)[0] + "_diffed_to_stationary.csv"
            diffed_df.to_csv(diffed_path, index=False, encoding="utf-8-sig")
            print(f"Differenced data saved to: {diffed_path}")

            # Print summary
            print(f"\n=== Stationarity Test Summary ===")
            print(f"Total series: {len(emotion_columns)}")
            print(f"I(0) stationary series: {len(integ_df[integ_df['final_order'] == 0])}")
            print(f"I(1) first difference series: {len(integ_df[integ_df['final_order'] == 1])}")
            print(f"I(2) second difference series: {len(integ_df[integ_df['final_order'] == 2])}")

            # Organize emotion groups for further analysis
            excluded_emotions = {"positive", "negative"}
            emotion_groups = {}
            for _, row in integ_df.iterrows():
                series_name = row["series"]
                if series_name.startswith("climate_"):
                    emotion = series_name.replace("climate_", "").replace("_score_freq", "")
                    if emotion not in emotion_groups:
                        emotion_groups[emotion] = {}
                    emotion_groups[emotion]["climate"] = row
                elif series_name.startswith("covid_"):
                    emotion = series_name.replace("covid_", "").replace("_score_freq", "")
                    if emotion not in emotion_groups:
                        emotion_groups[emotion] = {}
                    emotion_groups[emotion]["covid"] = row

            filtered_emotions = [e for e in sorted(emotion_groups.keys()) if e not in excluded_emotions]

            # Run causality analysis
            print("\n=== Running Causality Analysis ===")
            coint_results = []

            for emotion in filtered_emotions:
                group = emotion_groups[emotion]
                if "climate" not in group or "covid" not in group:
                    continue

                climate_order = group["climate"]["final_order"]
                covid_order = group["covid"]["final_order"]

                climate_col = f"climate_{emotion}_score_freq"
                covid_col = f"covid_{emotion}_score_freq"
                climate_series = df_scen[climate_col]
                covid_series = df_scen[covid_col]

                # Determine analysis method based on integration orders
                if (climate_order == 1 and covid_order == 1) or (climate_order == 0 and covid_order == 1) or (climate_order == 1 and covid_order == 0):
                    # I(1)+I(1) or mixed I(0)/I(1): use ARDL bounds test
                    print(f"\n{emotion.upper()} emotion: I({climate_order}) + I({covid_order}) - Running ARDL bounds tests")
                    ardl_result = ardl_bounds_test(
                        climate_series,
                        covid_series,
                        climate_col,
                        covid_col,
                        dates=df_scen[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                    )
                    coint_results.append(ardl_result)
                    if ardl_result.get("error"):
                        print(f"  Error: {ardl_result['error']}")
                        continue

                    print(f"  F-statistic: {ardl_result['f_statistic']:.4f}")
                    print(f"  Bounds test: {ardl_result['bounds_result']}")
                    print(f"  Long-run relationship: {ardl_result['long_run_result']}")

                    # If bounds test passes, run ECM model + reverse direction
                    if ardl_result["bounds_result"] == "cointegrated":
                        print("  Running ARDL-ECM model...")
                        exog_vars = {}
                        if exogenous_columns:
                            for exog_col in exogenous_columns:
                                exog_vars[exog_col] = df_scen[exog_col]
                        # Add weekday dummies (Tue-Sun), with Monday as baseline
                        try:
                            wd_df = build_weekday_dummies(df_scen[date_col], index=df_scen.index, prefix="wd")
                            for c in wd_df.columns:
                                if c not in exog_vars:
                                    exog_vars[c] = wd_df[c]
                        except Exception:
                            pass

                        ecm_result = ardl_ecm_model(
                            climate_series,
                            covid_series,
                            climate_col,
                            covid_col,
                            dates=df_scen[date_col],
                            weekday_ols=args.weekday_ols,
                            weekday_trend=args.weekday_trend,
                            exog_vars=exog_vars,
                        )
                        coint_results.append(ecm_result)
                        if ecm_result.get("error"):
                            print(f"    Error: {ecm_result['error']}")
                        else:
                            _print_and_export_ecm(
                                "forward",
                                str(emotion),
                                climate_col,
                                covid_col,
                                ecm_result,
                                ardl_result,
                                output_dir_scen,
                                exog_output_dir_scen,
                                args,
                            )

                        # reverse: climate → covid
                        print("  Running reverse ARDL bounds test...")
                        ardl_result_rev = ardl_bounds_test(
                            covid_series,
                            climate_series,
                            covid_col,
                            climate_col,
                            dates=df_scen[date_col],
                            weekday_ols=args.weekday_ols,
                            weekday_trend=args.weekday_trend,
                        )
                        coint_results.append(ardl_result_rev)
                        if ardl_result_rev.get("error"):
                            print(f"  Reverse ARDL bounds test error: {ardl_result_rev['error']}")
                        else:
                            print(f"  Reverse F-statistic: {ardl_result_rev['f_statistic']:.4f}")
                            print(f"  Reverse bounds test: {ardl_result_rev['bounds_result']}")
                            print(f"  Reverse long-run relationship: {ardl_result_rev['long_run_result']}")
                            if ardl_result_rev["bounds_result"] == "cointegrated":
                                print("  Running reverse ARDL-ECM model...")
                                ecm_result_2 = ardl_ecm_model(
                                    covid_series,
                                    climate_series,
                                    covid_col,
                                    climate_col,
                                    dates=df_scen[date_col],
                                    weekday_ols=args.weekday_ols,
                                    weekday_trend=args.weekday_trend,
                                    exog_vars=exog_vars,
                                )
                                coint_results.append(ecm_result_2)
                                if not ecm_result_2.get("error"):
                                    _print_and_export_ecm(
                                        "reverse",
                                        str(emotion),
                                        covid_col,
                                        climate_col,
                                        ecm_result_2,
                                        ardl_result_rev,
                                        output_dir_scen,
                                        exog_output_dir_scen,
                                        args,
                                    )
                            else:
                                print("  Reverse bounds test failed, no reverse ECM model")

                else:
                    # I(0) + I(0): Pesaran ARDL bounds test also applies (critical values cover I(0) case);
                    # run bounds tests for both forward and reverse directions
                    print(f"\n{emotion.upper()} emotion: I(0) + I(0) - Running ARDL bounds tests (applicable for I(0) variables per Pesaran et al.)")
                    ardl_result = ardl_bounds_test(
                        climate_series,
                        covid_series,
                        climate_col,
                        covid_col,
                        dates=df_scen[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                    )
                    coint_results.append(ardl_result)
                    if ardl_result.get("error"):
                        print(f"  Error: {ardl_result['error']}")
                    else:
                        print(f"  F-statistic: {ardl_result['f_statistic']:.4f}")
                        print(f"  Bounds test: {ardl_result['bounds_result']}")
                        print(f"  Long-run relationship: {ardl_result['long_run_result']}")

                    # reverse bounds test
                    print("  Running reverse ARDL bounds test...")
                    ardl_result_rev = ardl_bounds_test(
                        covid_series,
                        climate_series,
                        covid_col,
                        climate_col,
                        dates=df_scen[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                    )
                    coint_results.append(ardl_result_rev)
                    if ardl_result_rev.get("error"):
                        print(f"  Reverse ARDL bounds test error: {ardl_result_rev['error']}")
                    else:
                        print(f"  Reverse F-statistic: {ardl_result_rev['f_statistic']:.4f}")
                        print(f"  Reverse bounds test: {ardl_result_rev['bounds_result']}")
                        print(f"  Reverse long-run relationship: {ardl_result_rev['long_run_result']}")

                    exog_vars = {}
                    if exogenous_columns:
                        for exog_col in exogenous_columns:
                            exog_vars[exog_col] = df_scen[exog_col]
                    # Add weekday dummies (Tue-Sun), with Monday as baseline
                    try:
                        wd_df = build_weekday_dummies(df_scen[date_col], index=df_scen.index, prefix="wd")
                        for c in wd_df.columns:
                            if c not in exog_vars:
                                exog_vars[c] = wd_df[c]
                    except Exception:
                        pass

                    # forward: covid -> climate (climate as dependent variable)
                    ecm_result = ardl_ecm_model(
                        climate_series,
                        covid_series,
                        climate_col,
                        covid_col,
                        dates=df_scen[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                        exog_vars=exog_vars,
                        exog_significant_only=args.exog_significant_only,
                        exog_threshold=args.exog_threshold,
                    )
                    coint_results.append(ecm_result)
                    if not ecm_result.get("error"):
                        _print_and_export_ecm(
                            "forward",
                            str(emotion),
                            climate_col,
                            covid_col,
                            ecm_result,
                            ardl_result,
                            output_dir_scen,
                            exog_output_dir_scen,
                            args,
                        )

                    # reverse: climate -> covid (covid as dependent variable)
                    ecm_result_rev = ardl_ecm_model(
                        covid_series,
                        climate_series,
                        covid_col,
                        climate_col,
                        dates=df_scen[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                        exog_vars=exog_vars,
                        exog_significant_only=args.exog_significant_only,
                        exog_threshold=args.exog_threshold,
                    )
                    coint_results.append(ecm_result_rev)
                    if not ecm_result_rev.get("error"):
                        _print_and_export_ecm(
                            "reverse",
                            str(emotion),
                            covid_col,
                            climate_col,
                            ecm_result_rev,
                            ardl_result_rev,
                            output_dir_scen,
                            exog_output_dir_scen,
                            args,
                        )

            # Save causality results
            if coint_results:
                coint_df = pd.DataFrame(coint_results)
                coint_path = os.path.splitext(output_file_scen)[0] + "_causality.csv"
                coint_df.to_csv(coint_path, index=False, encoding="utf-8-sig")
                print(f"\nCausality analysis results saved to: {coint_path}")

            # Export ARDL bounds test summary (8 emotions × 2 directions)
            print("\n=== Generating ARDL Bounds Test Summary ===")
            export_ardl_bounds_summary(output_dir_scen, coint_results)

            # Export ECM coefficients summary for 8 emotions
            print("\n=== Generating ECM Coefficients Summary ===")
            export_ecm_coefficients_summary(output_dir_scen, coint_results)

            # Generate forest plot for short-run and long-run coefficients
            print("\n=== Generating Short-Run and Long-Run Coefficients Forest Plot ===")
            plot_short_long_coefficients_forest(output_dir_scen, coint_results, confidence_level=0.95)

            # Generate time series plots if requested
            if args.timeseries_plots:
                print("\n=== Generating Time Series Plots ===")
                print(f"Will generate time series plots for {len(filtered_emotions)} emotion groups")

                for emotion in filtered_emotions:
                    try:
                        plot_emotion_timeseries(df_scen, date_col, emotion, output_dir_scen)
                    except Exception as e:
                        print(f"Failed to generate {emotion} time series plot: {str(e)}")

                print(f"\nTime series plot generation completed, saved in: {output_dir_scen}")

            print(f"\nScenario {scenario_label} completed.")

        # Build appendix placebo comparison table (real vs placebo_shuffle)
        _build_placebo_appendix_table(script_dir, output_dir)

        # Build Supplementary Table 6 (real vs placebo_shuffle, bounds + ECM comparison)
        try:
            print("\n=== Building Supplementary Table 6 (Placebo Comparison) ===")
            build_supplementary_table_6(output_dir)
        except Exception as e:
            print(f"Failed to build Supplementary Table 6: {e}")

        print("\nComplete analysis finished successfully!")
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")


if __name__ == "__main__":
    main()
