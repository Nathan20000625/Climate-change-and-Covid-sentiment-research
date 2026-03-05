import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
try:
    # FMOLS cointegration regression (Phillips-Hansen)
    from arch.unitroot.cointegration import FullyModifiedOLS
except ImportError:
    FullyModifiedOLS = None
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
            # 解析系列名称
            base = os.path.splitext(fn)[0]  # weekday_ols_xxx
            series_name = base[len("weekday_ols_"):]
            series_names.append(series_name)
            # 仅保留系数行（排除 r_squared / aic）
            if "term" in df.columns:
                df = df[(df["term"].astype(str) != "r_squared") & (df["term"].astype(str) != "aic")].copy()
            else:
                continue
            # 需要的列
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
            # 保留必要列
            series_to_df[series_name] = df[["term", "coefficient", "star"]].copy()

        if not series_to_df:
            return
        # 收集所有变量名（按出现顺序去重）
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
    对两个序列进行协整检验
    
    Args:
        series1, series2: 原始时间序列
        series1_name, series2_name: 序列名称
        dates: 日期序列，用于预处理
        weekday_ols: 是否去除工作日效应
        weekday_trend: 是否去除线性趋势
    
    Returns:
        Dict containing cointegration test results
    """
    # 清理数据
    s1 = pd.to_numeric(series1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s2 = pd.to_numeric(series2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # 如果需要进行预处理（去除季节性和趋势）
    if weekday_ols and dates is not None:
        # 对齐日期和序列
        s1_dates = dates.loc[s1.index]
        s2_dates = dates.loc[s2.index]
        
        # 去除工作日效应和趋势
        s1 = remove_weekday_effect(s1, s1_dates, add_trend=weekday_trend, polynomial_degree=weekday_poly_degree)
        s2 = remove_weekday_effect(s2, s2_dates, add_trend=weekday_trend, polynomial_degree=weekday_poly_degree)
        
        # 重新清理数据
        s1 = pd.to_numeric(s1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s2 = pd.to_numeric(s2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # 对齐索引
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
        # Engle-Granger协整检验
        coint_stat, coint_pvalue, _ = coint(s1_aligned, s2_aligned)
        result["coint_stat"] = coint_stat
        result["coint_pvalue"] = coint_pvalue
        result["coint_result"] = "cointegrated" if coint_pvalue < 0.05 else "not_cointegrated"
    except Exception as e:
        result["coint_result"] = f"error: {str(e)[:50]}"
    
    try:
        # Johansen协整检验
        data = np.column_stack([s1_aligned, s2_aligned])
        johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        result["johansen_trace_stat"] = johansen_result.lr1[0]  # 迹统计量
        result["johansen_trace_pvalue"] = johansen_result.cvt[0, 1]  # 迹检验p值
        result["johansen_max_eigen_stat"] = johansen_result.lr2[0]  # 最大特征值统计量
        result["johansen_max_eigen_pvalue"] = johansen_result.cve[0, 1] if hasattr(johansen_result, 'cve') else np.nan  # 最大特征值检验p值
        
        # 判断协整关系
        if johansen_result.lr1[0] > johansen_result.cvt[0, 1]:  # 迹检验
            result["johansen_result"] = "cointegrated"
        else:
            result["johansen_result"] = "not_cointegrated"
            
    except Exception as e:
        result["johansen_result"] = f"error: {str(e)[:50]}"
    
    return result


def select_optimal_lags(data: np.ndarray, max_lags: int = 6) -> int:
    """
    使用信息准则选择最优滞后期
    
    Args:
        data: 时间序列数据
        max_lags: 最大滞后期
    
    Returns:
        最优滞后期
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    
    try:
        model = VAR(data)
        lag_order = model.select_order(maxlags=max_lags)
        
        # 优先使用AIC，如果AIC不可用则使用BIC
        if hasattr(lag_order, 'aic') and not np.isnan(lag_order.aic):
            return lag_order.aic
        elif hasattr(lag_order, 'bic') and not np.isnan(lag_order.bic):
            return lag_order.bic
        else:
            return 1  # 默认值
    except:
        return 1  # 如果选择失败，使用默认值


# 已移除 VEC 模型估计函数（当前数据不包含 I(1)+I(1) 组合）


def ardl_ecm_model(series1: pd.Series, series2: pd.Series, series1_name: str, series2_name: str, max_lags: int = 6, dates: pd.Series = None, weekday_ols: bool = False, weekday_trend: bool = False, exog_vars: Dict[str, pd.Series] = None, exog_significant_only: bool = False, exog_threshold: float = 0.10) -> Dict[str, object]:
    """
    ARDL-ECM模型估计
    
    Args:
        series1, series2: 原始时间序列
        series1_name, series2_name: 序列名称
        max_lags: 最大滞后期
        dates: 日期序列，用于预处理
        weekday_ols: 是否去除工作日效应
        weekday_trend: 是否去除线性趋势
        exog_vars: 外生变量字典
        exog_significant_only: 是否只保留显著的外生变量
        exog_threshold: 外生变量显著性阈值
    
    Returns:
        Dict containing ARDL-ECM model results
    """
    # 清理数据
    s1 = pd.to_numeric(series1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s2 = pd.to_numeric(series2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # 如果需要进行预处理（去除季节性和趋势）
    if weekday_ols and dates is not None:
        # 对齐日期和序列
        s1_dates = dates.loc[s1.index]
        s2_dates = dates.loc[s2.index]
        
        # 去除工作日效应和趋势
        s1 = remove_weekday_effect(s1, s1_dates, add_trend=weekday_trend)
        s2 = remove_weekday_effect(s2, s2_dates, add_trend=weekday_trend)
        
        # 重新清理数据
        s1 = pd.to_numeric(s1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s2 = pd.to_numeric(s2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # 对齐索引
    common_index = s1.index.intersection(s2.index)
    if len(common_index) < 30:
        return {
            "series1": series1_name,
            "series2": series2_name,
            "error": "数据不足",
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
            result["error"] = "数据点不足"
            return result
        
        y = s1_aligned.values
        x = s2_aligned.values
        
        # 网格搜索 ARDL-ECM(p,q)（p=1..6，q=0..6），按 AIC 选最优；使用 HAC 稳健标准误
        # 改进：同时考虑残差序列相关性（LBQ检验），优先选择序列相关性较低的模型
        # 增加最大滞后项以更好地捕捉动态关系，减少残差序列相关性
        max_p, max_q = 6, 6
        best_aic = np.inf
        best_fit = None
        best_design = None
        best_meta = None
        dy_full = np.diff(y)
        dx_full = np.diff(x)
        # 预处理外生变量到矩阵
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
                    # Δx当前与滞后（若q=0则只含当前一阶差分）
                    q_use = max(1, q)
                    for j in range(0, q_use):
                        cols.append(dx_full[start - 1 - j: start - 1 - j + T])
                        col_names.append(f"dx_lag{j}")
                    # y_{t-1}, x_{t-1}
                    cols.append(y[start - 1: start - 1 + T]); col_names.append("y_lag")
                    cols.append(x[start - 1: start - 1 + T]); col_names.append("x_lag")
                    # 外生变量（与 t 对齐）
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

                    # 选阶：只看 AIC（越小越好）
                    if np.isfinite(aic_val) and (best_fit is None or aic_val < best_aic):
                        best_aic = aic_val
                        best_fit = fit_try
                        best_design = (dy, X_try)
                        best_meta = {"p": p, "q": q, "col_names": col_names}
                except Exception:
                    continue

        if best_fit is None:
            raise RuntimeError("ARDL-ECM 选阶失败")
        fitted_model = best_fit
        dy, X = best_design
        col_names = best_meta.get("col_names", [])
        
        # 提取结果
        params = fitted_model.params
        pvalues = fitted_model.pvalues
        # 记录选阶
        result["selected_p"] = best_meta.get("p", np.nan)
        result["selected_q"] = best_meta.get("q", np.nan)
        
        # ECM系数 (y_{t-1}的系数，应该为负)
        try:
            ylag_idx = col_names.index("y_lag")
        except Exception:
            ylag_idx = None
        if ylag_idx is not None and ylag_idx < len(params):
            result["ecm_coefficient"] = params[ylag_idx]
            result["ecm_pvalue"] = pvalues[ylag_idx] if ylag_idx < len(pvalues) else np.nan
        
        # 短期系数：提取所有dy和dx的滞后项
        # 从选阶结果获取实际的p和q值
        selected_p = best_meta.get("p", 1)
        selected_q = best_meta.get("q", 0)
        
        # 提取所有dy_lag{i}系数（从dy_lag1到dy_lag{p-1}，因为p是最大滞后阶数）
        for i in range(1, selected_p):
            lag_name = f"dy_lag{i}"
            try:
                lag_idx = col_names.index(lag_name)
                if lag_idx < len(params):
                    result["short_run_coefficients"][f"dy_lag{i}"] = params[lag_idx]
                    result["short_run_coefficients"][f"dy_lag{i}_pvalue"] = pvalues[lag_idx] if lag_idx < len(pvalues) else np.nan
            except ValueError:
                # 该滞后项不存在，跳过
                pass
        
        # 提取所有dx_lag{j}系数（从dx_lag0到dx_lag{q-1}，q_use = max(1, q)）
        q_use = max(1, selected_q)
        for j in range(0, q_use):
            lag_name = f"dx_lag{j}"
            try:
                lag_idx = col_names.index(lag_name)
                if lag_idx < len(params):
                    if j == 0:
                        # 保持向后兼容，dx_lag0也保存为delta_x
                        result["short_run_coefficients"]["delta_x"] = params[lag_idx]
                        result["short_run_coefficients"]["delta_x_pvalue"] = pvalues[lag_idx] if lag_idx < len(pvalues) else np.nan
                    result["short_run_coefficients"][f"dx_lag{j}"] = params[lag_idx]
                    result["short_run_coefficients"][f"dx_lag{j}_pvalue"] = pvalues[lag_idx] if lag_idx < len(pvalues) else np.nan
            except ValueError:
                # 该滞后项不存在，跳过
                pass
        
        # 如果没有找到任何dx_lag项，设置默认值（向后兼容）
        if "delta_x" not in result["short_run_coefficients"]:
            result["short_run_coefficients"]["delta_x"] = np.nan
            result["short_run_coefficients"]["delta_x_pvalue"] = np.nan
        
        # 长期系数 (通过ECM项计算，使用Delta方法计算标准误)
        try:
            xlag_idx = col_names.index("x_lag")
        except Exception:
            xlag_idx = None
        if ylag_idx is not None and xlag_idx is not None and ylag_idx < len(params) and xlag_idx < len(params):
            if abs(params[ylag_idx]) > 1e-10:
                # 长期系数 = -params[x_lag] / params[y_lag]
                long_run_coef = -params[xlag_idx] / params[ylag_idx]
                result["long_run_coefficient"] = long_run_coef
                
                # 使用Delta方法计算长期系数的标准误
                try:
                    # 获取协方差矩阵
                    cov_matrix = fitted_model.cov_params()
                    
                    # 提取相关参数
                    a = params[ylag_idx]  # y_lag系数
                    d = params[xlag_idx]  # x_lag系数
                    
                    # 提取方差和协方差
                    var_a = cov_matrix.iloc[ylag_idx, ylag_idx] if hasattr(cov_matrix, 'iloc') else cov_matrix[ylag_idx, ylag_idx]
                    var_d = cov_matrix.iloc[xlag_idx, xlag_idx] if hasattr(cov_matrix, 'iloc') else cov_matrix[xlag_idx, xlag_idx]
                    
                    # 提取协方差（注意：协方差矩阵是对称的）
                    if hasattr(cov_matrix, 'iloc'):
                        cov_ad = cov_matrix.iloc[ylag_idx, xlag_idx]
                    else:
                        cov_ad = cov_matrix[ylag_idx, xlag_idx] if ylag_idx < cov_matrix.shape[0] and xlag_idx < cov_matrix.shape[1] else 0
                    
                    # Delta方法：长期系数的方差
                    # long_run = -d/a
                    # var(long_run) = (1/a²) * var(d) + (d²/a⁴) * var(a) - 2*(d/a³) * cov(a,d)
                    var_long = (1 / (a**2)) * var_d + (d**2 / (a**4)) * var_a - 2 * (d / (a**3)) * cov_ad
                    
                    # 确保方差非负
                    var_long = max(0, var_long)
                    se_long = np.sqrt(var_long)
                    
                    # 计算t统计量和p值
                    # 使用t分布（自由度 = n - k，其中k是参数个数）
                    n_obs = len(dy)
                    k_params = len(params)
                    df = n_obs - k_params
                    
                    if se_long > 1e-10 and df > 0:
                        t_stat = long_run_coef / se_long
                        # 双侧检验
                        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
                        result["long_run_std_error"] = se_long
                        result["long_run_tstat"] = t_stat
                        result["long_run_pvalue"] = p_value
                        
                        # 计算95%置信区间
                        t_critical = t.ppf(0.975, df)
                        result["long_run_ci_lower"] = long_run_coef - t_critical * se_long
                        result["long_run_ci_upper"] = long_run_coef + t_critical * se_long
                    else:
                        # 如果标准误太小或自由度无效，使用原始方法作为后备
                        result["long_run_std_error"] = np.nan
                        result["long_run_tstat"] = np.nan
                        result["long_run_pvalue"] = pvalues[xlag_idx] if xlag_idx < len(pvalues) else np.nan
                        result["long_run_ci_lower"] = np.nan
                        result["long_run_ci_upper"] = np.nan
                        
                except Exception as e:
                    # 如果Delta方法失败，使用原始方法作为后备
                    result["long_run_std_error"] = np.nan
                    result["long_run_tstat"] = np.nan
                    result["long_run_pvalue"] = pvalues[xlag_idx] if xlag_idx < len(pvalues) else np.nan
                    result["long_run_ci_lower"] = np.nan
                    result["long_run_ci_upper"] = np.nan
        
        # 提取外生变量系数与 p 值（若存在外生变量，则从拟合结果中抓取对应列）
        try:
            exog_coeffs_map = {}
            # 外生变量本身
            for name in exog_names_order:
                if name in col_names:
                    idx = col_names.index(name)
                    if idx < len(params):
                        exog_coeffs_map[str(name)] = {
                            "coefficient": float(params[idx]),
                            "pvalue": float(pvalues[idx]) if idx < len(pvalues) else float("nan"),
                        }
            # 常数项（与外生变量一起作为设计列使用）
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

        # 附加：保存拟合对象的必要信息以便 IRF/动态乘数与CI
        result["_fitted_params"] = {
            "params": params,
            "pvalues": fitted_model.pvalues,
            # 协方差矩阵和标准误都保存下来，方便后续稳健获取 std_error
            "cov": fitted_model.cov_params(),
            "bse": getattr(fitted_model, "bse", None),
            "design_cols": list(col_names),
        }
        # 打印选阶与AIC（控制台最小输出）
        try:
            p_val = int(result.get('selected_p'))
            q_val = int(result.get('selected_q'))
            aic_val = float(result.get('aic'))
            print(f"  [ECM 选阶] p={p_val}, q={q_val}, AIC={aic_val:.3f}")
        except Exception:
            pass
        
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


def check_causality_exists(ardl_result: Dict[str, object], ecm_result: Dict[str, object], alpha: float = 0.10) -> bool:
    """
    检查因果关系是否存在
    
    判断标准：
    1. ARDL边界检验通过（bounds_result == 'cointegrated'）- 表明存在长期协整关系
    2. ECM模型成功运行（无error）
    3. ECM系数显著（ecm_pvalue < alpha）- 核心判断标准
    
    说明：
    - ECM系数显著表明存在误差修正机制，即存在因果关系（至少是短期因果关系）
    - 长期关系显著不是必要条件：
      * ECM系数显著但长期关系不显著：可能存在短期因果关系，长期关系较弱
      * ECM系数显著且长期关系显著：存在完整的因果关系（长期+短期）
    
    Args:
        ardl_result: ARDL边界检验结果
        ecm_result: ECM模型估计结果
        alpha: 显著性水平，默认0.05
    
    Returns:
        bool: True表示存在因果关系，False表示不存在
    """
    # 检查边界检验是否通过
    if ardl_result.get('error') or ardl_result.get('bounds_result') != 'cointegrated':
        return False
    
    # 检查ECM模型是否成功运行
    if ecm_result.get('error'):
        return False
    
    # 核心判断：ECM系数是否显著
    # ECM系数显著表明存在误差修正机制，即存在因果关系（至少是短期）
    ecm_pvalue = ecm_result.get('ecm_pvalue', 1.0)
    if pd.isna(ecm_pvalue) or float(ecm_pvalue) >= alpha:
        return False
    
    # ECM系数显著即认为存在因果关系
    # 长期关系显著不是必要条件，但可以作为额外信息
    return True


def format_causality_detail(ardl_result: Dict[str, object], ecm_result: Dict[str, object], alpha: float = 0.10) -> str:
    """
    基于ARDL/ECM结果输出：短期因果/长期因果是否显著，以及影响方向（正/负）。

    规则：
    - 短期因果：以 Δx 的当期系数（delta_x）p 值判断（< alpha 视为存在），符号决定方向
    - 长期因果：以长期系数 p 值判断（< alpha 视为存在），且需边界检验为 cointegrated
    - 方向：系数 > 0 为"正向"，< 0 为"负向"，=0 记为"零"
    - 注意：ECM项显著（用于判断"存在因果"）仅表明存在误差修正机制，不一定意味着短期或长期直接因果关系都显著
    """
    try:
        # 检查ECM项是否显著（用于判断"存在因果"的核心标准）
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
        
        # 短期
        sr_coef = None
        sr_p = None
        try:
            sr_coef = ecm_result.get('short_run_coefficients', {}).get('delta_x', np.nan)
            sr_p = ecm_result.get('short_run_coefficients', {}).get('delta_x_pvalue', np.nan)
        except Exception:
            sr_coef, sr_p = np.nan, np.nan
        sr_exist = (not pd.isna(sr_p)) and float(sr_p) < alpha
        if pd.isna(sr_coef):
            sr_dir = "未知"
            sr_info = "未知"
        else:
            sr_dir = "正向" if float(sr_coef) > 0 else ("负向" if float(sr_coef) < 0 else "零")
            if pd.isna(sr_p):
                sr_info = f"{sr_dir}（p值缺失）"
            else:
                sr_info = f"{sr_dir}（p={float(sr_p):.4f}{pstars(sr_p)}）"

        # 长期（需cointegrated）
        lr_coef = ecm_result.get('long_run_coefficient', np.nan)
        lr_p = ecm_result.get('long_run_pvalue', np.nan)
        lr_exist = (
            ardl_result.get('bounds_result') == 'cointegrated'
            and (not pd.isna(lr_p))
            and float(lr_p) < alpha
        )
        if pd.isna(lr_coef):
            lr_dir = "未知"
            lr_info = "未知"
        else:
            lr_dir = "正向" if float(lr_coef) > 0 else ("负向" if float(lr_coef) < 0 else "零")
            if pd.isna(lr_p):
                lr_info = f"{lr_dir}（p值缺失）"
            else:
                lr_info = f"{lr_dir}（p={float(lr_p):.4f}{pstars(lr_p)}）"

        # ECM项方向（作为说明）
        ecm_note = ""
        if not pd.isna(ecm_coef):
            ecm_dir = "负向" if float(ecm_coef) < 0 else ("正向" if float(ecm_coef) > 0 else "零")
            if ecm_significant:
                ecm_note = f"；ECM项显著（{ecm_dir}，p={float(ecm_p):.4f}{pstars(ecm_p)}）→ 存在误差修正机制"
            else:
                ecm_note = f"；ECM项不显著（{ecm_dir}，p={float(ecm_p):.4f}{pstars(ecm_p)}）"

        return (
            f"短期因果：{'存在' if sr_exist else '不存在'}（{sr_info}）；"
            f"长期因果：{'存在' if lr_exist else '不存在'}（{lr_info}）" + ecm_note
        )
    except Exception:
        return "短期因果：未知；长期因果：未知"


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
    统一的 ARDL-ECM 结果输出/导出（系数、外生、IRF、因果详情）。
    direction: "forward" (covid->climate, dep=climate) 或 "reverse" (climate->covid, dep=covid)
    """
    try:
        # 打印核心系数
        if direction == "forward":
            print(f"  ARDL-ECM模型结果:")
        else:
            print(f"  Reverse ARDL-ECM模型结果:")
        if ecm_result.get('error'):
            if direction == "forward":
                print(f"    错误: {ecm_result['error']}")
            else:
                print(f"    Reverse ECM Error: {ecm_result['error']}")
        else:
            if direction == "forward":
                print(f"    ECM系数: {ecm_result['ecm_coefficient']:.4f} (p={ecm_result['ecm_pvalue']:.4f})")
                print(f"    短期系数: {ecm_result['short_run_coefficients'].get('delta_x', 'N/A'):.4f}")
                # 显示长期系数的详细信息（包括标准误、置信区间）
                lr_coef = ecm_result.get('long_run_coefficient', np.nan)
                lr_se = ecm_result.get('long_run_std_error', np.nan)
                lr_pval = ecm_result.get('long_run_pvalue', np.nan)
                lr_ci_lower = ecm_result.get('long_run_ci_lower', np.nan)
                lr_ci_upper = ecm_result.get('long_run_ci_upper', np.nan)
                if not pd.isna(lr_coef):
                    if not pd.isna(lr_se) and not pd.isna(lr_pval):
                        print(f"    长期系数: {lr_coef:.4f} (SE={lr_se:.4f}, p={lr_pval:.4f}, 95% CI=[{lr_ci_lower:.4f}, {lr_ci_upper:.4f}])")
                    else:
                        print(f"    长期系数: {lr_coef:.4f}")
                else:
                    print(f"    长期系数: N/A")
            else:
                print(f"    Reverse ECM coefficient: {ecm_result['ecm_coefficient']:.4f} (p={ecm_result['ecm_pvalue']:.4f})")
                print(f"    Reverse Short-run coefficient: {ecm_result['short_run_coefficients'].get('delta_x', 'N/A'):.4f}")
                # 显示长期系数的详细信息（包括标准误、置信区间）
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
            # 外生变量导出
            pair_key = f"{dep_col}_<-_{indep_col}"
            export_exog_coefficients(exog_out_dir, pair_key, ecm_result.get('exog_coefficients', {}))
            mode = "ECM_forward" if direction == "forward" else "ECM_reverse"
            append_exog_summary(exog_out_dir, emotion=emotion, mode=mode, pair_key=pair_key, exog_coeffs=ecm_result.get('exog_coefficients', {}))
            export_all_ardl_ecm_coefficients(exog_out_dir, emotion, direction, pair_key, ecm_result)
            # 刷新方向宽表
            build_direction_wide_coefficients(exog_out_dir, direction="forward" if direction == "forward" else "reverse")
            # IRF 导出
            try:
                coef = ecm_result.get('ecm_coefficient', np.nan)
                print(f"  Debug: args.irf = {args.irf}, ecm_coefficient = {coef}")
                if args.irf and not np.isnan(coef):
                    is_forward = direction == "forward"
                    # 水平响应（只计算95%置信区间）
                    irf_obj = compute_ecm_dynamic_multiplier_with_ci(ecm_result, horizon=args.irf_horizon, n_boot=args.irf_boot, return_delta=False, multi_ci=[0.95])
                    # 构建包含95%置信区间的DataFrame
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
                    # 差分响应（只计算95%置信区间）
                    irf_obj_delta = compute_ecm_dynamic_multiplier_with_ci(ecm_result, horizon=args.irf_horizon, n_boot=args.irf_boot, return_delta=True, multi_ci=[0.95])
                    # 构建包含95%置信区间的DataFrame
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
            # 此处不再根据模型自动给出“存在/不存在因果”的结论，仅保留系数与IRF等数值输出
    except Exception as e:
        print(f"ECM reporting/export error: {str(e)}")

def ardl_bounds_test(series1: pd.Series, series2: pd.Series, series1_name: str, series2_name: str, max_lags: int = 4, dates: pd.Series = None, weekday_ols: bool = False, weekday_trend: bool = False) -> Dict[str, object]:
    """
    ARDL边界检验 (Pesaran et al., 2001)
    适用于I(0)/I(1)混合阶组合（包括 I(1)+I(1)），但不应包含 I(2) 或更高阶
    
    Args:
        series1, series2: 原始时间序列
        series1_name, series2_name: 序列名称
        max_lags: 最大滞后期
        dates: 日期序列，用于预处理
        weekday_ols: 是否去除工作日效应
        weekday_trend: 是否去除线性趋势
    
    Returns:
        Dict containing ARDL bounds test results
    """
    # 清理数据
    s1 = pd.to_numeric(series1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s2 = pd.to_numeric(series2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # 如果需要进行预处理（去除季节性和趋势）
    if weekday_ols and dates is not None:
        # 对齐日期和序列
        s1_dates = dates.loc[s1.index]
        s2_dates = dates.loc[s2.index]
        
        # 去除工作日效应和趋势
        s1 = remove_weekday_effect(s1, s1_dates, add_trend=weekday_trend)
        s2 = remove_weekday_effect(s2, s2_dates, add_trend=weekday_trend)
        
        # 重新清理数据
        s1 = pd.to_numeric(s1, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        s2 = pd.to_numeric(s2, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
    # 对齐索引
    common_index = s1.index.intersection(s2.index)
    if len(common_index) < 30:  # ARDL需要更多数据
        return {
            "series1": series1_name,
            "series2": series2_name,
            "f_statistic": np.nan,
            "bounds_result": "insufficient_data",
            "long_run_result": "insufficient_data",
            "error": "数据不足"
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
        # 简化的ARDL边界检验
        # 使用简单的线性回归检验长期关系
        
        n = len(s1_aligned)
        if n < 20:
            result["error"] = "数据点不足"
            return result
        
        y = s1_aligned.values
        x = s2_aligned.values
        
        # 构建简单的ARDL模型: y_t = α + β₁y_{t-1} + β₂x_{t-1} + ε_t
        y_lag = y[:-1]  # y_{t-1}
        x_lag = x[:-1]  # x_{t-1}
        y_current = y[1:]  # y_t
        
        # 添加常数项
        X = np.column_stack([np.ones(len(y_lag)), y_lag, x_lag])
        
        # 拟合模型
        model = OLS(y_current, X)
        fitted_model = model.fit()
        
        # 计算F统计量 (检验β₁ = β₂ = 0)
        # F = (R^2/(k-1)) / ((1-R^2)/(n-k))
        r_squared = fitted_model.rsquared
        n_obs = len(y_current)
        k = 3  # 常数项 + 2个解释变量
        
        if r_squared < 0.99:  # 避免除零
            f_stat = (r_squared / (k-1)) / ((1-r_squared) / (n_obs - k))
        else:
            f_stat = 1000  # 高R^2情况
        
        result["f_statistic"] = f_stat
        result["n_obs"] = n_obs
        result["df1"] = k - 1
        result["df2"] = n_obs - k
        try:
            # 使用 F 分布近似计算 p 值，用于显著性星号
            result["p_value"] = float(1.0 - f.cdf(f_stat, dfn=result["df1"], dfd=result["df2"]))
        except Exception:
            result["p_value"] = np.nan
        
        # 边界检验临界值（这里使用 Pesaran 等(2001) 提供的 Case III / k≈1 的近似临界值）
        # 1%: I(0)=6.84, I(1)=7.84；5%: I(0)=3.17, I(1)=4.61
        # 与下方判断保持一致：F < I(0) 判为 not_cointegrated，F > I(1) 判为 cointegrated
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
        
        # 长期关系检验
        if len(fitted_model.params) >= 2:
            beta1_pvalue = fitted_model.pvalues[1] if len(fitted_model.pvalues) > 1 else 1.0
            result["long_run_result"] = "significant" if beta1_pvalue < 0.05 else "not_significant"
        
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


# === ARDL-ECM 动态乘数 / IRF 与系数导出工具 ===
def _simulate_ecm_irf(params: np.ndarray, horizon: int = 30) -> np.ndarray:
    """
    给定简化 ECM 参数（与 ardl_ecm_model 一致的次序: const, b(Δx), a(y_{t-1}), d(x_{t-1})），
    生成对单位"永久冲击"（h=0 时 Δx=1，此后 Δx=0，x 水平保持+1）的 y 水平响应路径。

    说明：
    - 为隔离冲击，常数项在仿真中设为 0（不影响相对响应路径）。
    - 初始稳态为 y_{-1}=0, x_{-1}=0（冲击前的状态）。
    - h=0 是冲击发生的当期：y_0 = y_{-1} + Δy_0 = 0 + b*1 = b（短期系数）。
    - 仅使用该简化ECM（无Δy与Δx的更高阶滞后），与当前实现一致。
    - 返回数组长度为 horizon+1，包括 h=0 到 h=horizon。
    """
    if params is None or len(params) < 4:
        return np.full(horizon + 1, np.nan)
    const = 0.0  # 隔离冲击
    b = float(params[1])  # Δx 当期系数
    a = float(params[2])  # y_{t-1} 系数（应为负）
    d = float(params[3])  # x_{t-1} 系数

    y_prev = 0.0  # y_{-1} = 0 (冲击前的状态)
    x_prev = 0.0  # x_{-1} = 0 (冲击前的状态)
    x_level = 0.0
    path_y = []  # 从空列表开始，h=0将是冲击发生的当期
    for t in range(horizon + 1):  # 包括h=0，所以是horizon+1个点
        if t == 0:
            delta_x = 1.0  # h=0时发生冲击
        else:
            delta_x = 0.0  # h>0时，x水平保持不变（永久冲击）
        # Δy_t = const + b*Δx_t + a*y_{t-1} + d*x_{t-1}
        delta_y = const + b * delta_x + a * y_prev + d * x_prev
        y_level = y_prev + delta_y
        # h=0时：y_0 = y_{-1} + Δy_0 = 0 + b*1 = b (短期系数)
        # 更新 x 水平
        x_level = x_prev + delta_x
        # 推进状态
        y_prev = y_level
        x_prev = x_level
        path_y.append(y_level)
    return np.array(path_y)


def compute_ecm_dynamic_multiplier_with_ci(ecm_result: Dict[str, object], horizon: int = 30, n_boot: int = 500, ci: float = 0.95, return_delta: bool = False, multi_ci: List[float] = None) -> Dict[str, object]:
    """
    基于 ECM 拟合的参数与协方差做参数自助法（正态近似）生成 IRF/动态乘数及置信区间。
    
    参数:
        return_delta: 如果为True，返回差分响应（Δy）而非水平响应（Level）。
                     差分响应展示当期影响，不会累积放大，数值更小更直观。
        multi_ci: 如果提供，计算多个置信区间（例如 [0.95]）。此时会返回多个置信区间的上下限。
                 返回字典中会包含 "lower_95", "upper_95" 等键。
                 如果为None，则只计算ci指定的置信区间（保持向后兼容）。
    
    返回 {"h": array, "irf": array, "lower": array, "upper": array, "long_run": float, ...}
    注意：当return_delta=True时，irf表示差分响应而非累积水平响应
    注意：当multi_ci不为None时，返回字典还会包含 "lower_{int(ci*100)}", "upper_{int(ci*100)}" 等键
    """
    fp = ecm_result.get("_fitted_params", {})
    params = fp.get("params", None)
    cov = fp.get("cov", None)
    cols = fp.get("design_cols", [])

    # 将完整参数向量按列名映射到简化ECM所需的4个核心参数：
    # [const, b(Δx 当期 dx_lag0), a(y_{t-1}), d(x_{t-1})]
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
        b = _get("dx_lag0", 0.0)  # 当期 Δx 的系数
        # a 与 d 必须存在；缺失则返回 NaN 以触发下游的 NaN 路径
        a = _get("y_lag", np.nan, allow_nan=True)
        d = _get("x_lag", np.nan, allow_nan=True)
        return np.array([const, b, a, d], dtype=float)

    if params is None or cov is None:
        # 即使参数缺失，也返回多置信区间的结构（如果请求了multi_ci）
        result = {"h": np.arange(horizon + 1), "irf": np.full(horizon + 1, np.nan), "long_run": np.nan}
        if multi_ci is not None and len(multi_ci) > 0:
            for ci_level in multi_ci:
                ci_label = int(ci_level * 100)
                result[f"lower_{ci_label}"] = np.full(horizon + 1, np.nan)
                result[f"upper_{ci_label}"] = np.full(horizon + 1, np.nan)
            # 也设置默认的lower和upper
            result["lower"] = np.full(horizon + 1, np.nan)
            result["upper"] = np.full(horizon + 1, np.nan)
        else:
            result["lower"] = np.full(horizon + 1, np.nan)
            result["upper"] = np.full(horizon + 1, np.nan)
        return result

    # 点估计路径（水平响应）：基于映射后的核心参数
    core_params = _extract_core_params(params)
    if np.isnan(core_params).any():
        irf_level = np.full(horizon + 1, np.nan)
    else:
        irf_level = _simulate_ecm_irf(core_params, horizon=horizon)
    
    # 计算差分响应（如果请求）
    if return_delta:
        # 差分响应 = 当期水平 - 前一期水平
        # 现在水平响应在h=0时已经是b（冲击发生的当期），y_{-1}=0
        # 所以：h=0时，差分响应 = y_0 - y_{-1} = b - 0 = b（短期系数）
        irf_point = np.diff(irf_level, prepend=0.0)  # prepend=0表示y_{-1}=0
        # 由于水平响应在h=0时已经是b，所以差分响应在h=0时自动等于b，无需额外修正
    else:
        irf_point = irf_level
    
    # 长期效应（lambda）- 基于水平响应计算
    try:
        a = float(core_params[2])
        d = float(core_params[3])
        long_run = -d / a if abs(a) > 1e-12 else np.nan
    except Exception:
        long_run = np.nan

    # 参数自助法
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
    
    # 计算差分响应的置信区间
    if return_delta:
        # 由于水平响应在h=0时已经是b，所以差分响应会自动正确
        irf_draws = np.array([np.diff(path, prepend=0.0) for path in irf_draws_level])
    else:
        irf_draws = irf_draws_level
    
    # 构建返回字典
    result = {"h": np.arange(horizon + 1), "irf": irf_point, "long_run": long_run}
    
    # 如果指定了多个置信区间，计算所有置信区间
    if multi_ci is not None and len(multi_ci) > 0:
        # 计算所有指定的置信区间
        for ci_level in multi_ci:
            lower_q = (1 - ci_level) / 2
            upper_q = 1 - lower_q
            ci_label = int(ci_level * 100)
            result[f"lower_{ci_label}"] = np.nanpercentile(irf_draws, lower_q * 100, axis=0)
            result[f"upper_{ci_label}"] = np.nanpercentile(irf_draws, upper_q * 100, axis=0)
        # 为了向后兼容，也设置默认的lower和upper（使用最大的置信区间）
        max_ci = max(multi_ci)
        lower_q = (1 - max_ci) / 2
        upper_q = 1 - lower_q
        result["lower"] = np.nanpercentile(irf_draws, lower_q * 100, axis=0)
        result["upper"] = np.nanpercentile(irf_draws, upper_q * 100, axis=0)
    else:
        # 只计算单个置信区间（保持向后兼容）
        lower_q = (1 - ci) / 2
        upper_q = 1 - lower_q
        result["lower"] = np.nanpercentile(irf_draws, lower_q * 100, axis=0)
        result["upper"] = np.nanpercentile(irf_draws, upper_q * 100, axis=0)
    
    return result


def save_irf_plot(csv_dir: str, pair_key: str, direction: str, irf: Dict[str, np.ndarray], response_type: str = "level"):
    """
    保存IRF图
    
    参数:
        response_type: "level" 表示水平响应（累积），"delta" 表示差分响应（当期影响）
    """
    try:
        h = irf["h"]
        y = irf["irf"]
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(h, y, label="IRF", color="#1f77b4", linewidth=2)
        
        # 检查是否有多个置信区间（如lower_90, upper_90, lower_95, upper_95）
        # 首先检查是否有带数字的置信区间键（如lower_90, upper_90等）
        ci_keys = []
        for key in irf.keys():
            if key.startswith("lower_") and len(key) > 6:
                suffix = key[6:]  # 获取"lower_"之后的部分
                if suffix.isdigit():
                    ci_keys.append(key)
            elif key.startswith("upper_") and len(key) > 6:
                suffix = key[6:]  # 获取"upper_"之后的部分
                if suffix.isdigit():
                    ci_keys.append(key)
        
        ci_levels = set()
        for key in ci_keys:
            if key.startswith("lower_"):
                ci_levels.add(key.replace("lower_", ""))
            elif key.startswith("upper_"):
                ci_levels.add(key.replace("upper_", ""))
        
        # 如果找到带数字的置信区间键，使用多置信区间模式
        # 验证每个置信水平都有对应的lower和upper键，且数据有效
        valid_ci_levels = []
        for ci_level_str in ci_levels:
            try:
                ci_level = int(ci_level_str)
                lower_key = f"lower_{ci_level}"
                upper_key = f"upper_{ci_level}"
                if lower_key in irf and upper_key in irf:
                    lo = irf[lower_key]
                    hi = irf[upper_key]
                    # 检查是否至少有一些有效数据（不全为NaN）
                    if lo is not None and hi is not None and np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                        valid_ci_levels.append(ci_level)
            except (ValueError, TypeError):
                continue
        
        # 只绘制95%置信区间，过滤掉90%的
        if valid_ci_levels:
            # 只保留95%置信区间
            filtered_ci_levels = [ci for ci in valid_ci_levels if ci == 95]
            if filtered_ci_levels:
                # 定义95%置信区间的颜色和透明度
                ci_colors = {
                    95: ("#1f77b4", 0.25),  # 95%置信区间：较深的蓝色，较高透明度
                }
                # 默认颜色（如果置信水平不在预设中）
                default_color = ("#1f77b4", 0.2)
                
                for ci_level in filtered_ci_levels:
                    lower_key = f"lower_{ci_level}"
                    upper_key = f"upper_{ci_level}"
                    lo = irf[lower_key]
                    hi = irf[upper_key]
                    # 使用有效数据绘制（处理NaN值）
                    valid_mask = np.isfinite(lo) & np.isfinite(hi)
                    if np.any(valid_mask):
                        color, alpha = ci_colors.get(ci_level, default_color)
                        plt.fill_between(h[valid_mask], lo[valid_mask], hi[valid_mask], color=color, alpha=alpha, label=f"{ci_level}% CI")
        else:
            # 向后兼容：只绘制默认的置信区间
            lo = irf.get("lower")
            hi = irf.get("upper")
            if lo is not None and hi is not None:
                # 检查是否至少有一些有效数据
                valid_mask = np.isfinite(lo) & np.isfinite(hi)
                if np.any(valid_mask):
                    plt.fill_between(h[valid_mask], lo[valid_mask], hi[valid_mask], color="#1f77b4", alpha=0.2, label="CI")
        
        plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        plt.xlabel("Horizon")
        plt.legend(loc="best", fontsize=9)
        
        # 简化标题：提取情绪名称和方向
        emotion = ""
        if "climate_" in pair_key and "covid_" in pair_key:
            # 提取情绪名称
            emotion = pair_key.split("climate_")[1].split("_score_freq")[0]
        elif "VAR_" in pair_key:
            # VAR模型的情况，尝试从pair_key中提取
            parts = pair_key.split("_")
            for i, part in enumerate(parts):
                if part in ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]:
                    emotion = part
                    break
        
        # 简化方向显示
        if "covid" in direction and "climate" in direction:
            if "covid" in direction.split("->")[0]:
                dir_text = "COVID → Climate"
            else:
                dir_text = "Climate → COVID"
        else:
            dir_text = direction.replace("->", " → ")
        
        # 设置标题和Y轴标签
        emotion_title = emotion.title() if emotion else "VAR"
        # 在标题中加入 λ（长期效应）便于肉眼比对
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
        
        # 简化文件名
        safe_emotion = emotion.replace(':', '_').replace('/', '_').replace('<', '_').replace('>', '_') if emotion else "var"
        if "covid" in direction and "climate" in direction:
            if "covid" in direction.split("->")[0]:
                file_suffix = "covid_to_climate"
            else:
                file_suffix = "climate_to_covid"
        else:
            file_suffix = direction.replace('->', 'to').replace(':', '_').replace('/', '_').replace('$', '')
        
        # 根据响应类型添加后缀，避免文件名冲突
        response_suffix = "_delta" if response_type == "delta" else "_level"
        out_path = os.path.join(csv_dir, f"irf_{safe_emotion}_{file_suffix}{response_suffix}.png")
        plt.tight_layout()
        # 确保覆盖旧文件（某些查看器会缓存时间戳）
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
    """导出简化ARDL-ECM各项系数（所有dy和dx滞后项、y_{t-1}、x_{t-1}及外生）。"""
    try:
        fp = ecm_result.get("_fitted_params", {})
        params = fp.get("params", None)
        pvalues = fp.get("pvalues", None)
        cols = fp.get("design_cols", [])
        
        # 获取short_run_coefficients中的p值映射（用于向后兼容）
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
                # 优先使用pvalues数组中的值，如果没有则从pvals_map获取
                if pvalues is not None and i < len(pvalues) and not np.isnan(pvalues[i]):
                    pval_val = float(pvalues[i])
                else:
                    # 检查是否是dy_lag或dx_lag项，从short_run_coefficients获取
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
            # 添加显著性星标列
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
            # 显示格式：系数科学计数法两位有效数字；p值两位小数
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
    """将 result["exog_coefficients"] 导出为 CSV（列：variable, coefficient, pvalue）。"""
    # 安全文件名转换
    def _safe_name(name: str) -> str:
        try:
            s = str(name)
        except Exception:
            s = "name"
        # 替换Windows非法字符: <>:"/\|?*
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
    df_out = pd.DataFrame(rows, columns=["variable", "coefficient", "pvalue"])  # 即使为空也导出
    # 添加显著性星标列
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
    # 显示格式：系数科学计数法两位有效数字；p值两位小数
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
    导出ARDL-ECM模型的所有系数到一个CSV文件
    
    Args:
        csv_dir: 输出目录
        emotion: 情绪类型
        direction: 方向（'forward'或'reverse'）
        pair_key: 配对键
        ecm_result: ARDL-ECM模型结果字典
    """
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception as e:
        print(f"  Failed to create dir {csv_dir}: {e}")
        return
    
    rows = []
    
    # 检查是否有错误
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
        # 提取基本信息
        series1 = ecm_result.get('series1', '')
        series2 = ecm_result.get('series2', '')
        
        # 提取拟合参数
        fp = ecm_result.get("_fitted_params", {})
        params = fp.get("params", None)
        pvalues = fp.get("pvalues", None)
        design_cols = list(fp.get("design_cols", []))
        cov = fp.get("cov", None)
        bse = fp.get("bse", None)

        # 输出所有设计列对应的系数（同时记录标准误，便于后续宽表中展示 ECM 系数的 SE）
        if params is not None and design_cols:
            for i, name in enumerate(design_cols):
                coef_val = float(params[i]) if i < len(params) and not np.isnan(params[i]) else np.nan
                pval_val = float(pvalues[i]) if (pvalues is not None and i < len(pvalues) and not np.isnan(pvalues[i])) else np.nan
                # 从协方差矩阵中提取该系数的标准误；如失败则回退到 bse 数组
                se_val = np.nan
                if cov is not None and isinstance(cov, (np.ndarray, list)) and np.shape(cov)[0] > i and np.shape(cov)[1] > i:
                    try:
                        se_val = float(np.sqrt(cov[i][i]))
                    except Exception:
                        se_val = np.nan
                # 若协方差矩阵不可用或给出 NaN，则尝试使用拟合结果中的 bse
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

        # 长期系数 (通过ECM计算的，使用Delta方法)
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

        # 外生变量列已包含在 design_cols 中，这里不重复追加
        
        # 模型统计量
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
    
    # 导出到CSV
    if rows:
        df_out = pd.DataFrame(rows)
        # 添加显著性星标列
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
        # 显示格式：系数科学计数法两位有效数字；p值两位小数
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
        
        # 如果文件已存在，追加数据；否则创建新文件
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
    基于 ardl_ecm_all_coefficients.csv 生成方向聚合的宽表：
    - 列：各情绪
    - 行：变量名称对应的系数、p 值和标准误；额外添加 ECM 系数、其 p 值和标准误三行
    - 输出：coefficients_forward_wide.csv 或 coefficients_reverse_wide.csv
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
        # 只保留必要列
        for col in ["coefficient_name", "coefficient_value", "pvalue", "std_error", "star"]:
            if col not in df_dir.columns:
                df_dir[col] = np.nan
        emotions = sorted(df_dir['emotion'].dropna().astype(str).unique().tolist())
        variables = df_dir['coefficient_name'].dropna().astype(str).unique().tolist()

        # 定义外生变量名称集合（在宽表中会以 exog_ 前缀展示，但系数完全来自 ECM 模型本身）
        exog_base_vars = set([
            "US_daily_covid_death", "debates", "climatenews", "GovernmentResponseIndex_Average",
            "WinterStorm", "Wildfire", "TropicalCyclone", "SevereStorm", "Flood", "Drought",
            "const"
        ] + [f"wd_{k}" for k in range(1, 7)])
        rows: List[Dict[str, object]] = []
        def _format_val(v):
            """
            系数格式化：正常情况下为科学计数法；若缺失则返回空字符串，保持单元格为空。
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
            p 值格式化：正常情况下保留 4 位小数；若缺失则返回空字符串。
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
            # 显示用的变量名：对外生变量加 exog_ 前缀，其余保持原名
            display_var = f"exog_{var}" if str(var) in exog_base_vars else str(var)

            # r_squared 和 aic 只保留一行系数，不生成 p 值和标准误
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

            # 其余变量：系数行 + p 值行 + 标准误行
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

        # 额外添加 ECM 系数及其 p 值、标准误三行（基于 y_lag 行）
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
            # p 值保留 4 位小数，便于阅读
            ecm_p_row[emo] = _format_p(p_val)
            ecm_se_row[emo] = _format_val(se_val)
        rows.append(ecm_coef_row)
        rows.append(ecm_p_row)
        rows.append(ecm_se_row)

        # 直接构建宽表，不再添加 NOTE 行；NA 的含义可在论文或图表说明中自行解释
        df_wide = pd.DataFrame(rows, columns=["variable"] + emotions)
        out_name = outfile if outfile else ("coefficients_forward_wide.csv" if direction == "forward" else "coefficients_reverse_wide.csv")
        out_path = os.path.join(csv_dir, out_name)
        df_wide.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def append_exog_summary(csv_dir: str, emotion: str, mode: str, pair_key: str, exog_coeffs: Dict[str, Dict[str, object]]):
    """将一次导出的外生变量系数追加到总汇总CSV（exog_coefs_summary.csv）。
    列：emotion, mode, pair_key, variable, coefficient, pvalue
    """
    # 目录与文件
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception:
        return
    summary_path = os.path.join(csv_dir, "exog_coefs_summary.csv")
    # 规范化
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    rows: List[Dict[str, object]] = []
    for var_name, stats in (exog_coeffs or {}).items():
        pval_val = _to_float(stats.get("pvalue", np.nan))
        # 经济学常用显著性标注：* p<0.10, ** p<0.05, *** p<0.01
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
    # 即使空也写出表头，便于查验
    df_rows = pd.DataFrame(rows, columns=["emotion", "mode", "pair_key", "variable", "coefficient", "pvalue", "star"])
    try:
        if not os.path.exists(summary_path):
            # 初次写入（带表头）
            df_rows.to_csv(summary_path, index=False, encoding="utf-8-sig")
        else:
            # 发现旧文件可能没有 star 列，进行一次性迁移
            try:
                existing = pd.read_csv(summary_path)
                if 'star' not in existing.columns:
                    # 为旧数据回填 star 列
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
                    # 以统一列顺序重写
                    cols = ["emotion", "mode", "pair_key", "variable", "coefficient", "pvalue", "star"]
                    for c in cols:
                        if c not in existing.columns:
                            existing[c] = np.nan
                    existing = existing[cols]
                    existing.to_csv(summary_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"  Exogenous coefficients summary migrate-check failed: {e}")
            # 追加新行（不重复表头）
            with open(summary_path, "a", encoding="utf-8-sig", newline="") as f:
                if len(df_rows) > 0:
                    df_rows.to_csv(f, header=False, index=False)
        print(f"  Exogenous coefficients summary updated: {summary_path} (+{len(df_rows)} rows)")
    except Exception as e:
        print(f"  Exogenous coefficients summary update failed: {summary_path}, error={e}")


def export_ecm_coefficients_summary(output_dir: str, coint_results: List[Dict[str, object]]):
    """
    汇总8个情感的ECM系数、短期系数、长期系数及其显著性，按两个方向生成两个CSV文件
    
    Args:
        output_dir: 输出目录
        coint_results: 协整检验结果列表，包含所有ECM结果
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    
    # 定义8个情感
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    
    # 辅助函数：生成显著性星标
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
    
    # 辅助函数：格式化数值
    def _format_value(val):
        try:
            if pd.isna(val) or val == "":
                return ""
            # 如果是字符串格式的科学计数法，先转换
            if isinstance(val, str):
                try:
                    val = float(val)
                except:
                    return val
            return f"{float(val):.4f}"
        except Exception:
            return ""
    
    # 辅助函数：格式化p值
    def _format_pvalue(pval):
        try:
            if pd.isna(pval) or pval == "":
                return ""
            # 如果是字符串格式的科学计数法，先转换
            if isinstance(pval, str):
                try:
                    pval = float(pval)
                except:
                    return pval
            return f"{float(pval):.4f}"
        except Exception:
            return ""

    # 置信区间使用的 z 值（95%）
    try:
        from scipy import stats

        z_critical = stats.norm.ppf(0.975)
    except Exception:
        stats = None
        z_critical = None
    
    # 首先构建一个映射：从(emotion, direction)到coint_result的映射
    result_map = {}  # key: (emotion, direction), value: coint_result

    # 用于存储正向与反向结果的“长表”（上下拼接），第一列为方向
    stacked_rows = []  # each row: direction, emotion, 3 coefficients + their p-values
    
    # 遍历coint_results，识别每个结果对应的情感和方向
    for result in coint_results:
        result_series1 = str(result.get('series1', '')).lower()
        result_series2 = str(result.get('series2', '')).lower()
        
        # 识别情感
        matched_emotion = None
        for emotion in emotions:
            emotion_lower = emotion.lower()
            if emotion_lower in result_series1 or emotion_lower in result_series2:
                matched_emotion = emotion
                break
        
        if not matched_emotion:
            continue
        
        # 识别方向
        # forward: climate在series1，covid在series2
        # reverse: covid在series1，climate在series2
        direction = None
        if 'climate' in result_series1 and 'covid' in result_series2:
            direction = "forward"
        elif 'covid' in result_series1 and 'climate' in result_series2:
            direction = "reverse"
        
        if direction:
            key = (matched_emotion.lower(), direction)
            # 如果已经有结果，优先保留有完整系数信息的
            if key not in result_map:
                result_map[key] = result
            else:
                # 如果新结果有系数信息，而旧结果没有，则替换
                old_result = result_map[key]
                old_has_coef = 'ecm_coefficient' in old_result and not pd.isna(old_result.get('ecm_coefficient', np.nan))
                new_has_coef = 'ecm_coefficient' in result and not pd.isna(result.get('ecm_coefficient', np.nan))
                if new_has_coef and not old_has_coef:
                    result_map[key] = result
    
    # 从coint_results中直接读取数据，确保使用原始值
    for direction in ["forward", "reverse"]:
        rows = []
        rows_sig = []
        rows_ci_low = []
        rows_ci_high = []
        
        for emotion in emotions:
            # 从result_map中查找
            key = (emotion.lower(), direction)
            matched_result = result_map.get(key)
            
            if matched_result:
                # 直接从result字典中读取原始值
                ecm_coef = matched_result.get('ecm_coefficient', np.nan)
                ecm_pval = matched_result.get('ecm_pvalue', np.nan)
                short_coef = matched_result.get('short_run_coefficients', {}).get('delta_x', np.nan)
                short_pval = matched_result.get('short_run_coefficients', {}).get('delta_x_pvalue', np.nan)
                long_coef = matched_result.get('long_run_coefficient', np.nan)
                long_pval = matched_result.get('long_run_pvalue', np.nan)
                
                # 优先使用已保存的置信区间（使用Delta方法计算）
                long_ci_lower = matched_result.get('long_run_ci_lower', np.nan)
                long_ci_upper = matched_result.get('long_run_ci_upper', np.nan)

                # 计算95%置信区间
                ecm_ci = (np.nan, np.nan)
                short_ci = (np.nan, np.nan)
                # 如果已有长期系数的置信区间，直接使用；否则重新计算
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
                        # 短期：dx_lag0 / delta_x
                        dx_idx = design_cols.index("dx_lag0") if "dx_lag0" in design_cols else None
                        if dx_idx is not None and dx_idx < len(params_arr) and dx_idx < cov_arr.shape[0]:
                            var = cov_arr[dx_idx, dx_idx]
                            if pd.notna(var):
                                se = np.sqrt(max(0, var))
                                short_ci = (short_coef - z_critical * se, short_coef + z_critical * se)
                        # 长期：如果还没有计算，使用Delta方法重新计算
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
                # 如果没有找到，使用空值
                ecm_coef = ecm_pval = np.nan
                short_coef = short_pval = np.nan
                long_coef = long_pval = np.nan
                ecm_ci = short_ci = long_ci = (np.nan, np.nan)
            
            ecm_star = _get_star(ecm_pval)
            short_star = _get_star(short_pval)
            long_star = _get_star(long_pval)

            # 追加到“上下拼接长表”（只保留：ECM/短期/长期系数及其 p 值）
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
            
            # 收集系数值
            rows.append({
                "emotion": emotion.upper(),
                "ecm_coefficient": _format_value(ecm_coef),
                "short_run_coefficient": _format_value(short_coef),
                "long_run_coefficient": _format_value(long_coef)
            })
            
            # 收集显著性星号
            rows_sig.append({
                "emotion": emotion.upper(),
                "ecm_coefficient": ecm_star,
                "short_run_coefficient": short_star,
                "long_run_coefficient": long_star
            })

            # 收集置信区间
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
        
        # 保存CSV文件（转置）
        df = pd.DataFrame(rows)
        df_sig = pd.DataFrame(rows_sig)
        df_ci_low = pd.DataFrame(rows_ci_low)
        df_ci_high = pd.DataFrame(rows_ci_high)
        
        # 转置DataFrame：将emotion列作为索引，然后转置
        df_transposed = df.set_index('emotion').T
        df_sig_transposed = df_sig.set_index('emotion').T
        df_ci_low_transposed = df_ci_low.set_index('emotion').T
        df_ci_high_transposed = df_ci_high.set_index('emotion').T
        
        # 重置索引，将原来的列名作为第一列
        df_transposed = df_transposed.reset_index()
        df_sig_transposed = df_sig_transposed.reset_index()
        df_ci_low_transposed = df_ci_low_transposed.reset_index()
        df_ci_high_transposed = df_ci_high_transposed.reset_index()
        
        # 将第一列重命名为合适的名称，并为星号行添加后缀
        df_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_sig_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_sig_transposed['coefficient_type'] = df_sig_transposed['coefficient_type'] + '_sig'
        df_ci_low_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_ci_low_transposed['coefficient_type'] = df_ci_low_transposed['coefficient_type'] + '_ci_low'
        df_ci_high_transposed.rename(columns={'index': 'coefficient_type'}, inplace=True)
        df_ci_high_transposed['coefficient_type'] = df_ci_high_transposed['coefficient_type'] + '_ci_high'
        
        # 合并系数值行和星号行，按系数类型交替排列
        result_rows = []
        for coef_type in ['ecm_coefficient', 'short_run_coefficient', 'long_run_coefficient']:
            # 添加系数值行
            coef_row = df_transposed[df_transposed['coefficient_type'] == coef_type].iloc[0].to_dict()
            result_rows.append(coef_row)
            # 添加置信区间行（下限/上限）
            ci_low_row = df_ci_low_transposed[df_ci_low_transposed['coefficient_type'] == coef_type + '_ci_low'].iloc[0].to_dict()
            result_rows.append(ci_low_row)
            ci_high_row = df_ci_high_transposed[df_ci_high_transposed['coefficient_type'] == coef_type + '_ci_high'].iloc[0].to_dict()
            result_rows.append(ci_high_row)
            # 添加星号行
            sig_row = df_sig_transposed[df_sig_transposed['coefficient_type'] == coef_type + '_sig'].iloc[0].to_dict()
            result_rows.append(sig_row)
        
        df_final = pd.DataFrame(result_rows)
        filename = f"ecm_coefficients_summary_{direction}.csv"
        filepath = os.path.join(output_dir, filename)
        df_final.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  ECM coefficients summary ({direction}) saved: {filepath}")

    # 生成一个额外的 CSV：将 forward / reverse 两个方向上下拼接，第一列为 direction
    if stacked_rows:
        # 固定列顺序，避免因缺失值导致列顺序漂移
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
    汇总8个情感、两个方向的 ARDL bounds test 结果到一个 CSV：
    emotion, direction, Optimal Lags, F-statistic(附显著性星号), 1% Bounds (I0 - I1), 5% Bounds (I0 - I1), Cointegration
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

    def _format_f_with_stars(f_val, p_val) -> str:
        try:
            if pd.isna(f_val):
                return ""
            stars = ""
            if not pd.isna(p_val):
                p = float(p_val)
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
            return f"{float(f_val):.4f}{stars}"
        except Exception:
            return ""

    def _find_optimal_lags(s1: str, s2: str) -> str:
        """
        在 coint_results 中查找与给定 (series1, series2) 匹配的 ECM 结果，读取 selected_p/selected_q 作为最优滞后阶 (p,q)
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
            continue  # 只保留 ARDL bounds test 结果

        s1 = res.get("series1", "")
        s2 = res.get("series2", "")
        emotion = _extract_emotion(s1) or _extract_emotion(s2)
        if emotion not in emotions:
            continue

        direction = _get_direction(s1, s2)
        optimal_lags = _find_optimal_lags(s1, s2)
        f_with_stars = _format_f_with_stars(res.get("f_statistic", np.nan), res.get("p_value", np.nan))
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
    生成短期和长期系数的森林图（置信区间图）
    
    根据图像描述，生成类似的双面板森林图：
    - 面板A：显示某些因变量的短期系数
    - 面板B：显示某些因变量的长期系数
    
    Args:
        output_dir: 输出目录
        coint_results: 协整检验结果列表，包含所有ECM结果
        confidence_level: 置信水平，默认0.95
    """
    try:
        from scipy import stats
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义8个情感
        emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
        
        # 构建结果映射
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
        
        # 定义显著性标志函数
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
        
        # 为每个方向生成森林图，并记录输出路径用于后续组合图
        forest_paths = {}
        for direction in ["forward", "reverse"]:
            # 收集短期和长期系数数据
            short_run_data = []  # [(emotion, coefficient, lower_ci, upper_ci, pval), ...]
            long_run_data = []   # [(emotion, coefficient, lower_ci, upper_ci, pval), ...]
            
            for emotion in emotions:
                key = (emotion.lower(), direction)
                matched_result = result_map.get(key)
                
                if matched_result:
                    # 获取拟合参数信息
                    fp = matched_result.get("_fitted_params", {})
                    params = fp.get("params", None)
                    cov = fp.get("cov", None)
                    design_cols = fp.get("design_cols", [])
                    
                    # 短期系数（delta_x）
                    short_coef = matched_result.get('short_run_coefficients', {}).get('delta_x', np.nan)
                    short_pval = matched_result.get('short_run_coefficients', {}).get('delta_x_pvalue', np.nan)
                    
                    # 计算短期系数的置信区间
                    if not pd.isna(short_coef) and not pd.isna(short_pval) and params is not None and cov is not None:
                        try:
                            # 找到delta_x或dx_lag0的索引
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
                            # 如果计算失败，尝试从p值反推标准误（近似）
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
                    
                    # 长期系数
                    long_coef = matched_result.get('long_run_coefficient', np.nan)
                    long_pval = matched_result.get('long_run_pvalue', np.nan)
                    
                    # 优先使用已保存的置信区间（在ardl_ecm_model中使用Delta方法和t分布计算）
                    long_ci_lower = matched_result.get('long_run_ci_lower', np.nan)
                    long_ci_upper = matched_result.get('long_run_ci_upper', np.nan)
                    
                    if not pd.isna(long_coef):
                        # 如果已有保存的置信区间，直接使用（推荐，因为使用了t分布）
                        if pd.notna(long_ci_lower) and pd.notna(long_ci_upper):
                            long_run_data.append((emotion.upper(), long_coef, long_ci_lower, long_ci_upper, long_pval))
                        # 如果没有保存的置信区间，重新计算（使用Delta方法）
                        elif params is not None and cov is not None:
                            try:
                                # 长期系数 = -params[x_lag] / params[y_lag]
                                # 需要计算这个非线性函数的置信区间
                                ylag_idx = design_cols.index('y_lag') if 'y_lag' in design_cols else None
                                xlag_idx = design_cols.index('x_lag') if 'x_lag' in design_cols else None
                                
                                if ylag_idx is not None and xlag_idx is not None:
                                    a = params[ylag_idx]  # y_lag系数
                                    d = params[xlag_idx]  # x_lag系数
                                    
                                    if abs(a) > 1e-10:
                                        # Delta方法：长期系数的方差
                                        # var(long_run) ≈ (1/a²) * var(d) + (d²/a⁴) * var(a) - 2*(d/a³) * cov(a,d)
                                        # 处理协方差矩阵可能是DataFrame或numpy数组的情况
                                        if hasattr(cov, 'iloc'):
                                            var_a = cov.iloc[ylag_idx, ylag_idx]
                                            var_d = cov.iloc[xlag_idx, xlag_idx]
                                            cov_ad = cov.iloc[ylag_idx, xlag_idx] if ylag_idx < cov.shape[0] and xlag_idx < cov.shape[1] else 0
                                        else:
                                            var_a = cov[ylag_idx, ylag_idx]
                                            var_d = cov[xlag_idx, xlag_idx]
                                            cov_ad = cov[ylag_idx, xlag_idx] if ylag_idx < cov.shape[0] and xlag_idx < cov.shape[1] else 0
                                        
                                        var_long = (1 / (a**2)) * var_d + (d**2 / (a**4)) * var_a - 2 * (d / (a**3)) * cov_ad
                                        se_long = np.sqrt(max(0, var_long))  # 确保非负
                                        
                                        # 尝试使用t分布（如果可能），否则使用正态分布
                                        try:
                                            # 计算自由度（需要从结果中获取观测数）
                                            # 如果无法获取，使用正态分布作为近似
                                            n_obs = len(params) + 10  # 粗略估计，实际应该从模型结果获取
                                            k_params = len(params)
                                            df = max(1, n_obs - k_params)
                                            t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
                                            lower_ci = long_coef - t_critical * se_long
                                            upper_ci = long_coef + t_critical * se_long
                                        except Exception:
                                            # 如果t分布计算失败，使用正态分布
                                            z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                                            lower_ci = long_coef - z_critical * se_long
                                            upper_ci = long_coef + z_critical * se_long
                                        
                                        long_run_data.append((emotion.upper(), long_coef, lower_ci, upper_ci, long_pval))
                            except Exception:
                                # 如果Delta方法失败，尝试从p值反推（最后的后备方案）
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
            
            # 如果数据不足，跳过绘图
            if not short_run_data and not long_run_data:
                continue
            
            # 创建图形：双面板布局
            # 设置方向标注
            direction_label = "COVID→Climate Change" if direction == "forward" else "Climate Change→COVID"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
            # 只保留括号内方向文本作为标题
            fig.suptitle(f'{direction_label}', fontsize=16, fontweight='bold')

            # 在整张图左上角添加小写标记（forward 用 a，reverse 用 b）
            panel_label = "a" if direction == "forward" else "b"
            try:
                fig.text(0.02, 0.98, panel_label, fontsize=16, fontweight='bold', va='top', ha='left')
            except Exception:
                pass
            
            # 面板A：短期系数
            if short_run_data:
                # 不显示面板标题（title）
                ax1.set_title('')
                emotions_short, coefs_short, lowers_short, uppers_short, pvals_short = zip(*short_run_data)
                y_pos_short = np.arange(len(emotions_short))
                
                # 绘制置信区间
                for i, (emotion, coef, lower, upper, pval) in enumerate(short_run_data):
                    ax1.plot([lower, upper], [i, i], color='#1f77b4', linewidth=2.5, alpha=0.8)
                    ax1.plot(coef, i, 'o', color='#1f77b4', markersize=8)
                    # 标注系数值
                    ax1.text(coef, i + 0.15, f'{coef:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # 创建带显著性标志的y轴标签
                emotion_labels_short = [f"{emotion} {_star(pval)}" for emotion, pval in zip(emotions_short, pvals_short)]
                
                ax1.set_yticks(y_pos_short)
                ax1.set_yticklabels(emotion_labels_short)
                ax1.set_xlabel('Short-run coefficient', fontsize=12)
                ax1.set_ylabel('emotions', fontsize=12)
                # 只保留坐标轴，无网格；保留黑色虚线零线
                ax1.grid(False)
                ax1.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                # 取消浅蓝色背景，使用默认白色背景
                ax1.set_facecolor('white')
            
            # 面板B：长期系数
            if long_run_data:
                # 不显示面板标题（title）
                ax2.set_title('')
                emotions_long, coefs_long, lowers_long, uppers_long, pvals_long = zip(*long_run_data)
                y_pos_long = np.arange(len(emotions_long))
                
                # 绘制置信区间
                for i, (emotion, coef, lower, upper, pval) in enumerate(long_run_data):
                    ax2.plot([lower, upper], [i, i], color='#1f77b4', linewidth=2.5, alpha=0.8)
                    ax2.plot(coef, i, 'o', color='#1f77b4', markersize=8)
                    # 标注系数值
                    ax2.text(coef, i + 0.15, f'{coef:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # 创建带显著性标志的y轴标签
                emotion_labels_long = [f"{emotion} {_star(pval)}" for emotion, pval in zip(emotions_long, pvals_long)]
                
                ax2.set_yticks(y_pos_long)
                ax2.set_yticklabels(emotion_labels_long)
                ax2.set_xlabel('Long-run coefficient', fontsize=12)
                ax2.set_ylabel('emotions', fontsize=12)
                # 只保留坐标轴，无网格；保留黑色虚线零线
                ax2.grid(False)
                ax2.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                # 取消浅蓝色背景，使用默认白色背景
                ax2.set_facecolor('white')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图形
            output_path = os.path.join(output_dir, f"coefficients_forest_plot_{direction}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            forest_paths[direction] = output_path
            print(f"  短期长期系数置信区间图 ({direction}) 已保存: {output_path}")

        # 生成上下拼接的组合图（forward 在上，reverse 在下）
        try:
            import matplotlib.image as mpimg

            forward_path = forest_paths.get("forward")
            reverse_path = forest_paths.get("reverse")
            if forward_path and reverse_path and os.path.exists(forward_path) and os.path.exists(reverse_path):
                img_forward = mpimg.imread(forward_path)
                img_reverse = mpimg.imread(reverse_path)

                # 创建新的画布：上下两个子图
                fig_combined, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 12))

                ax_top.imshow(img_forward)
                ax_top.axis('off')

                ax_bottom.imshow(img_reverse)
                ax_bottom.axis('off')

                plt.tight_layout()
                combined_path = os.path.join(output_dir, "coefficients_forest_plot_combined.png")
                fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
                plt.close(fig_combined)
                print(f"  组合森林图 (forward+reverse) 已保存: {combined_path}")
        except Exception as _:
            # 若组合图失败，不影响单独图像输出
            pass
        
    except Exception as e:
        print(f"  生成短期长期系数置信区间图失败: {str(e)}")
        import traceback
        traceback.print_exc()


## 已删除：残差诊断汇总导出（DW/LBQ/正态性/异方差）


def summarize_effects(csv_dir: str, pair_key: str, ecm_result: Dict[str, object], irf: Dict[str, object], horizons: List[int]):
    """汇总当期（h=0）、累计h期、长期效应。"""
    try:
        b = ecm_result.get("short_run_coefficients", {}).get("delta_x", np.nan)
        long_run = irf.get("long_run", np.nan)
        h_arr = irf.get("h", np.array([]))
        y_arr = irf.get("irf", np.array([]))
        rows: List[Dict[str, object]] = []
        for H in horizons:
            if len(y_arr) > H:
                cumulative = float(y_arr[H])  # 因为 y 是水平响应，已是累计
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

        # 附加：汇总一条“快速核对”记录到总表（H=14优先，否则取最大H）
        try:
            # 解析情绪与方向
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
            # 目标H
            H_target = 14
            if not isinstance(horizons, (list, tuple)) or (14 not in horizons):
                try:
                    H_target = int(np.max(horizons)) if horizons else 14
                except Exception:
                    H_target = 14
            level_at_H = float(y_arr[H_target]) if (len(y_arr) > H_target and np.isfinite(y_arr[H_target])) else np.nan
            ecm_coef = ecm_result.get("ecm_coefficient", np.nan)
            # 写入/追加
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


def run_fmols_for_emotions(
    df: pd.DataFrame,
    emotion_groups: Dict[str, Dict[str, pd.Series]],
    filtered_emotions: List[str],
    output_dir: str,
) -> None:
    """
    使用 Phillips-Hansen FMOLS 对每个情绪跑 2 个方向（covid→climate, climate→covid）的协整回归，
    理论上最多得到 16 个模型（8 个情绪 × 2 个方向），结果汇总到一个 CSV 中。
    """
    if FullyModifiedOLS is None:
        print("\nFMOLS 模型跳过：未安装 arch 库（pip install arch 即可启用 FMOLS）")
        return

    try:
        # 使用正态近似，根据 t 值计算双侧 p 值；避免依赖特定版本的 arch 对 pvalues 的实现
        def _pvalue_from_t(t_val: float) -> float:
            try:
                if pd.isna(t_val):
                    return np.nan
                # 双侧检验：p = 2 * (1 - Φ(|t|))；Φ 为标准正态分布的累积分布函数
                return float(2 * (1 - norm.cdf(abs(t_val))))
            except Exception:
                return np.nan

        rows: List[Dict[str, object]] = []
        model_count = 0

        for emotion in filtered_emotions:
            group = emotion_groups.get(emotion, {})
            if "climate" not in group or "covid" not in group:
                continue

            climate_col = f"climate_{emotion}_score_freq"
            covid_col = f"covid_{emotion}_score_freq"

            if climate_col not in df.columns or covid_col not in df.columns:
                continue

            # 清理并对齐两列
            y_climate = pd.to_numeric(df[climate_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            y_covid = pd.to_numeric(df[covid_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

            pair_df = pd.concat([y_climate.rename(climate_col), y_covid.rename(covid_col)], axis=1).dropna()
            if pair_df.shape[0] < 30:
                # FMOLS 需要足够样本
                continue

            # 方向 1：covid → climate（以 climate 为被解释变量）
            try:
                fm1 = FullyModifiedOLS(pair_df[climate_col], pair_df[[covid_col]], trend="c")
                res1 = fm1.fit()
                model_count += 1
                for param_name, coef in res1.params.items():
                    t_val = float(res1.tvalues.get(param_name, np.nan))
                    rows.append(
                        {
                            "emotion": emotion,
                            "direction": "covid_to_climate",
                            "dep_var": climate_col,
                            "indep_var": covid_col if param_name != "const" else "const",
                            "param": str(param_name),
                            "coefficient": float(coef) if pd.notna(coef) else np.nan,
                            "std_error": float(res1.std_errors.get(param_name, np.nan)),
                            "tvalue": t_val,
                            "pvalue": _pvalue_from_t(t_val),
                            "rsquared": float(res1.rsquared),
                            "rsquared_adj": float(res1.rsquared_adj),
                            "bandwidth": float(res1.bandwidth),
                            "kernel": str(res1.kernel),
                            "nobs": int(pair_df.shape[0]),
                        }
                    )
            except Exception as e:
                print(f"  FMOLS covid→climate（{emotion}）失败: {str(e)[:80]}")

            # 方向 2：climate → covid（以 covid 为被解释变量）
            try:
                fm2 = FullyModifiedOLS(pair_df[covid_col], pair_df[[climate_col]], trend="c")
                res2 = fm2.fit()
                model_count += 1
                for param_name, coef in res2.params.items():
                    t_val = float(res2.tvalues.get(param_name, np.nan))
                    rows.append(
                        {
                            "emotion": emotion,
                            "direction": "climate_to_covid",
                            "dep_var": covid_col,
                            "indep_var": climate_col if param_name != "const" else "const",
                            "param": str(param_name),
                            "coefficient": float(coef) if pd.notna(coef) else np.nan,
                            "std_error": float(res2.std_errors.get(param_name, np.nan)),
                            "tvalue": t_val,
                            "pvalue": _pvalue_from_t(t_val),
                            "rsquared": float(res2.rsquared),
                            "rsquared_adj": float(res2.rsquared_adj),
                            "bandwidth": float(res2.bandwidth),
                            "kernel": str(res2.kernel),
                            "nobs": int(pair_df.shape[0]),
                        }
                    )
            except Exception as e:
                print(f"  FMOLS climate→covid（{emotion}）失败: {str(e)[:80]}")

        if rows:
            fmols_df = pd.DataFrame(rows)
            out_path = os.path.join(output_dir, "fmols_results.csv")
            fmols_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"\nFMOLS 模型估计完成，共估计 {model_count} 个模型（理论上最多 16 个），结果保存至: {out_path}")
        else:
            print("\nFMOLS 模型未估计：没有满足样本量要求的情绪配对。")
    except Exception as e:
        print(f"\nFMOLS 分析出错: {str(e)[:120]}")


def main():
    # 直接运行（无参数）来执行完整分析
    print("=" * 60)
    print("Complete Analysis Mode - All Features Included")
    print("=" * 60)
    
    # 设置默认路径
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
    # 专用外生系数导出目录
    exog_output_dir = os.path.join(output_dir, "exog_coeffs")
    try:
        os.makedirs(exog_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create exog output dir: {exog_output_dir}, error={e}")
    
    # 设置默认参数
    args = type('Args', (), {
        'input': input_file,
        'output': output_file,
        'date_col': None,
        # 不再在单位根检验/协整检验前预先剔除工作日效应与趋势
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

        # 导出：工作日效应与趋势回归报告（按当前设置）
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
            # 导出 weekday OLS 汇总宽表
            try:
                build_weekday_ols_wide(weekday_dir, outfile="weekday_ols_summary_wide.csv")
            except Exception:
                pass
        except Exception:
            pass
        
        # Run stationarity tests
        print("\n=== Running Stationarity Tests ===")
        integ_rows = []
        diffed_df = df[[date_col]].copy()
        
        for col in emotion_columns:
            base_series = df[col]
            # 直接对原始序列做单位根检验（不预先去除工作日效应/趋势）
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
        integ_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Stationarity test completed, saved to: {output_file}")
        
        # Save differenced data
        diffed_path = os.path.splitext(output_file)[0] + "_diffed_to_stationary.csv"
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
            series_name = row['series']
            if series_name.startswith('climate_'):
                emotion = series_name.replace('climate_', '').replace('_score_freq', '')
                if emotion not in emotion_groups:
                    emotion_groups[emotion] = {}
                emotion_groups[emotion]['climate'] = row
            elif series_name.startswith('covid_'):
                emotion = series_name.replace('covid_', '').replace('_score_freq', '')
                if emotion not in emotion_groups:
                    emotion_groups[emotion] = {}
                emotion_groups[emotion]['covid'] = row
        
        filtered_emotions = [e for e in sorted(emotion_groups.keys()) if e not in excluded_emotions]
        
        # Run causality analysis
        print("\n=== Running Causality Analysis ===")
        coint_results = []
        
        for emotion in filtered_emotions:
            group = emotion_groups[emotion]
            if 'climate' in group and 'covid' in group:
                climate_order = group['climate']['final_order']
                covid_order = group['covid']['final_order']
                
                climate_col = f"climate_{emotion}_score_freq"
                covid_col = f"covid_{emotion}_score_freq"
                climate_series = df[climate_col]
                covid_series = df[covid_col]
                
                # Determine analysis method based on integration orders
                if (climate_order == 1 and covid_order == 1) or (climate_order == 0 and covid_order == 1) or (climate_order == 1 and covid_order == 0):
                    # I(1)+I(1) 或 I(0)+I(1)/I(1)+I(0)：统一使用 ARDL bounds test
                    print(f"\n{emotion.upper()} emotion: I({climate_order}) + I({covid_order}) - Running ARDL bounds tests")
                    ardl_result = ardl_bounds_test(climate_series, covid_series, climate_col, covid_col,
                                                  dates=df[date_col], weekday_ols=args.weekday_ols, weekday_trend=args.weekday_trend)
                    coint_results.append(ardl_result)
                    if ardl_result.get('error'):
                        print(f"  Error: {ardl_result['error']}")
                    else:
                        print(f"  F-statistic: {ardl_result['f_statistic']:.4f}")
                        print(f"  Bounds test: {ardl_result['bounds_result']}")
                        print(f"  Long-run relationship: {ardl_result['long_run_result']}")
                        
                        # If bounds test passes, run ECM model
                        if ardl_result['bounds_result'] == 'cointegrated':
                                print(f"  Running ARDL-ECM model...")
                                exog_vars = {}
                                if exogenous_columns:
                                    for exog_col in exogenous_columns:
                                        exog_vars[exog_col] = df[exog_col]
                                # 加入工作日虚拟变量（周二~周日），周一为基准
                                try:
                                    wd_df = build_weekday_dummies(df[date_col], index=df.index, prefix="wd")
                                    for c in wd_df.columns:
                                        if c not in exog_vars:
                                            exog_vars[c] = wd_df[c]
                                except Exception:
                                    pass
                                
                                ecm_result = ardl_ecm_model(climate_series, covid_series, climate_col, covid_col,
                                                          dates=df[date_col], weekday_ols=args.weekday_ols, weekday_trend=args.weekday_trend, exog_vars=exog_vars)
                                coint_results.append(ecm_result)
                                ecm_error = ecm_result.get('error')
                                if ecm_error:
                                    print(f"    Error: {ecm_result['error']}")
                                if not ecm_error:
                                    _print_and_export_ecm("forward", str(emotion), climate_col, covid_col, ecm_result, ardl_result, output_dir, exog_output_dir, args)
                                
                                # 方向2：covid -> climate（以covid为被解释变量）
                                print(f"  Running reverse ARDL bounds test...")
                                ardl_result_rev = ardl_bounds_test(
                                    covid_series,
                                    climate_series,
                                    covid_col,
                                    climate_col,
                                    dates=df[date_col],
                                    weekday_ols=args.weekday_ols,
                                    weekday_trend=args.weekday_trend,
                                )
                                # 无论检验结果如何，都将反向 ARDL bounds test 结果加入 coint_results，
                                # 便于后续统一导出 8 个情绪 × 2 个方向的 bounds test 汇总
                                coint_results.append(ardl_result_rev)
                                if ardl_result_rev.get('error'):
                                    print(f"  Reverse ARDL bounds test error: {ardl_result_rev['error']}")
                                else:
                                    print(f"  Reverse F-statistic: {ardl_result_rev['f_statistic']:.4f}")
                                    print(f"  Reverse bounds test: {ardl_result_rev['bounds_result']}")
                                    print(f"  Reverse long-run relationship: {ardl_result_rev['long_run_result']}")
                                    
                                    if ardl_result_rev['bounds_result'] == 'cointegrated':
                                        print(f"  Running reverse ARDL-ECM model...")
                                        ecm_result_2 = ardl_ecm_model(
                                            covid_series,
                                            climate_series,
                                            covid_col,
                                            climate_col,
                                            dates=df[date_col],
                                            weekday_ols=args.weekday_ols,
                                            weekday_trend=args.weekday_trend,
                                            exog_vars=exog_vars,
                                        )
                                        coint_results.append(ecm_result_2)
                                        _print_and_export_ecm("reverse", str(emotion), covid_col, climate_col, ecm_result_2, ardl_result_rev, output_dir, exog_output_dir, args)
                                            
                                            # Draw DAG (disabled)
                                            # try:
                                            #     edges = []
                                            #     if covid_to_climate_exists:
                                            #         edges.append(('covid', 'climate'))
                                            #     if climate_to_covid_exists:
                                            #         edges.append(('climate', 'covid'))
                                            #     if edges:
                                            #         save_causality_dag(output_dir, emotion, edges)
                                            # except Exception:
                                            #     pass
                                    else:
                                        print(f"  Reverse bounds test failed, no reverse ECM model")
                
                else:
                    # I(0) + I(0)：Pesaran ARDL 边界检验同样适用（临界值覆盖 I(0) 情形），需做 forward 与 reverse 两个方向的 bounds test
                    print(f"\n{emotion.upper()} emotion: I(0) + I(0) - Running ARDL bounds tests (applicable for I(0) variables per Pesaran et al.)")
                    ardl_result = ardl_bounds_test(climate_series, covid_series, climate_col, covid_col,
                                                  dates=df[date_col], weekday_ols=args.weekday_ols, weekday_trend=args.weekday_trend)
                    coint_results.append(ardl_result)
                    if ardl_result.get('error'):
                        print(f"  Error: {ardl_result['error']}")
                    else:
                        print(f"  F-statistic: {ardl_result['f_statistic']:.4f}")
                        print(f"  Bounds test: {ardl_result['bounds_result']}")
                        print(f"  Long-run relationship: {ardl_result['long_run_result']}")

                    # reverse bounds test（与 I(1)+I(1) 分支一致）
                    print(f"  Running reverse ARDL bounds test...")
                    ardl_result_rev = ardl_bounds_test(covid_series, climate_series, covid_col, climate_col,
                                                      dates=df[date_col], weekday_ols=args.weekday_ols, weekday_trend=args.weekday_trend)
                    coint_results.append(ardl_result_rev)
                    if ardl_result_rev.get('error'):
                        print(f"  Reverse ARDL bounds test error: {ardl_result_rev['error']}")
                    else:
                        print(f"  Reverse F-statistic: {ardl_result_rev['f_statistic']:.4f}")
                        print(f"  Reverse bounds test: {ardl_result_rev['bounds_result']}")
                        print(f"  Reverse long-run relationship: {ardl_result_rev['long_run_result']}")

                    exog_vars = {}
                    if exogenous_columns:
                        for exog_col in exogenous_columns:
                            exog_vars[exog_col] = df[exog_col]
                    # 加入工作日虚拟变量（周二~周日），周一为基准
                    try:
                        wd_df = build_weekday_dummies(df[date_col], index=df.index, prefix="wd")
                        for c in wd_df.columns:
                            if c not in exog_vars:
                                exog_vars[c] = wd_df[c]
                    except Exception:
                        pass

                    # forward: covid → climate（climate 为被解释变量）
                    ecm_result = ardl_ecm_model(
                        climate_series,
                        covid_series,
                        climate_col,
                        covid_col,
                        dates=df[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                        exog_vars=exog_vars,
                        exog_significant_only=args.exog_significant_only,
                        exog_threshold=args.exog_threshold,
                    )
                    coint_results.append(ecm_result)
                    if not ecm_result.get("error"):
                        _print_and_export_ecm("forward", str(emotion), climate_col, covid_col, ecm_result, ardl_result, output_dir, exog_output_dir, args)

                    # reverse: climate → covid（covid 为被解释变量）
                    ecm_result_rev = ardl_ecm_model(
                        covid_series,
                        climate_series,
                        covid_col,
                        climate_col,
                        dates=df[date_col],
                        weekday_ols=args.weekday_ols,
                        weekday_trend=args.weekday_trend,
                        exog_vars=exog_vars,
                        exog_significant_only=args.exog_significant_only,
                        exog_threshold=args.exog_threshold,
                    )
                    coint_results.append(ecm_result_rev)
                    if not ecm_result_rev.get("error"):
                        _print_and_export_ecm("reverse", str(emotion), covid_col, climate_col, ecm_result_rev, ardl_result_rev, output_dir, exog_output_dir, args)
            
            # Save causality results
            if coint_results:
                coint_df = pd.DataFrame(coint_results)
                coint_path = os.path.splitext(output_file)[0] + "_causality.csv"
                coint_df.to_csv(coint_path, index=False, encoding="utf-8-sig")
                print(f"\nCausality analysis results saved to: {coint_path}")

            # Export ARDL bounds test summary (8 emotions × 2 directions)
            print("\n=== Generating ARDL Bounds Test Summary ===")
            export_ardl_bounds_summary(output_dir, coint_results)

            # Export ECM coefficients summary for 8 emotions
            print("\n=== Generating ECM Coefficients Summary ===")
            export_ecm_coefficients_summary(output_dir, coint_results)

            # Generate forest plot for short-run and long-run coefficients
            print("\n=== Generating Short-Run and Long-Run Coefficients Forest Plot ===")
            plot_short_long_coefficients_forest(output_dir, coint_results, confidence_level=0.95)

            # Run FMOLS models (up to 16 models: 8 emotions × 2 directions)
            print("\n=== Running FMOLS Cointegration Regressions (up to 16 models) ===")
            run_fmols_for_emotions(df, emotion_groups, filtered_emotions, output_dir)

            # Generate time series plots if requested
            if args.timeseries_plots:
                print("\n=== Generating Time Series Plots ===")
                print(f"Will generate time series plots for {len(filtered_emotions)} emotion groups")
                
                for emotion in filtered_emotions:
                    try:
                        plot_emotion_timeseries(df, date_col, emotion, output_dir)
                    except Exception as e:
                        print(f"Failed to generate {emotion} time series plot: {str(e)}")
                
                print(f"\nTime series plot generation completed, saved in: {output_dir}")
            
            print("\nComplete analysis finished successfully!")
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")


if __name__ == "__main__":
    main()
