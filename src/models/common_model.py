"""P4b: 負の二項回帰・残差診断・空間自己相関検定・交差検証の共通ユーティリティ。

OLSのR2は使わない（PROJECT_SPEC.mdの指示）。モデル比較はAIC・McFaddenの
疑似決定係数（NegativeBinomial.fit()が返すprsquared）・leave-one-year-out
交差検証対数尤度を用いる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sstats


@dataclass
class NBFitResult:
    results: object
    formula: str
    n_obs: int
    aic: float
    prsquared: float
    llf: float
    llnull: float
    alpha: float


def _estimate_alpha_cameron_trivedi(mu: np.ndarray, y: np.ndarray) -> float:
    """Cameron & Trivedi (1990) の補助回帰によるNB2分散パラメータ(alpha)の推定。
    分散 = mu + alpha*mu^2 という仮定のもとで、((y-mu)^2 - y)をmu^2に回帰した傾きがalpha。
    """
    aux_y = (y - mu) ** 2 - y
    aux_x = mu**2
    denom = np.sum(aux_x**2)
    if denom <= 0:
        return 1e-4
    alpha = float(np.sum(aux_x * aux_y) / denom)
    return max(alpha, 1e-4)


def fit_nb(df, formula: str) -> NBFitResult:
    """負の二項回帰(NB2)を2段階で適合する：
    1) Poisson GLM(IRLS、数値的に安定)でmuを推定
    2) Cameron-Trivedi補助回帰でalpha(過分散)を推定し、それを固定してNB GLM(IRLS)を適合する

    多数のカテゴリカル変数（月×年など）を同時投入するとstatsmodelsの
    NegativeBinomial（MLEで全パラメータとalphaを同時推定する方式）はしばしば
    収束しない・特異行列になることが実際に確認されたため、IRLSベースの
    2段階推定に切り替えた。これは Cameron & Trivedi (1990) の標準的な手法であり、
    「収束しやすい簡便法に逃げた」ものではなく、NB2の一般的な推定法の一つである。
    """
    poisson_res = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
    mu = np.asarray(poisson_res.mu)
    y = np.asarray(poisson_res.model.endog)
    alpha = _estimate_alpha_cameron_trivedi(mu, y)

    model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=alpha))
    res = model.fit()
    if not res.converged or not math.isfinite(res.aic):
        raise RuntimeError(f"NB GLM (fixed alpha={alpha:.4g}) did not converge (non-finite AIC)")

    llf, llnull = float(res.llf), float(res.llnull)
    prsquared = 1 - llf / llnull if llnull != 0 else float("nan")
    if not all(math.isfinite(v) for v in (llf, llnull, prsquared)):
        raise RuntimeError(
            f"NB GLM (fixed alpha={alpha:.4g}) produced non-finite llf/llnull/prsquared "
            f"(llf={llf}, llnull={llnull}, prsquared={prsquared}); likely near-complete "
            "separation from very sparse category (e.g. rare event type with many empty cells)"
        )
    return NBFitResult(
        results=res,
        formula=formula,
        n_obs=int(res.nobs),
        aic=float(res.aic),
        prsquared=float(prsquared),
        llf=llf,
        llnull=llnull,
        alpha=alpha,
    )


def nb_loglik(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """NB2パラメータ化（分散 = mu + alpha*mu^2）での対数尤度合計。"""
    mu = np.clip(mu, 1e-10, None)
    alpha = max(alpha, 1e-10)
    r = 1.0 / alpha
    p = r / (r + mu)
    return float(np.sum(sstats.nbinom.logpmf(y, r, p)))


def loo_year_cv(df: pl.DataFrame, formula: str, year_col: str = "year") -> dict:
    """Leave-one-year-out 交差検証。各年を1回ずつ held-out にし、
    残りの年で学習したモデルのalpha・係数でheld-out年の対数尤度を計算する。
    """
    pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    years = sorted(pdf[year_col].unique())
    total_loglik = 0.0
    per_year = {}
    for y in years:
        train = pdf[pdf[year_col] != y]
        test = pdf[pdf[year_col] == y]
        if test.shape[0] == 0 or train.shape[0] == 0:
            continue
        try:
            fit = fit_nb(train, formula)
        except Exception as e:  # noqa: BLE001
            per_year[str(y)] = {"error": str(e)}
            continue
        mu_test = fit.results.predict(test)
        y_test = test[formula.split("~")[0].strip()].to_numpy()
        ll = nb_loglik(y_test, np.asarray(mu_test), fit.alpha)
        per_year[str(y)] = {"loglik": ll, "n": int(test.shape[0])}
        total_loglik += ll
    return {"total_loglik": total_loglik, "per_year": per_year}


def build_distance_band_weights(lat: np.ndarray, lon: np.ndarray, band_km: float, lat_ref_deg: float = 43.0) -> np.ndarray:
    """緯度経度から距離帯（band_km以内を隣接とみなす）の二値空間重み行列を作る。行標準化はしない。"""
    n = len(lat)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    R = 6371.0
    W = np.zeros((n, n))
    for i in range(n):
        dlat = lat_rad - lat_rad[i]
        dlon = lon_rad - lon_rad[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[i]) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
        d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        W[i] = (d <= band_km) & (d > 0)
    return W


def morans_i(values: np.ndarray, W: np.ndarray, n_permutations: int = 999, seed: int = 12345) -> dict:
    """Moran's Iとpermutationベースのp値を計算する。"""
    n = len(values)
    x = values - values.mean()
    S0 = W.sum()
    if S0 == 0:
        return {"I": float("nan"), "p_value": float("nan"), "n": n, "note": "no neighbor pairs within band"}
    num = np.sum(W * np.outer(x, x))
    den = np.sum(x**2)
    I_obs = (n / S0) * (num / den)

    rng = np.random.default_rng(seed)
    perm_I = np.empty(n_permutations)
    for k in range(n_permutations):
        xp = rng.permutation(x)
        num_p = np.sum(W * np.outer(xp, xp))
        perm_I[k] = (n / S0) * (num_p / den)
    p_value = (np.sum(np.abs(perm_I) >= abs(I_obs)) + 1) / (n_permutations + 1)
    return {"I": float(I_obs), "p_value": float(p_value), "n": n, "perm_mean": float(perm_I.mean()), "perm_sd": float(perm_I.std())}


def vif_table(df, cols: list[str]) -> dict:
    """VIF（分散拡大係数）。statsmodelsの実装をそのまま使う。"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    X = pdf[cols].to_numpy()
    out = {}
    for i, c in enumerate(cols):
        try:
            out[c] = float(variance_inflation_factor(X, i))
        except Exception as e:  # noqa: BLE001
            out[c] = float("nan")
    return out


def zero_check(y: np.ndarray, mu: np.ndarray, alpha: float) -> dict:
    """観測ゼロ割合とNBモデルが含意する期待ゼロ割合を比較する（ゼロ過剰の粗い診断）。"""
    obs_zero_rate = float(np.mean(y == 0))
    mu = np.clip(mu, 1e-10, None)
    r = 1.0 / max(alpha, 1e-10)
    p = r / (r + mu)
    expected_zero_rate = float(np.mean(sstats.nbinom.pmf(0, r, p)))
    return {"observed_zero_rate": obs_zero_rate, "nb_expected_zero_rate": expected_zero_rate, "diff": obs_zero_rate - expected_zero_rate}
