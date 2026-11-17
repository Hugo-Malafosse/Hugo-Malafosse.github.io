"""Outils du notebook de démonstration du TD2 (équipe enseignante).

Ce fichier contient des implémentations complètes (tirage discret, EM pour mélange gaussien,
population Monte Carlo). Il sert à produire les figures du notebook : NE PAS le projeter ni le
distribuer avant la date de rendu, il contient les réponses aux questions de code du TD.
"""
import time
import numpy as np
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2E5EAA", "#D9731A", "#3A8F55", "#B23A48", "#7A7A7A", "#6A4C93"
COLORS = [BLUE, ORANGE, GREEN, PURPLE, RED, GRAY]


def style():
    plt.rcParams.update({"figure.figsize": (7, 3), "figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "font.size": 10})


# ---------------------------------------------------------------- exercice 1
def sample_discrete(p, x, N, rng):
    """Inversion de la fonction de répartition discrète (recherche dichotomique)."""
    cdf = np.cumsum(p)
    cdf[-1] = 1.0                              # évite U > cdf[-1] à cause des arrondis
    k = np.searchsorted(cdf, rng.random(N), side="right")
    return np.asarray(x)[k]


def sample_discrete_linear(p, x, N, rng):
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    U = rng.random(N)
    k = (U[:, None] >= cdf[None, :]).sum(1)    # balayage linéaire, O(n) par tirage
    return np.asarray(x)[k]


def alias_table(p):
    n = len(p)
    prob = np.asarray(p, float) * n
    alias = np.zeros(n, int)
    small = [i for i in range(n) if prob[i] < 1]
    large = [i for i in range(n) if prob[i] >= 1]
    while small and large:
        s, l = small.pop(), large.pop()
        alias[s] = l
        prob[l] -= 1 - prob[s]
        (small if prob[l] < 1 else large).append(l)
    for i in small + large:
        prob[i] = 1.0
    return prob, alias


def sample_alias(prob, alias, x, N, rng):
    k = rng.integers(len(prob), size=N)
    k = np.where(rng.random(N) < prob[k], k, alias[k])
    return np.asarray(x)[k]


def timing_discrete(sizes=(10, 100, 1000, 10000), N=200_000, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    for n in sizes:
        p = rng.random(n); p /= p.sum(); x = np.arange(n)
        row = {"n": n}
        if n <= 1000:
            t0 = time.perf_counter(); sample_discrete_linear(p, x, N, rng); row["lineaire"] = time.perf_counter() - t0
        t0 = time.perf_counter(); sample_discrete(p, x, N, rng); row["dichotomie"] = time.perf_counter() - t0
        prob, alias = alias_table(p)
        t0 = time.perf_counter(); sample_alias(prob, alias, x, N, rng); row["alias"] = time.perf_counter() - t0
        out.append(row)
    return out


# ---------------------------------------------------------------- exercice 2
def gmm_sample(n, alphas, mus, Sigmas, rng):
    z = sample_discrete(alphas, np.arange(len(alphas)), n, rng)
    X = np.array([rng.multivariate_normal(mus[j], Sigmas[j]) for j in z])
    return X, z


def log_gauss(X, mu, Sigma):
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    sol = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (sol**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


def gmm_logcomp(X, alphas, mus, Sigmas):
    return np.column_stack([np.log(a) + log_gauss(X, m, S) for a, m, S in zip(alphas, mus, Sigmas)])


def gmm_loglik(X, alphas, mus, Sigmas, w=None):
    lp = logsumexp(gmm_logcomp(X, alphas, mus, Sigmas), axis=1)
    return lp.sum() if w is None else (w * lp).sum()


def em_gmm(X, m, rng, init=None, n_iter=500, tol=1e-9, reg=1e-6, w=None, keep_history=False):
    """EM (éventuellement pondéré par des poids w normalisés) pour un mélange gaussien."""
    n, d = X.shape
    w = np.full(n, 1.0 / n) if w is None else w / w.sum()
    if init is None:
        idx = rng.choice(n, m, replace=False, p=w)
        alphas, mus = np.full(m, 1.0 / m), X[idx].copy()
        C = np.cov(X.T, aweights=w).reshape(d, d)
        Sigmas = np.array([C.copy() for _ in range(m)])
    else:
        alphas, mus, Sigmas = (np.array(a, float).copy() for a in init)
    hist, lls = [], []
    for t in range(n_iter):
        lc = gmm_logcomp(X, alphas, mus, Sigmas)
        lp = logsumexp(lc, axis=1)
        lls.append(n * (w * lp).sum())
        if keep_history:
            hist.append((alphas.copy(), mus.copy(), Sigmas.copy()))
        tau = np.exp(lc - lp[:, None]) * w[:, None]
        nk = tau.sum(0)
        dead = nk < 1e-10 * nk.sum()               # composante vide : on garde ses anciens paramètres
        new_mus = (tau.T @ X) / np.maximum(nk, 1e-300)[:, None]
        new_S = np.array([((tau[:, j, None] * (X - new_mus[j])).T @ (X - new_mus[j])) / max(nk[j], 1e-300) + reg * np.eye(d)
                          for j in range(m)])
        mus = np.where(dead[:, None], mus, new_mus)
        Sigmas = np.where(dead[:, None, None], Sigmas, new_S)
        alphas = np.maximum(nk / nk.sum(), 1e-3)
        alphas /= alphas.sum()
        if t > 0 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-1]):
            break
    return (alphas, mus, Sigmas), np.array(lls), hist


def n_params(m, d):
    return (m - 1) + m * d + m * d * (d + 1) // 2


def ellipse(ax, mu, Sigma, color, k=2.0, **kw):
    vals, vecs = np.linalg.eigh(Sigma)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    e = Ellipse(mu, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang, fill=False, color=color, **kw)
    ax.add_patch(e)


# ---------------------------------------------------------------- exercice 3
def ess(w):
    w = np.asarray(w, float)
    return w.sum() ** 2 / (w**2).sum()


def pmc(logtarget, init, n, n_iter, rng, M=None):
    """Population Monte Carlo : q^(t) mélange gaussien, mise à jour par EM pondéré."""
    alphas, mus, Sigmas = (np.array(a, float).copy() for a in init)
    out = []
    for t in range(n_iter):
        z = sample_discrete(alphas, np.arange(len(alphas)), n, rng)
        X = np.array([rng.multivariate_normal(mus[j], Sigmas[j]) for j in z])
        logq = logsumexp(gmm_logcomp(X, alphas, mus, Sigmas), axis=1)
        logw = logtarget(X) - logq
        w = np.exp(logw - logw.max()); w /= w.sum()
        out.append(dict(alphas=alphas, mus=mus, Sigmas=Sigmas, X=X, w=w, ess=1 / (w**2).sum()))
        (alphas, mus, Sigmas), _, _ = em_gmm(X, len(alphas), rng, init=(alphas, mus, Sigmas), n_iter=50, w=w, reg=1e-6)
    return out
