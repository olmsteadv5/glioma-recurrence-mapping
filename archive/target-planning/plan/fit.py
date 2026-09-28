# Fitting the three biological parameters, by Gaussian process over a small box.

import functools
import math

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc

from plan import baseline, barriers, conduits, planeval, speed, target

# the only three parameters that are neither cited nor set by the machine
NAMES = ("wm_gain", "tract_gain", "risk_gain")
BOX = {"wm_gain": (1.0, 10.0), "tract_gain": (1.0, 6.0), "risk_gain": (0.0, 4.0)}
DEFAULTS = {"wm_gain": speed.WM_GAIN, "tract_gain": speed.TRACT_GAIN,
            "risk_gain": speed.RISK_GAIN}

# a share of what the target could reach, not of the clinical plan. Both give
# every subject its own scale, but only this one holds the question fixed: the
# score is only sensitive to the parameters when the target has to reject
# volume, and a share of the plan rejects a different fraction on every subject
# because plans run 265 to 709 mL against a reach that tracks the tumor instead.
# 0.55 leaves 45% of the reachable region on the table, matching the corridor null.
BUDGET = 0.55

# a median over fewer than this many subjects says more about which ones
# survived than about the parameters
FLOOR_N = 3

SEED = 20250925


def _point(params):
    if isinstance(params, dict):
        return tuple(float(params[n]) for n in NAMES)
    return tuple(float(v) for v in params)


def _named(vector):
    return dict(zip(NAMES, (float(v) for v in vector)))


# ---- the objective -------------------------------------------------------

@functools.lru_cache(maxsize=64)
def budget_ml(first, share=BUDGET):
    return share * target.reachable(first)


def rejected(pairs, share=BUDGET):
    # the diagnostic the budget change exists for: every subject should be
    # asked to reject the same fraction, and the plan share should be the thing
    # that varies rather than the question
    rows = []
    for first, _ in pairs:
        reach, plan = target.reachable(first), baseline.volume(first, baseline.rtog(first))
        want = share * reach
        rows.append({"subject": first[-5:], "reach_ml": round(reach, 1),
                     "plan_ml": round(plan, 1), "budget_ml": round(want, 1),
                     "rejects": round(1.0 - want / reach, 3),
                     "of_plan": round(want / plan, 3)})
    return rows


@functools.lru_cache(maxsize=256)
def _sphere(first, later, want_ml):
    # the comparator never sees the three parameters, so it is paid for once
    mask, _ = baseline.isotropic(first, want_ml)
    return planeval.coverage(later, mask)["coverage"]


def warm(pairs):
    # none of these depend on the parameters; warming them once per subject
    # takes most of the cost out of the search
    for first, _ in pairs:
        barriers.masks(first)
        conduits.corridors(first, speed.TUBE_MM)
        conduits.disruption(first)


@functools.lru_cache(maxsize=8192)
def _delta(first, later, point, share):
    want = budget_ml(first, share)
    mask, diag = target.build(first, want, **_named(point))
    if diag["capped"]:
        return None
    ours = planeval.coverage(later, mask)["coverage"]
    return ours - _sphere(first, later, want)


def deltas(params, pairs, budget=BUDGET):
    # per subject, so a median can be read against the spread behind it
    point = _point(params)
    return [(f[-5:], _delta(f, l, point, budget)) for f, l in pairs]


def evaluate(params, pairs, budget=BUDGET):
    # median gain over the volume-matched sphere. A target that could not spend
    # its budget is not comparable to one that could, so that subject is dropped
    kept = [d for _, d in deltas(params, pairs, budget) if d is not None]
    if len(kept) < min(FLOOR_N, len(pairs)):
        return float("nan")
    return float(np.median(kept))


# ---- the surrogate -------------------------------------------------------

def _matern(a, b, scale):
    # Matern 5/2: twice differentiable, no smoother than this surface looks
    gap = (a[:, None, :] - b[None, :, :]) / scale
    root = math.sqrt(5.0) * np.sqrt(np.maximum((gap ** 2).sum(-1), 0.0))
    return (1.0 + root + root ** 2 / 3.0) * np.exp(-root)


def _covariance(theta, x, k):
    amp, noise = math.exp(theta[-2]), math.exp(theta[-1])
    return amp * _matern(x, x, np.exp(theta[:k])) + (noise + 1e-8) * np.eye(len(x))


def _nll(theta, x, y):
    # the negative log marginal likelihood, up to the constant term
    try:
        lower = cho_factor(_covariance(theta, x, x.shape[1]), lower=True)
    except LinAlgError:
        return 1e6
    return float(0.5 * y @ cho_solve(lower, y) + np.log(np.diag(lower[0])).sum())


def _learn(x, y, seed=SEED):
    # lengthscales and noise by maximizing the log marginal likelihood
    rng = np.random.default_rng(seed)
    k = x.shape[1]
    lo = np.array([math.log(0.05)] * k + [math.log(1e-2), math.log(1e-6)])
    hi = np.array([math.log(5.0)] * k + [math.log(1e2), math.log(1.0)])
    starts = [np.array([math.log(0.5)] * k + [0.0, math.log(1e-3)])]
    starts += [rng.uniform(lo, hi) for _ in range(4)]

    top, theta = None, starts[0]
    for start in starts:
        out = minimize(_nll, start, args=(x, y), method="L-BFGS-B",
                       bounds=list(zip(lo, hi)))
        if np.isfinite(out.fun) and (top is None or out.fun < top):
            top, theta = float(out.fun), out.x
    return theta


def _posterior(theta, x, y, probe):
    k = x.shape[1]
    amp = math.exp(theta[-2])
    lower = cho_factor(_covariance(theta, x, k), lower=True)
    cross = amp * _matern(probe, x, np.exp(theta[:k]))
    mean = cross @ cho_solve(lower, y)
    var = amp - np.einsum("ij,ji->i", cross, cho_solve(lower, cross.T))
    return mean, np.sqrt(np.maximum(var, 1e-12))


def _improvement(mean, sd, best, xi=0.01):
    gain = mean - best - xi
    z = gain / sd
    return gain * ndtr(z) + sd * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)


def _edges(bounds):
    lo = np.array([bounds[n][0] for n in NAMES], dtype=float)
    return lo, np.array([bounds[n][1] for n in NAMES], dtype=float)


def suggest(observed, bounds=BOX, seed=SEED):
    # the next point to try, by expected improvement over the best score so far
    lo, hi = _edges(bounds)
    rng = np.random.default_rng(seed + len(observed))
    seen = [(p, s) for p, s in observed if np.isfinite(s)]
    if len(seen) < 3:
        return _named(rng.uniform(lo, hi))

    x = (np.array([_point(p) for p, _ in seen]) - lo) / (hi - lo)
    raw = np.array([float(s) for _, s in seen])
    spread = float(raw.std())
    y = (raw - raw.mean()) / spread if spread > 1e-12 else raw - raw.mean()
    theta = _learn(x, y, seed)
    best = float(y.max())

    # a quasi-random net over the box, plus a ring around what is already good
    net = qmc.Sobol(len(NAMES), scramble=True, seed=rng).random(2048)
    near = np.clip(x + rng.normal(0.0, 0.05, x.shape), 0.0, 1.0)
    probe = np.vstack([net, near])
    mean, sd = _posterior(theta, x, y, probe)
    pick = probe[int(np.argmax(_improvement(mean, sd, best)))]

    # the net is coarse even in three dimensions, so polish the winner
    def cost(q):
        m, s = _posterior(theta, x, y, np.atleast_2d(q))
        return -float(_improvement(m, s, best)[0])

    out = minimize(cost, pick, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(NAMES))
    if np.isfinite(out.fun) and out.fun < cost(pick):
        pick = np.clip(out.x, 0.0, 1.0)

    # expected improvement collapses on a point already taken; nudge off it
    if float(np.abs(x - pick).sum(1).min()) < 1e-3:
        pick = np.clip(pick + rng.normal(0.0, 0.05, pick.shape), 0.0, 1.0)
    return _named(lo + pick * (hi - lo))


def _feed(trace):
    # a NaN is still information: it marks where the budget could not be spent.
    # The trace keeps it as a NaN; the surrogate is handed a pessimistic stand-in
    # so the acquisition stops walking back into that corner
    live = [r["score"] for r in trace if np.isfinite(r["score"])]
    if not live:
        return [(r["params"], r["score"]) for r in trace]
    floor = min(live) - ((max(live) - min(live)) or 1.0)
    return [(r["params"], r["score"] if np.isfinite(r["score"]) else floor)
            for r in trace]


def _search(score, bounds=BOX, calls=40, start=10, seed=SEED):
    lo, hi = _edges(bounds)
    net = qmc.LatinHypercube(len(NAMES), seed=seed).random(min(start, calls))
    trace = [{"params": _named(lo + row * (hi - lo)), "step": "design"}
             for row in net]
    for row in trace:
        row["score"] = score(row["params"])
    while len(trace) < calls:
        point = suggest(_feed(trace), bounds, seed)
        trace.append({"params": point, "step": "ei", "score": score(point)})
    return trace


def best(trace):
    live = [r for r in trace if np.isfinite(r["score"])]
    return max(live, key=lambda r: r["score"]) if live else None


def optimize(pairs, budget=BUDGET, calls=40, bounds=BOX, start=10, seed=SEED):
    warm(pairs)
    return _search(lambda p: evaluate(p, pairs, budget), bounds, calls, start, seed)


def loo(pairs, budget=BUDGET, calls=30, bounds=BOX, start=8, seed=SEED):
    # fit on n-1 and score the one left out. Three parameters against five
    # subjects will memorize them, so the fitted score is not the number
    warm(pairs)
    pairs = list(pairs)
    rows = []
    for i, held in enumerate(pairs):
        rest = pairs[:i] + pairs[i + 1:]
        # one seed for every fold, so the folds share their design points and
        # only diverge where the scores do
        trace = _search(lambda p: evaluate(p, rest, budget), bounds, calls,
                        start, seed)
        top = best(trace)
        if top is None:
            rows.append({"held": held[0][-5:], "params": None,
                         "fitted": float("nan"), "held_out": float("nan")})
            continue
        rows.append({"held": held[0][-5:], "params": top["params"],
                     "fitted": top["score"],
                     "held_out": evaluate(top["params"], [held], budget)})
    return rows


def sweep(param, values, pairs, budget=BUDGET, **fixed):
    # one parameter at a time. If nothing moves here, no optimizer will find
    # anything, and that is the more useful finding
    warm(pairs)
    base = dict(DEFAULTS, **fixed)
    rows = []
    for value in values:
        point = dict(base, **{param: float(value)})
        got = deltas(point, pairs, budget)
        rows.append({param: float(value),
                     "score": evaluate(point, pairs, budget),
                     "kept": sum(d is not None for _, d in got),
                     "per_subject": got})
    return rows


# ---- a known answer, to check the optimizer before trusting it -----------

def bowl(params):
    # a Branin rescaled onto the box, plus a quadratic third axis. The maximum
    # is -0.3979, at three points, with the third coordinate at 2
    a, b, c = _point(params)
    x = -5.0 + 15.0 * (a - 1.0) / 9.0
    y = 15.0 * (b - 1.0) / 5.0
    z = (c - 2.0) / 2.0
    curve = 5.1 / (4 * math.pi ** 2)
    slope = 5.0 / math.pi
    wave = 1.0 / (8 * math.pi)
    return -((y - curve * x ** 2 + slope * x - 6.0) ** 2
             + 10.0 * (1.0 - wave) * math.cos(x) + 10.0 + 5.0 * z ** 2)


def check(calls=40, bounds=BOX, start=10, seed=SEED):
    trace = _search(bowl, bounds, calls, start, seed)
    return trace, best(trace)
