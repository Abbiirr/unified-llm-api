"""
router_policy.py — runtime scorer for the learned auto-router (v1).

Loads models/router_v1.json (an empirical, hierarchically-smoothed utility
table) and, given a request's features + the set of currently-healthy aliases,
returns the best alias to route to. Pure stdlib, <0.2 ms per decision, fails
safe: any load/scoring error -> not ready -> caller uses the heuristic.

Design references:
  - RouteLLM win-rate routing      https://www.lmsys.org/blog/2024-07-01-routellm/
  - OpenRouter Auto Router dial     https://openrouter.ai/docs/guides/routing/routers/auto-router
  - bandit-feedback caveat          https://arxiv.org/abs/2510.07429
"""

from __future__ import annotations

import json
import math
import os
import random

import router_features as rf

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "router_v1.json")

# Aliases safe for very large context (>=12k tok): 128k–1M models AND covered by
# smart_router's CTX_RESCUE. Small-context pools (coding/fast/tools/tools_stable)
# are excluded for xl requests to avoid ContextWindowExceeded with no rescue.
LARGE_CONTEXT_SAFE = frozenset({"tools_large", "swebench", "big", "bench", "default"})


class Decision:
    __slots__ = ("alias", "confidence", "utility", "ranked", "cost_bias", "reason")

    def __init__(self, alias, confidence, utility, ranked, cost_bias, reason):
        self.alias = alias
        self.confidence = confidence
        self.utility = utility
        self.ranked = ranked
        self.cost_bias = cost_bias
        self.reason = reason

    def as_dict(self):
        return {
            "alias": self.alias,
            "confidence": self.confidence,
            "utility": round(self.utility, 4) if self.utility is not None else None,
            "cost_bias": self.cost_bias,
            "reason": self.reason,
            "ranked": [
                {"alias": a, "util": round(u, 4), "p_success": round(p, 3), "support": s}
                for (a, u, p, s) in self.ranked[:5]
            ],
        }


class RouterPolicy:
    def __init__(self, model: dict):
        self.model = model
        self.table = model.get("table", {})
        self.pool = model.get("auto_pool", {})
        w = model.get("weights", {})
        self.w_success = w.get("w_success", 1.0)
        self.w_loglat = w.get("w_loglat", 0.22)
        self.w_rl = w.get("w_rate_limit", 0.30)
        self.prior = w.get("prior_strength", 25.0)
        self.cost_bias_default = w.get("cost_bias_default", 5)
        self.min_confidence = w.get("min_confidence", 40)
        # heuristic-as-prior: override the heuristic's structural pick only when
        # the learned winner is both well-supported AND clearly better. Bounds
        # variance from sparse cells and the bandit-feedback failure mode.
        self.override_margin = w.get("override_margin", 0.05)
        # load-spreading exploration: softmax temperature over candidates within
        # explore_margin of the best utility (0 temp = deterministic argmax).
        self.explore_temp = w.get("explore_temp", 0.0)
        self.explore_margin = w.get("explore_margin", 0.12)
        self.loglat_lo = model.get("loglat_lo", math.log1p(200))
        self.loglat_hi = model.get("loglat_hi", math.log1p(90000))
        self.global_p = model.get("global_success", 0.6)
        self.global_loglat = model.get("global_loglat", self.loglat_lo)

    # ── loading ──
    @classmethod
    def load(cls, path: str | None = None):
        path = path or DEFAULT_MODEL_PATH
        try:
            with open(path, encoding="utf-8") as f:
                model = json.load(f)
            if not model.get("table"):
                return None
            return cls(model)
        except Exception:
            return None

    def is_ready(self) -> bool:
        return bool(self.table)

    # ── per-alias smoothed estimates with hierarchical back-off ──
    def _shrink_success(self, keys, alias):
        p, support = self.global_p, 0
        for key in reversed(keys):  # general -> specific
            cell = self.table.get(key, {}).get(alias)
            if not cell:
                continue
            n, ok = cell[0], cell[1]
            p = (ok + self.prior * p) / (n + self.prior)
            support = n
        return p, support

    def _shrink_loglat(self, keys, alias):
        v = self.global_loglat
        for key in reversed(keys):
            cell = self.table.get(key, {}).get(alias)
            if not cell:
                continue
            lat_n, loglat_sum = cell[3], cell[4]
            if lat_n <= 0:
                continue
            v = (loglat_sum + self.prior * self.global_loglat) / (lat_n + self.prior)
        return v

    def _shrink_rate_limit(self, keys, alias):
        rl = 1.0 - self.global_p
        for key in reversed(keys):
            cell = self.table.get(key, {}).get(alias)
            if not cell:
                continue
            n, rate_limited = cell[0], cell[2]
            rl = (rate_limited + self.prior * (1.0 - self.global_p)) / (n + self.prior)
        return rl

    # ── main entry ──
    def candidate_aliases(self, feat: dict) -> list[str]:
        mod = rf.modality(feat)
        pool = self.pool.get(mod) or []
        # Large-context gating applies to text/tools only. Image requests carry
        # a base64-inflated token estimate, and vision/ocr manage their own
        # context — never strip them. Never let the filter empty the pool.
        if mod != "image" and rf.requires_large_context(feat):
            filtered = [a for a in pool if a in LARGE_CONTEXT_SAFE]
            if filtered:
                pool = filtered
        return pool

    def decide(self, feat: dict, healthy: set | None = None, cost_bias: int | None = None,
               heuristic_alias: str | None = None, explore_temp: float | None = None,
               health_penalty: dict | None = None) -> Decision:
        cb = self.cost_bias_default if cost_bias is None else max(0, min(10, int(cost_bias)))
        temp = self.explore_temp if explore_temp is None else explore_temp
        hp = health_penalty or {}
        keys = rf.bucket_keys(feat)
        pool = self.candidate_aliases(feat)
        # Always consider the heuristic's pick too (it may sit outside the curated
        # pool, e.g. an image->ocr or a size-rewritten alias) so we can compare.
        if heuristic_alias and heuristic_alias not in pool:
            pool = pool + [heuristic_alias]
        if healthy is not None:
            pool = [a for a in pool if a in healthy]
        if not pool:
            return Decision(None, 0, None, [], cb, "no_healthy_candidate")

        # cost dial: higher cost_bias -> stronger latency penalty (prefer cheap/fast)
        eff_w_loglat = self.w_loglat * (cb / 5.0)

        ranked = []
        for a in pool:
            p, support = self._shrink_success(keys, a)
            loglat = self._shrink_loglat(keys, a)
            rl = self._shrink_rate_limit(keys, a)
            norm_lat = max(0.0, min(1.0, (loglat - self.loglat_lo) / (self.loglat_hi - self.loglat_lo)))
            # live alias-health penalty: route around aliases that are failing or
            # hanging right now (e.g. a provider went down) — real-time feedback.
            util = (self.w_success * p) - (eff_w_loglat * norm_lat) - (self.w_rl * rl) - hp.get(a, 0.0)
            ranked.append((a, util, p, support))
        ranked.sort(key=lambda x: x[1], reverse=True)
        best = ranked[0]

        # ── heuristic-as-prior arbitration ──
        # Determine the base pick + whether we trust the learned ranking enough to
        # diversify among near-best (we never diversify when deferring to the
        # structural heuristic).
        chosen, reason, trust_learned = best, "learned", True
        if heuristic_alias:
            h = next((r for r in ranked if r[0] == heuristic_alias), None)
            if h is not None and best[0] != heuristic_alias:
                margin = best[1] - h[1]
                # When the heuristic's own pick is currently degraded (its provider
                # is hanging), relax the support gate so we decisively route around
                # it to the best healthy alias instead of staying on a bad route.
                heuristic_degraded = hp.get(heuristic_alias, 0.0) >= 0.2
                confident = best[3] >= self.min_confidence or heuristic_degraded
                if confident and margin >= self.override_margin:
                    tag = "reroute_degraded" if heuristic_degraded and best[3] < self.min_confidence else "learned_override"
                    chosen, reason = best, f"{tag}(+{margin:.3f} vs {heuristic_alias})"
                else:
                    chosen, reason, trust_learned = h, "heuristic_kept", False
            elif h is not None:
                chosen, reason = h, "agree"
        elif best[3] < self.min_confidence:
            reason = "learned_low_support"

        # ── load-spreading exploration ──
        # Sample among candidates within `explore_margin` of the best utility,
        # weighted by softmax(util / temp). Spreads provider load (different
        # aliases have different latency-winning providers), yields counterfactual
        # outcomes for the feedback loop, and barely dents quality since every
        # near-best alias is high-utility. Off when temp<=0 or when we kept the
        # heuristic. https://arxiv.org/abs/2510.07429
        if temp and temp > 0 and trust_learned:
            near = [r for r in ranked if r[1] >= best[1] - self.explore_margin]
            if len(near) >= 2:
                mx = max(r[1] for r in near)
                weights = [math.exp((r[1] - mx) / temp) for r in near]
                total = sum(weights)
                pick, acc, thresh = near[-1], 0.0, random.random() * total
                for r, wgt in zip(near, weights):
                    acc += wgt
                    if thresh <= acc:
                        pick = r
                        break
                if pick[0] != chosen[0]:
                    return Decision(pick[0], pick[3], pick[1], ranked, cb, reason + "+explore")

        return Decision(chosen[0], chosen[3], chosen[1], ranked, cb, reason)


# Process-wide singleton, lazily (re)loadable so the gateway can hot-reload the
# policy after a nightly rebuild without a restart.
_policy: RouterPolicy | None = None


def get_policy(reload: bool = False) -> RouterPolicy | None:
    global _policy
    if _policy is None or reload:
        _policy = RouterPolicy.load()
    return _policy


if __name__ == "__main__":
    pol = get_policy()
    if not pol:
        print("no model loaded")
        raise SystemExit(1)
    tests = [
        {"name": "short hi", "feat": {"user_text_length": 2, "estimated_total_tokens": 5}, "h": "fast"},
        {"name": "python code", "feat": {"estimated_total_tokens": 120, "has_code_blocks": True, "primary_language": "python"}, "h": "coding"},
        {"name": "tools req", "feat": {"has_tools": True, "tool_count": 2, "estimated_total_tokens": 900}, "h": "tools"},
        {"name": "image", "feat": {"has_images": True, "estimated_total_tokens": 1500}, "h": "vision"},
        {"name": "huge ctx", "feat": {"estimated_total_tokens": 40000}, "h": "big"},
        {"name": "medium plain", "feat": {"estimated_total_tokens": 1200}, "h": "thinking"},
    ]
    for t in tests:
        d = pol.decide(t["feat"], heuristic_alias=t["h"])
        print(f'{t["name"]:14s} heur={t["h"]:9s} -> {str(d.alias):12s} conf={d.confidence} '
              f'util={(d.utility or 0):.3f} reason={d.reason}')
