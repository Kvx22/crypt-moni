"""The extractor model: a small linear text classifier trained in numpy.

A story becomes a sparse bag of hashed words and word pairs (title words count double), plus
a few rows that say what the rule labeller (labels.rule_label) thought: its catalyst, direction
and flags. A few hundred short headlines are too few to learn 23 catalysts from words alone,
so the model starts from the rules and learns, from your corrections, where they go wrong.
Three heads read the same features:
  catalyst    softmax over taxonomy.CATALYSTS
  direction   softmax over -1 / 0 / +1
  risk flags  one yes/no (sigmoid) per taxonomy.RISK_FLAGS key

It is trained with Adagrad on mini-batches, touching only the weight rows a batch uses, so
20k examples train in seconds. Hashing uses zlib.crc32, never Python's salted hash(), so a
saved model reads the same features in every process. Weights are kept at float16
precision, which makes a saved and reloaded model predict exactly like the original.

The service trains a challenger on SEED + your corrections + confident rule labels and
promotes it only when challenger_wins() says it beats the champion on held-out labels.
"""
from __future__ import annotations

import io
import json
import math
import zlib

import numpy as np

from .labels import rule_label
from .taxonomy import CATALYSTS, DIRECTIONS, RISK_FLAGS

try:
    from .text import tokens as _tokens
except ImportError:   # text.py not there yet: a plain lowercase word splitter
    import re
    _TOK = re.compile(r"\$?[^\W_]+(?:['.][^\W_]+)*%?")

    def _tokens(text):
        return [t for t in (m.lower() for m in _TOK.findall(text or "")) if len(t) > 1 or t.isdigit()]

FLAGS = tuple(RISK_FLAGS)
_NC, _ND, _NF = len(CATALYSTS), len(DIRECTIONS), len(FLAGS)
_K = _NC + _ND + _NF          # all heads side by side in one weight matrix
_R = _K                       # dedicated rule rows after the hashed rows: one per label value
MODEL_FORMAT = 1


def _h(s, n):
    return zlib.crc32(s.encode("utf-8")) % n


def _str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return x if isinstance(x, str) else ""


def features(title, text="", n_features=2 ** 15):
    """(idx int32, val float32): hashed unigrams + bigrams, title x2, log1p counts, L2-normalised."""
    counts = {}
    for part, w in ((_str(title), 2.0), (_str(text)[:4000], 1.0)):
        toks = _tokens(part)
        for i, t in enumerate(toks):
            k = _h(t, n_features)
            counts[k] = counts.get(k, 0.0) + w
            if i:
                k = _h(toks[i - 1] + " " + t, n_features)
                counts[k] = counts.get(k, 0.0) + w
    if not counts:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    idx = np.fromiter(sorted(counts), np.int32, len(counts))
    val = np.log1p(np.array([counts[i] for i in idx], np.float32))
    val /= np.float32(np.sqrt((val * val).sum()))
    return idx, val


def rule_features(title, text="", offset=0):
    """(idx, val) for the rule labeller's verdict: catalyst (valued at its confidence),
    direction and each flag, in rows offset .. offset + _R."""
    r = rule_label(title, text)
    idx = [CATALYSTS.index(r["catalyst"]), _NC + DIRECTIONS.index(r["direction"])]
    val = [r["confidence"], 1.0]
    for f in r["risk_flags"]:
        idx.append(_NC + _ND + FLAGS.index(f))
        val.append(1.0)
    return np.array(idx, np.int32) + np.int32(offset), np.array(val, np.float32)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class TextModel:
    """Catalyst, direction and risk-flag heads over hashed text features (+ the rules' verdict
    when rules=True)."""

    def __init__(self, n_features=2 ** 15, l2=1e-5, rules=True):
        self.n_features, self.l2, self.rules = int(n_features), float(l2), bool(rules)
        self.W = np.zeros((self.n_features + (_R if self.rules else 0), _K), np.float32)
        self.b = np.zeros(_K, np.float32)
        self.n_train, self.version = 0, ""

    def _feats(self, title, text):
        idx, val = features(title, text, self.n_features)
        if not self.rules:
            return idx, val
        ri, rv = rule_features(title, text, self.n_features)
        return np.concatenate([idx, ri]), np.concatenate([val, rv])

    # ---- training ---------------------------------------------------------------------------
    def _encode(self, examples):
        X, Y, wts = [], [], []
        for e in examples:
            cat = e.get("catalyst")
            if cat not in CATALYSTS:
                continue
            try:
                d = int(e.get("direction") or 0)
            except (TypeError, ValueError):
                d = 0
            y = np.zeros(_K, np.float32)
            y[CATALYSTS.index(cat)] = 1.0
            y[_NC + DIRECTIONS.index(max(-1, min(1, d)))] = 1.0
            for f in e.get("risk_flags") or ():
                if f in RISK_FLAGS:
                    y[_NC + _ND + FLAGS.index(f)] = 1.0
            X.append(self._feats(e.get("title"), e.get("text") or ""))
            Y.append(y)
            try:
                wts.append(max(0.0, float(e.get("weight", 1.0))))
            except (TypeError, ValueError):
                wts.append(1.0)
        return X, np.array(Y, np.float32).reshape(-1, _K), np.array(wts, np.float32)

    def _logits(self, batch):
        """Rows of logits for a list of (idx, val) pairs, plus the flattened sparse batch."""
        lens = [len(i) for i, _ in batch]
        idx = np.concatenate([i for i, _ in batch]) if batch else np.zeros(0, np.int32)
        val = np.concatenate([v for _, v in batch]) if batch else np.zeros(0, np.float32)
        row = np.repeat(np.arange(len(batch)), lens)
        Z = np.tile(self.b, (len(batch), 1))
        if len(idx):
            np.add.at(Z, row, self.W[idx] * val[:, None])
        return Z, idx, val, row

    def fit(self, examples, epochs=8, lr=0.5, seed=0, batch_size=16):
        """Train from scratch on dicts(title, text, catalyst, direction, risk_flags, weight=1)."""
        X, Y, wts = self._encode(list(examples or ()))
        self.W[:] = 0
        self.b[:] = 0
        self.n_train = len(X)
        if not X:
            self.version = "empty"
            return self
        G = np.zeros_like(self.W)       # Adagrad accumulators
        Gb = np.zeros_like(self.b)
        rng = np.random.default_rng(seed)
        wmean = float(wts.mean()) or 1.0
        for _ in range(int(epochs)):
            order = rng.permutation(len(X))
            for s in range(0, len(order), batch_size):
                ids = order[s:s + batch_size]
                Z, idx, val, row = self._logits([X[i] for i in ids])
                P = np.empty_like(Z)
                P[:, :_NC] = _softmax(Z[:, :_NC])
                P[:, _NC:_NC + _ND] = _softmax(Z[:, _NC:_NC + _ND])
                P[:, _NC + _ND:] = _sigmoid(Z[:, _NC + _ND:])
                E = (P - Y[ids]) * (wts[ids] / wmean)[:, None]    # dLoss/dlogit for every head
                if not len(idx):
                    continue
                uniq, inv = np.unique(idx, return_inverse=True)
                g = np.zeros((len(uniq), _K), np.float32)
                np.add.at(g, inv, E[row] * val[:, None])
                g = g / len(ids) + self.l2 * self.W[uniq]
                G[uniq] += g * g
                self.W[uniq] -= lr * g / (np.sqrt(G[uniq]) + 1e-8)
                gb = E.mean(axis=0)
                Gb += gb * gb
                self.b -= lr * gb / (np.sqrt(Gb) + 1e-8)
        # keep weights at the precision they are saved in: a reloaded model predicts identically
        self.W = self.W.astype(np.float16).astype(np.float32)
        self.b = self.b.astype(np.float16).astype(np.float32)
        self.version = f"{zlib.crc32(self.W.tobytes()) ^ zlib.crc32(self.b.tobytes()):08x}"
        return self

    # ---- prediction -------------------------------------------------------------------------
    def predict_many(self, pairs):
        """[(title, text)] -> [prediction dict] (see predict)."""
        batch = [self._feats(t, x) for t, x in pairs]
        if not batch:
            return []
        Z = self._logits(batch)[0]
        pc, pd, pf = _softmax(Z[:, :_NC]), _softmax(Z[:, _NC:_NC + _ND]), _sigmoid(Z[:, _NC + _ND:])
        out = []
        for i in range(len(batch)):
            c, d = int(pc[i].argmax()), int(pd[i].argmax())
            fp = {f: round(float(pf[i, j]), 4) for j, f in enumerate(FLAGS)}
            out.append(dict(catalyst=CATALYSTS[c], cat_p=float(pc[i, c]), direction=int(DIRECTIONS[d]), dir_p=float(pd[i, d]),
                            risk_flags=[f for f in FLAGS if fp[f] >= 0.5], flag_p=fp))
        return out

    def predict(self, title, text=""):
        """dict(catalyst, cat_p, direction, dir_p, risk_flags (p >= 0.5), flag_p {flag: p})."""
        return self.predict_many([(title, text)])[0]

    # ---- saving -----------------------------------------------------------------------------
    def to_blob(self) -> bytes:
        meta = dict(format=MODEL_FORMAT, n_features=self.n_features, l2=self.l2, rules=self.rules, n_train=self.n_train, version=self.version,
                    catalysts=list(CATALYSTS), directions=list(DIRECTIONS), flags=list(FLAGS))
        buf = io.BytesIO()
        np.savez_compressed(buf, W=self.W.astype(np.float16), b=self.b.astype(np.float16),
                            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8))
        return buf.getvalue()

    @classmethod
    def from_blob(cls, blob):
        with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes().decode("utf-8"))
            if (meta.get("catalysts") != list(CATALYSTS) or meta.get("flags") != list(FLAGS)
                    or meta.get("directions") != list(DIRECTIONS)):
                raise ValueError("model was trained on a different label set")
            m = cls(meta["n_features"], meta.get("l2", 1e-5), meta.get("rules", True))
            W, b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        if W.shape != m.W.shape or b.shape != m.b.shape:
            raise ValueError("model weights have the wrong shape")
        m.W, m.b = W, b
        m.n_train, m.version = int(meta.get("n_train", 0)), str(meta.get("version", ""))
        return m


# ---- evaluation and champion/challenger ------------------------------------------------------
def _macro_f1(truth, pred):
    f1s = []
    for c in sorted(set(truth) | set(pred)):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(model, examples) -> dict:
    """dict(n, cat_acc, cat_macro_f1, dir_acc, flag_f1) on labelled examples.
    flag_f1 is micro-F1 over all flags (1.0 when there were none to find and none were claimed)."""
    ex = [e for e in examples or () if e.get("catalyst") in CATALYSTS]
    if not ex or model is None:
        return dict(n=len(ex), cat_acc=0.0, cat_macro_f1=0.0, dir_acc=0.0, flag_f1=0.0)
    preds = model.predict_many([(e.get("title"), e.get("text") or "") for e in ex])
    tc, pc = [e["catalyst"] for e in ex], [p["catalyst"] for p in preds]
    dir_ok = sum(1 for e, p in zip(ex, preds) if int(e.get("direction") or 0) == p["direction"])
    tp = fp = fn = 0
    for e, p in zip(ex, preds):
        t, q = set(e.get("risk_flags") or ()), set(p["risk_flags"])
        tp, fp, fn = tp + len(t & q), fp + len(q - t), fn + len(t - q)
    n = len(ex)
    return dict(n=n, cat_acc=sum(a == b for a, b in zip(tc, pc)) / n, cat_macro_f1=_macro_f1(tc, pc), dir_acc=dir_ok / n,
                flag_f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)


def split_by_time(examples, holdout=0.25):
    """(train, holdout): the newest `holdout` share (by "ts") is held out."""
    ex = sorted(examples or (), key=lambda e: float(e.get("ts") or 0.0))
    k = int(math.floor(len(ex) * holdout + 0.5)) if ex else 0
    k = min(len(ex), max(0, k))
    return ex[:len(ex) - k], ex[len(ex) - k:]


def challenger_wins(champ_metrics, chall_metrics, min_n=40, margin=0.01):
    """(wins, reason). With no champion any trained challenger wins. Otherwise it needs at least
    min_n held-out examples, catalyst macro-F1 up by >= margin, and direction accuracy not
    down by more than margin."""
    ch = chall_metrics or {}
    n = int(ch.get("n") or 0)
    if not champ_metrics:
        return (n > 0, "first extractor model" if n > 0 else "no held-out examples to judge the model")
    if n < min_n:
        return False, f"only {n} held-out examples (need {min_n})"
    f_new, f_old = float(ch.get("cat_macro_f1") or 0), float(champ_metrics.get("cat_macro_f1") or 0)
    d_new, d_old = float(ch.get("dir_acc") or 0), float(champ_metrics.get("dir_acc") or 0)
    if d_new < d_old - margin:
        return False, f"direction accuracy fell from {d_old:.0%} to {d_new:.0%}"
    if f_new < f_old + margin:
        return False, f"catalyst F1 {f_new:.3f} is not {margin:.2f} better than {f_old:.3f}"
    return True, f"catalyst F1 up from {f_old:.3f} to {f_new:.3f}"
