import numpy as np

from .common import angle_deg, unit


def kabsch(P, U, w=None):
    P = np.asarray(P, float)
    U = np.asarray(U, float)
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    H = (P * w[:, None]).T @ U
    Us, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ Us.T))
    return Vt.T @ np.diag([1., 1., d]) @ Us.T


def geodesic_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def _undefined(v):
    return not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9


def reorient(anchor_a, anchor_b, goal_a, goal_b):
    a = np.asarray(anchor_a, float)
    b = np.asarray(anchor_b, float)
    if _undefined(a) or _undefined(b) or np.array_equal(a, b):
        return None
    return kabsch(unit(np.array([a, b])), unit(np.array([goal_a, goal_b], float)))


def true_direction(shape, pid, pose):
    c = np.array(next(p["c"] for p in shape["parts"] if p["pid"] == pid))
    return c if pose == "canon" else np.array(shape["Q"]).T @ c


def score_reorientation(task, shape, pred_a, pred_b):
    da = true_direction(shape, task["A"], task["pose"])
    db = true_direction(shape, task["B"], task["pose"])
    pa = np.asarray(pred_a, float)
    pb = np.asarray(pred_b, float)
    if _undefined(pa) or _undefined(pb) or np.array_equal(pa, pb):
        return 180.0, 180.0
    Rs = np.array(task["Rstar"])
    ua, ub = Rs @ unit(da), Rs @ unit(db)
    Rh = kabsch(unit(np.array([pa, pb])), np.array([ua, ub]))
    joint = max(angle_deg(Rh @ da, ua), angle_deg(Rh @ db, ub))
    return joint, geodesic_deg(Rh, Rs)


def pointing_error(pred, true):
    pred = np.asarray(pred, float)
    if np.linalg.norm(pred) < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(pred @ true / np.linalg.norm(pred) / np.linalg.norm(true), -1, 1))))
