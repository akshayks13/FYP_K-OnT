"""OnT-style Poincaré disk of the numeric probe.

OnT treats MiniLM output as a point in a Poincaré ball with c = 1 / dim
(see ont/hyperbolic.py). Distances used in training are manifold.dist and
manifold.dist0 — not cosine.

This draws that ball the way HiT/OnT figures do: a disk, origin in the
middle (more general), points farther out are more specific.

  Left:  v = encode(V(a))           — the real OnT space
  Right: z = concat(v, scaled x)    — export vector, drawn the same way
                                     so we can see if numbers move points.
                                     OnT does not train on z.
"""
from __future__ import annotations

import json
import os

import importlib.util
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from geoopt.manifolds import PoincareBall

_PROBE = os.path.join(os.path.dirname(__file__), "probe_numeric.py")
_spec = importlib.util.spec_from_file_location("probe_numeric", _PROBE)
_probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_probe)
PROBE_DIR = _probe.PROBE_DIR
encode_sentences = _probe.encode_sentences
feature_vector = _probe.feature_vector
unit_rows = _probe.unit_rows
verbalize = _probe.verbalize


def to_disk(embeds: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """High-dim Poincaré points → 2-d unit disk (logmap0 at origin, PCA, expmap0)."""
    man = PoincareBall(c=c)
    x = torch.tensor(embeds, dtype=torch.float64)
    # stay inside the ball (radius 1/sqrt(c))
    radius = 1.0 / np.sqrt(c)
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    x = x * torch.clamp((radius - 1e-5) / norms.clamp_min(1e-12), max=1.0)
    tangent = man.logmap0(x).numpy()
    tangent = tangent - tangent.mean(axis=0)
    _, _, vt = np.linalg.svd(tangent, full_matrices=False)
    xy = tangent @ vt[:2].T
    # map 2-d tangent vectors into the unit Poincaré disk (c=1), like paper figures
    disk = PoincareBall(c=1.0)
    t = torch.tensor(xy, dtype=torch.float64)
    # keep them off the boundary so labels stay readable
    max_n = float(torch.linalg.norm(t, dim=-1).max())
    t = t * (1.2 / max(max_n, 1e-8))
    pts = disk.expmap0(t).numpy()
    dist0 = man.dist0(x).numpy()
    n = x.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        dist[i] = man.dist(x[i : i + 1], x).numpy()
    return pts, dist0, dist


def draw_disk(ax, pts, labels, kinds, title):
    circ = plt.Circle((0, 0), 1.0, fill=False, lw=1.4, color="#333333")
    ax.add_patch(circ)
    ax.plot(0, 0, "+", ms=10, color="#333333", zorder=4)
    ax.annotate("origin\n(more general)", (0, 0), textcoords="offset points",
                xytext=(-28, 10), fontsize=7, color="#555555")
    colors = {"class": "#888888", "limgp": "#2f6fed", "liznp": "#c45c26"}
    markers = {"class": "D", "limgp": "o", "liznp": "s"}
    limgp_pts = [(x, y) for (x, y), kind in zip(pts, kinds) if kind == "limgp"]
    piled = (
        len(limgp_pts) >= 2
        and max(np.hypot(a[0] - limgp_pts[0][0], a[1] - limgp_pts[0][1]) for a in limgp_pts) < 0.03
    )
    labeled_limgp = False
    for (x, y), lab, kind in zip(pts, labels, kinds):
        ax.scatter(x, y, s=70, c=colors[kind], marker=markers[kind], zorder=3,
                   edgecolors="white", linewidths=0.4)
        if kind == "limgp" and piled:
            if labeled_limgp:
                continue
            ax.annotate(
                "LiMgP 0.00–8.00\n(all on one point)",
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
            )
            labeled_limgp = True
            continue
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def save_print_figure(probe_dir, pts_v, pts_z, labels, kinds, dist_v, dist_z) -> None:
    """A4-wide vector PDF + 300 dpi PNG for printing (not the Cursor canvas)."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.0, 6.4))
    ax0 = fig.add_axes([0.04, 0.28, 0.44, 0.64])
    ax1 = fig.add_axes([0.52, 0.28, 0.44, 0.64])
    draw_disk(ax0, pts_v, labels, kinds, "(a)  OnT space   v = encode(sentence)")
    # classes have no number list — do not draw them on the concat disk
    mat = [k != "class" for k in kinds]
    draw_disk(
        ax1,
        pts_z[mat],
        [l for l, m in zip(labels, mat) if m],
        [k for k, m in zip(kinds, mat) if m],
        "(b)  Export vector   z = concat(v, numbers)   (materials only)",
    )
    fig.text(
        0.50,
        0.96,
        "Poincaré disk  ·  same geometry OnT uses  ·  origin = more general",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.text(
        0.50,
        0.22,
        "Blue = LiMgP (only band gap changes)    Orange = LiZnP, same numbers    Grey = class names",
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )
    table_ax = fig.add_axes([0.12, 0.04, 0.76, 0.16])
    table_ax.axis("off")
    cell = [
        ["Pair", "OnT distance on v", "Distance on z (concat)"],
        ["LiMgP  0.00 vs 0.01 eV   (similar numbers)", f"{dist_v[0, 1]:.2f}", f"{dist_z[0, 1]:.2f}"],
        ["LiMgP  0.00 vs 8.00 eV   (different numbers)", f"{dist_v[0, 5]:.2f}", f"{dist_z[0, 5]:.2f}"],
        ["LiMgP vs LiZnP   (same numbers, different words)", f"{dist_v[0, 6]:.2f}", f"{dist_z[0, 6]:.2f}"],
    ]
    tab = table_ax.table(cellText=cell[1:], colLabels=cell[0], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 1.35)
    png = os.path.join(probe_dir, "poincare_print.png")
    pdf = os.path.join(probe_dir, "poincare_print.pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def save_embedding_figure(probe_dir, labels, V, Xz, Z, feat_names) -> None:
    """Printable view of the actual vectors: first dims of v, all of x, tail of z."""
    v_show = V[:, :24]
    z_tail = Z[:, -Xz.shape[1] :]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(v_show, aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].set_xlabel("first 24 dims of v  (384 total)")
    axes[0].set_title("(c)  Verbalization embedding  v")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Xz, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
    axes[1].set_yticks(range(len(labels)), labels, fontsize=8)
    short = ["bg", "Eform", "Ehull", "K", "G", "metal", "stable"]
    axes[1].set_xticks(range(len(short)), short, rotation=40, ha="right", fontsize=8)
    axes[1].set_title("(d)  Number list  x  (z-scored)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(z_tail, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[2].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[2].set_xticks(range(len(short)), short, rotation=40, ha="right", fontsize=8)
    axes[2].set_title("(e)  Last 7 dims of z = concat(v, x)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Embeddings for both cases.  v rows look the same when only the gap changes.  "
        "x / z-tail change with the number.",
        fontsize=11,
    )
    png = os.path.join(probe_dir, "embeddings_both_cases.png")
    pdf = os.path.join(probe_dir, "embeddings_both_cases.pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    rows = []
    for i, lab in enumerate(labels):
        rows.append({
            "id": lab,
            "v_dim": int(V.shape[1]),
            "z_dim": int(Z.shape[1]),
            "v_norm": float(np.linalg.norm(V[i])),
            "z_norm": float(np.linalg.norm(Z[i])),
            "v_first8": [round(float(x), 4) for x in V[i, :8]],
            "x_zscored": {n: round(float(Xz[i, j]), 4) for j, n in enumerate(feat_names)},
            "z_last7": [round(float(x), 4) for x in Z[i, -7:]],
        })
    out = os.path.join(probe_dir, "embeddings_preview.json")
    with open(out, "w") as f:
        json.dump({"note": "v = encode(V(a)); z = concat(unit(v), unit(x)). Full v is 384-d.",
                   "rows": rows}, f, indent=2)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {out}")


def main() -> None:
    probe_dir = os.path.abspath(PROBE_DIR)
    with open(os.path.join(probe_dir, "materials.json")) as f:
        payload = json.load(f)
    names = payload["feature_names"]
    rows = payload["materials"]
    # 6 synthetic LiMgP + formula control (skip duplicate reals)
    rows = [r for r in rows if r["kind"] in ("synthetic_band_gap", "formula_control")]
    sentences = [verbalize(r) for r in rows] + ["substance", "crystal"]
    labels = [
        "LiMgP 0.00",
        "LiMgP 0.01",
        "LiMgP 0.35",
        "LiMgP 1.34",
        "LiMgP 3.50",
        "LiMgP 8.00",
        "LiZnP (same numbers)",
        "substance",
        "crystal",
    ]
    kinds = ["limgp"] * 6 + ["liznp", "class", "class"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    V = encode_sentences(sentences, device)
    X = np.array([feature_vector(r, names) for r in rows], dtype=np.float64)
    n_mat = len(rows)
    std = X.std(axis=0)
    mean = X.mean(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    Xz = (X - mean) / std_safe
    Xz[:, std < 1e-12] = 0.0
    # concat only on materials; class names stay text-only in OnT
    Z_mat = np.concatenate([unit_rows(V[:n_mat]), unit_rows(Xz)], axis=1)

    pts_v, d0_v, dist_v = to_disk(V, c=1.0 / V.shape[1])
    pts_z_mat, d0_z, dist_z_mat = to_disk(Z_mat, c=1.0 / Z_mat.shape[1])
    pts_z = np.vstack([pts_z_mat, np.full((2, 2), np.nan)])
    dist_z = np.zeros((len(labels), len(labels)))
    dist_z[:n_mat, :n_mat] = dist_z_mat

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
    draw_disk(
        axes[0],
        pts_v,
        labels,
        kinds,
        "OnT Poincaré disk — verbalization only  v = encode(V(a))",
    )
    mat = [k != "class" for k in kinds]
    draw_disk(
        axes[1],
        pts_z[mat],
        [l for l, m in zip(labels, mat) if m],
        [k for k, m in zip(kinds, mat) if m],
        "Export z = concat(v, x)  — materials only, not OnT training",
    )
    fig.suptitle(
        "How OnT sees points: Poincaré disk, origin = more general. "
        "Blue = same LiMgP, only band gap changes.",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = os.path.join(probe_dir, "poincare_both_cases.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    save_print_figure(probe_dir, pts_v, pts_z, labels, kinds, dist_v, dist_z)
    save_embedding_figure(probe_dir, labels[:n_mat], V[:n_mat], Xz, Z_mat, names)

    i0, i8, izn, isub = 0, 5, 6, 7
    summary = {
        "labels": labels,
        "disk_v": pts_v.tolist(),
        "disk_z": pts_z.tolist(),
        "kinds": kinds,
        "dist0_v": d0_v.tolist(),
        "dist0_z": d0_z.tolist(),
        "hyperbolic_dist_v": {
            "LiMgP_0_vs_8": float(dist_v[i0, i8]),
            "LiMgP_0_vs_0.01": float(dist_v[i0, 1]),
            "LiMgP_vs_LiZnP": float(dist_v[i0, izn]),
            "LiMgP_vs_substance": float(dist_v[i0, isub]),
        },
        "hyperbolic_dist_z": {
            "LiMgP_0_vs_8": float(dist_z[i0, i8]),
            "LiMgP_0_vs_0.01": float(dist_z[i0, 1]),
            "LiMgP_vs_LiZnP": float(dist_z[i0, izn]),
            "LiMgP_vs_substance": None,
        },
        "note": "OnT losses use manifold.dist / dist0 on v only. z is export.",
    }
    out_json = os.path.join(probe_dir, "poincare_xy.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {out_png}")
    print(f"wrote {out_json}")
    print("OnT hyperbolic dist (v):", json.dumps(summary["hyperbolic_dist_v"], indent=2))
    print("export hyperbolic dist (z):", json.dumps(summary["hyperbolic_dist_z"], indent=2))
    print("dist0 v (origin → point):")
    for lab, d in zip(labels, d0_v):
        print(f"  {lab:24s} {d:.4f}")


if __name__ == "__main__":
    main()
