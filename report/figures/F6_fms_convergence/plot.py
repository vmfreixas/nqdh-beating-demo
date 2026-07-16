"""Figure 6 (fig:fmsconv): FMS basis-size convergence of the vibronic beat.

Panel a: velocity-flavor ensemble S2(t) resolved by spawned-basis-size
quintile (small / medium / large shown) against the 300-IC Ehrenfest
reference -- the largest-basis quintile recovers the mean-field first
recurrence. Panel b: first-recurrence amplitude (S2 peak in 17-23 fs minus
the curve's own minimum in 10-16 fs) vs mean basis size per quintile, both
FMS flavors, with the Ehrenfest amplitude as reference.

Self-contained: reads only data/ in this folder; writes figure.png/.pdf and
CSV exports for external replotting.

data/fms_pertraj_s2.npz    completed FMS runs only (salvaged runs carry no
    time_fs                 spawn count): {mode}_s2 (n_ic, nt) S2 per run,
    {mode}_s2               {mode}_n_spawns (n_ic,), {mode}_ic_indices
data/ehrenfest_mean.npz    time_fs, mean_pops (401, 3), 300 ICs
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "data", "fms_pertraj_s2.npz"))
e = np.load(os.path.join(HERE, "data", "ehrenfest_mean.npz"))

t = d["time_fs"]
te, pe = e["time_fs"], e["mean_pops"]
me = te <= float(t.max())

NQ = 5


def quintiles(ns):
    edges = np.percentile(ns, np.linspace(0, 100, NQ + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return [(ns > edges[i]) & (ns <= edges[i + 1]) for i in range(NQ)]


def bump(y, tv=None):
    tv = t if tv is None else tv
    wmin = (tv >= 10) & (tv <= 16)          # per-curve pre-recurrence minimum
    wpk = (tv >= 17) & (tv <= 23)
    return y[wpk].max() - y[wmin].min()


# Ehrenfest reference amplitude on its own grid
ehr_bump = bump(pe[:, 2], te)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.8, 6.6), height_ratios=[1.5, 1.0]
)

# --- panel a: velocity S2(t) by basis-size quintile (Q1/Q3/Q5) ---
s2, ns = d["velocity_s2"], d["velocity_n_spawns"]
qm = quintiles(ns)
shades = {0: "#c7d9e8", 2: "#7fa8c9", 4: "#c0392b"}
names = {0: "small basis (Q1", 2: "medium basis (Q3", 4: "large basis (Q5"}
ax1.plot(te[me], pe[me, 2], color="#2471a3", lw=2.2, label="Ehrenfest (300 ICs)")
for qi, col in shades.items():
    m = qm[qi]
    mean = s2[m].mean(0)
    sem = s2[m].std(0) / np.sqrt(m.sum())
    ax1.plot(t, mean, color=col, lw=2.0,
             label=f"FMS {names[qi]}, ~{ns[m].mean():.0f} spawns, {m.sum()} ICs)")
    ax1.fill_between(t, mean - sem, mean + sem, color=col, alpha=0.3, lw=0)
ax1.set_ylabel(r"mean $S_2$ population")
ax1.set_xlabel("time (fs)")
ax1.set_xlim(0, float(t.max()))
ax1.set_ylim(0, 1.0)
ax1.legend(frameon=False, fontsize=9)
ax1.set_title("velocity-flavor FMS approaches the mean-field beat as the basis grows",
              fontsize=10.5)

# --- panel b: first-recurrence amplitude vs basis size, both flavors ---
for mode, col, mk in (("velocity", "#c0392b", "o"), ("coupling", "#2e8b57", "s")):
    s2m, nsm = d[f"{mode}_s2"], d[f"{mode}_n_spawns"]
    xs, ys, es = [], [], []
    for m in quintiles(nsm):
        bumps = np.array([bump(y) for y in s2m[m]])
        xs.append(nsm[m].mean()); ys.append(bumps.mean())
        es.append(bumps.std() / np.sqrt(m.sum()))
    ax2.errorbar(xs, ys, yerr=es, color=col, marker=mk, lw=1.8, capsize=3,
                 label=f"FMS {mode} rescale")
ax2.axhline(ehr_bump, color="#2471a3", ls="--", lw=1.5,
            label=f"Ehrenfest ({ehr_bump:+.2f})")
ax2.axhline(0.0, color="gray", lw=0.8)
ax2.set_xlabel("mean spawned basis functions per trajectory (quintile)")
ax2.set_ylabel("first-recurrence\namplitude")
ax2.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "figure.png"), dpi=200)
fig.savefig(os.path.join(HERE, "figure.pdf"))
print("wrote figure.png / figure.pdf")

# --- CSV exports for external replotting ---
for mode in ("velocity", "coupling"):
    s2m, nsm = d[f"{mode}_s2"], d[f"{mode}_n_spawns"]
    qms = quintiles(nsm)
    with open(os.path.join(HERE, "data", f"quintile_curves_{mode}.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_fs"] + [f"Q{i+1}_{s}" for i in range(NQ) for s in ("mean", "sem")])
        cols = []
        for m in qms:
            cols += [s2m[m].mean(0), s2m[m].std(0) / np.sqrt(m.sum())]
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}"] + [f"{c[i]:.5f}" for c in cols])
    with open(os.path.join(HERE, "data", f"bump_scatter_{mode}.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_spawns", "bump", "ic_index"])
        for j in range(len(nsm)):
            w.writerow([int(nsm[j]), f"{bump(s2m[j]):.5f}", int(d[f'{mode}_ic_indices'][j])])

with open(os.path.join(HERE, "data", "bump_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "quintile", "mean_spawns", "n_ics", "bump_mean", "bump_sem"])
    for mode in ("velocity", "coupling"):
        s2m, nsm = d[f"{mode}_s2"], d[f"{mode}_n_spawns"]
        for i, m in enumerate(quintiles(nsm)):
            bumps = np.array([bump(y) for y in s2m[m]])
            w.writerow([mode, i + 1, f"{nsm[m].mean():.1f}", int(m.sum()),
                        f"{bumps.mean():.5f}", f"{bumps.std()/np.sqrt(m.sum()):.5f}"])
    w.writerow(["ehrenfest_reference", "", "", 300, f"{ehr_bump:.5f}", ""])
print("wrote quintile_curves_*.csv / bump_scatter_*.csv / bump_table.csv")
