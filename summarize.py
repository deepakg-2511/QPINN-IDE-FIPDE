"""Mean and standard deviation of the relative L2 error over seeds, per example and method.

    python summarize.py                 # reads results/, prints the table, writes results/summary.csv
    python summarize.py --out scratch
"""
import argparse
import glob
import json
import os

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.out, "*", "*", "seed_*", "metrics.json"))]
    if not rows:
        raise SystemExit(f"no finished runs under {a.out}/")
    df = pd.DataFrame(rows)
    df["case"] = df["case"].fillna("")
    err_cols = [c for c in ["relL2_u", "relL2_v"] if c in df]
    agg = {"seeds": ("seed", "count"), "params": ("n_params", "first"), "qubits": ("n_qubits", "first"),
           "layers": ("n_layers", "first"), "ms_per_epoch": ("sec_per_epoch", lambda s: 1e3 * s.mean())}
    for c in err_cols:
        agg[f"{c}_mean"] = (c, "mean")
        agg[f"{c}_std"] = (c, "std")
    s = df.groupby(["example", "method", "case"], dropna=False).agg(**agg).reset_index()
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    print(f"{'Example':8s} {'Method':9s} {'Case':9s} {'qubits x layers':>15s} {'params':>6s} {'seeds':>5s}  "
          f"relative L2 error (mean +- std)")
    for _, r in s.iterrows():
        errs = []
        for c in err_cols:
            if pd.notna(r[f"{c}_mean"]):
                tag = f"{c[-1]}: " if r["example"] == "6.2" else ""
                std = r[f"{c}_std"] if pd.notna(r[f"{c}_std"]) else 0.0
                errs.append(f"{tag}{r[f'{c}_mean']:.2e} +- {std:.2e}")
        print(f"{r['example']:8s} {r['method']:9s} {r['case']:9s} {r['qubits']:>7d} x {r['layers']:<5d} "
              f"{r['params']:>6d} {r['seeds']:>5d}  " + ";  ".join(errs))
    print(f"\nwritten: {os.path.join(a.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
