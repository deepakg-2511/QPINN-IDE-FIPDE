"""Train N-QPINN / A-QPINN on the examples of Section 6 with the settings of the paper (Table 7).

Examples
--------
# Example 6.1, A-QPINN, the five seeds of the paper
python run.py --example 6.1 --method A

# every example and method of Tables 1-4 (25 settings x 5 seeds; long, see README)
python run.py --all

# Example 6.4, weakly singular case, one seed
python run.py --example 6.4 --method N --case singular --seeds 1234

# quick check that everything runs (few epochs, small grids; results are not meaningful)
python run.py --example 6.3 --method A --seeds 1 --epochs 20 --lbfgs 0 --problem-kw nf=200 --out scratch

# list the runs and their sizes without training
python run.py --all --dry-run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qpinn import RunConfig, build, run_one                 # noqa: E402
from qpinn.qnn import n_params                               # noqa: E402
from qpinn.settings import CASES, COMMON, METHODS, PAPER, SEEDS   # noqa: E402


def parse_kv(items):
    """['nf=200', 'Q=10'] -> {'nf': 200, 'Q': 10}"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def settings_list(a):
    """(example, method, case) triples selected on the command line."""
    if a.all:
        return [(e, m, c) for (e, m) in PAPER for c in CASES.get(e, [""])]
    if not a.example:
        raise SystemExit("give --example (and --method) or --all")
    methods = [a.method] if a.method else [m for (e, m) in PAPER if e == a.example]
    cases = [a.case] if a.case else CASES.get(a.example, [""])
    out = []
    for m in methods:
        if (a.example, m) not in PAPER:
            raise SystemExit(f"Example {a.example} has no {METHODS[m]} (only N-QPINN applies to 6.4)")
        out += [(a.example, m, c) for c in cases]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", choices=sorted({e for e, _ in PAPER}))
    ap.add_argument("--method", choices=["A", "N"], help="A = A-QPINN, N = N-QPINN (default: both)")
    ap.add_argument("--case", choices=["smooth", "singular"], help="Example 6.4 only (default: both)")
    ap.add_argument("--all", action="store_true", help="every setting of Tables 1-4")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=COMMON["epochs"])
    ap.add_argument("--lbfgs", type=int, default=COMMON["lbfgs_iters"], help="L-BFGS iterations after Adam")
    ap.add_argument("--lr", type=float, help="initial learning rate (default: paper value)")
    ap.add_argument("--n-qubits", type=int, default=0, help="default: paper value")
    ap.add_argument("--n-layers", type=int, default=0, help="default: paper value")
    ap.add_argument("--problem-kw", nargs="*", help="problem options, e.g. n_colloc=50 Q=20 nf=5000")
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the runs and parameter counts only")
    a = ap.parse_args(argv)
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    pkw = parse_kv(a.problem_kw)
    # the paper's schedule is defined for 20 000 epochs; scale it if fewer epochs are requested
    scale = a.epochs / COMMON["epochs"]
    steps = [[max(1, int(e * scale)), f] for e, f in COMMON["lr_steps"]]
    for ex, m, case in settings_list(a):
        ps = PAPER[(ex, m)]
        for seed in a.seeds:
            cfg = RunConfig(example=ex, method=m, case=case, seed=seed, epochs=a.epochs,
                            lr=a.lr if a.lr else ps["lr"], lr_steps=steps, lbfgs_iters=a.lbfgs,
                            n_qubits=a.n_qubits or ps["n_qubits"], n_layers=a.n_layers or ps["n_layers"],
                            problem_kw=pkw, device=a.device, out=a.out, save_plots=not a.no_plots,
                            overwrite=a.overwrite)
            if a.dry_run:
                _, net, size = build(cfg, device="cpu")
                print(f"Example {ex}  {METHODS[m]:8s} {case:9s} seed={seed}  qubits={size['n_qubits']} "
                      f"layers={size['n_layers']}  params={n_params(net)}  lr={cfg.lr:g}  -> {cfg.run_dir}")
                continue
            run_one(cfg)


if __name__ == "__main__":
    main()
