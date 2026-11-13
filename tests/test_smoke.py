"""A few training epochs for every example and method, and the save / skip / resume / summarize pipeline."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from qpinn import RunConfig, run_one
from qpinn.settings import PAPER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = [(e, m, c) for (e, m) in PAPER for c in (["smooth", "singular"] if e == "6.4" else [""])]


def _cfg(tmp_path, tiny, ex, m, case="", **kw):
    base = dict(example=ex, method=m, case=case, seed=1, epochs=3, lbfgs_iters=2, val_every=1, log_every=100,
                n_qubits=3, n_layers=1, problem_kw=tiny[ex], device="cpu", out=str(tmp_path),
                save_plots=False, verbose=False)
    base.update(kw)
    return RunConfig(**base)


@pytest.mark.parametrize("ex,m,case", SETTINGS)
def test_train_a_few_epochs(tmp_path, tiny, ex, m, case):
    cfg = _cfg(tmp_path, tiny, ex, m, case)
    out = run_one(cfg)
    assert out["status"] == "ok" and np.isfinite(out["relL2"])
    for f in ("config.json", "history.csv", "metrics.json", "model.pt", "predictions.npz"):
        assert os.path.exists(os.path.join(cfg.run_dir, f)), f


def test_same_seed_same_result(tmp_path, tiny):
    a = run_one(_cfg(tmp_path / "a", tiny, "6.3", "A"))
    b = run_one(_cfg(tmp_path / "b", tiny, "6.3", "A"))
    assert a["relL2"] == b["relL2"]


def test_finished_runs_are_skipped(tmp_path, tiny):
    cfg = _cfg(tmp_path, tiny, "6.1", "A")
    first = run_one(cfg)
    mtime = os.path.getmtime(os.path.join(cfg.run_dir, "metrics.json"))
    again = run_one(cfg)
    assert os.path.getmtime(os.path.join(cfg.run_dir, "metrics.json")) == mtime
    assert again["relL2"] == first["relL2"]
    assert len(pd.read_csv(os.path.join(tmp_path, "index.csv"))) == 1


def test_resume_gives_the_uninterrupted_result(tmp_path, tiny, monkeypatch):
    """A run stopped after 4 of 8 epochs and resumed from checkpoint.pt ends where an uninterrupted run ends."""
    import torch
    full = run_one(_cfg(tmp_path / "full", tiny, "6.2", "N", epochs=8, lbfgs_iters=0, lr_steps=[[5, 0.1]]))
    part = _cfg(tmp_path / "part", tiny, "6.2", "N", epochs=8, lbfgs_iters=0, lr_steps=[[5, 0.1]], ckpt_every=4)
    orig_step, calls = torch.optim.Adam.step, {"n": 0}

    def interrupted_step(self, *a, **k):          # Ctrl-C during the 5th epoch
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return orig_step(self, *a, **k)
    monkeypatch.setattr(torch.optim.Adam, "step", interrupted_step)
    with pytest.raises(KeyboardInterrupt):
        run_one(part)
    monkeypatch.setattr(torch.optim.Adam, "step", orig_step)
    assert os.path.exists(os.path.join(part.run_dir, "checkpoint.pt"))
    resumed = run_one(part)
    assert abs(resumed["relL2"] - full["relL2"]) < 1e-6
    assert not os.path.exists(os.path.join(part.run_dir, "checkpoint.pt"))


def test_plots_are_written(tmp_path, tiny):
    for ex, m in (("6.1", "N"), ("6.4", "N")):
        cfg = _cfg(tmp_path, tiny, ex, m, "smooth" if ex == "6.4" else "", save_plots=True)
        run_one(cfg)
        assert os.path.exists(os.path.join(cfg.run_dir, "curves.png"))
        assert os.path.exists(os.path.join(cfg.run_dir, "solution.png"))


def test_cli_and_summary(tmp_path):
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--all", "--dry-run"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.count("params=") == 8 * 5              # 8 settings x 5 seeds
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--example", "6.2", "--method", "N",
                        "--seeds", "1", "2", "--epochs", "3", "--lbfgs", "0", "--n-qubits", "2", "--n-layers", "1",
                        "--problem-kw", "n_colloc=8", "Q=4", "n_val=20", "--device", "cpu", "--no-plots",
                        "--out", str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([sys.executable, os.path.join(ROOT, "summarize.py"), "--out", str(tmp_path)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    s = pd.read_csv(tmp_path / "summary.csv")
    assert len(s) == 1 and s.loc[0, "seeds"] == 2 and "relL2_v_mean" in s


def test_64_rejects_A(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--example", "6.4", "--method", "A",
                        "--dry-run"], capture_output=True, text=True)
    assert r.returncode != 0
