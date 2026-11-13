"""Training and result saving for one (example, method, seed) run.

Every run writes to   <out>/<example>/<A-QPINN|N-QPINN>[_<case>]/seed_<seed>/
    config.json       the full configuration (QNN size, problem options, parameter count)
    history.csv       loss components every epoch; validation errors every `val_every` epochs
    metrics.json      final and best relative L2 errors, parameters, seconds per epoch
    model.pt          state_dict of the trained QNN
    predictions.npz   validation coordinates, prediction and exact solution
    curves.png, solution.png
    checkpoint.pt     only while a run is unfinished; the same command resumes it
and appends one row to <out>/index.csv. A finished run (metrics.json present) is skipped unless
overwrite=True.
"""
import copy
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import torch

from .problems import get_problem
from .qnn import QNN, PhysicsNet, n_params
from .settings import METHODS
from .utils import ensure_dir, get_device, set_seed


@dataclass
class RunConfig:
    example: str                                    # "6.1", "6.2", "6.3" or "6.4"
    method: str                                     # "A" (A-QPINN) or "N" (N-QPINN)
    case: str = ""                                  # Example 6.4 only: "smooth" or "singular"
    seed: int = 1234
    epochs: int = 20000
    lr: float = 1e-2
    lr_steps: list = field(default_factory=list)   # [[epoch, factor], ...]: lr = base * factor from that epoch on
    lbfgs_iters: int = 0                            # L-BFGS refinement after Adam
    n_qubits: int = 0                               # 0: the example's default (Table 7)
    n_layers: int = 0                               # 0: the example's default (Table 7)
    problem_kw: dict = field(default_factory=dict)  # problem options, e.g. n_colloc, Q, n_grid
    val_every: int = 100
    log_every: int = 500
    ckpt_every: int = 500                           # write checkpoint.pt + partial history every N epochs (0 = off)
    device: str = "auto"
    out: str = "results"
    save_model: bool = True
    save_plots: bool = True
    overwrite: bool = False
    verbose: bool = True

    @property
    def run_dir(self):
        name = METHODS[self.method] + (f"_{self.case}" if self.case else "")
        return os.path.join(self.out, self.example, name, f"seed_{self.seed}")


def _lr_lambda(steps):
    steps = sorted(steps)

    def f(ep):
        factor = 1.0
        for e, fac in steps:
            if ep >= e:
                factor = fac
        return factor
    return f


def _append_index(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def build(cfg: RunConfig, device=None):
    """Problem, QNN (wrapped with input scaling) and the resolved QNN size."""
    device = device or get_device(cfg.device)
    pkw = dict(cfg.problem_kw)
    if cfg.case:
        pkw["case"] = cfg.case
    P = get_problem(cfg.example, **pkw)
    if cfg.method not in P.forms:
        raise ValueError(f"Example {cfg.example} supports {[METHODS[f] for f in P.forms]} only")
    size = dict(P.default_qnn[cfg.method])
    if cfg.n_qubits:
        size["n_qubits"] = cfg.n_qubits
    if cfg.n_layers:
        size["n_layers"] = cfg.n_layers
    model = QNN(len(P.coords), P.n_out[cfg.method], **size)
    net = PhysicsNet(model, P.lo, P.hi).to(device)
    return P, net, size


def run_one(cfg: RunConfig):
    rd = cfg.run_dir
    mpath = os.path.join(rd, "metrics.json")
    if os.path.exists(mpath) and not cfg.overwrite:
        if cfg.verbose:
            print(f"[skip] {rd} (done)")
        with open(mpath) as fh:
            return json.load(fh)
    ensure_dir(rd)
    device = get_device(cfg.device)
    set_seed(cfg.seed)
    P, net, size = build(cfg, device)
    pts = P.sample(cfg.method, device)
    n_par = n_params(net)
    with open(os.path.join(rd, "config.json"), "w") as fh:
        json.dump({**asdict(cfg), "qnn": size, "problem_opts": P.opt, "n_params": n_par, "device": str(device)},
                  fh, indent=2, default=str)
    if cfg.verbose:
        print(f"[run] Example {cfg.example} {METHODS[cfg.method]}{' (' + cfg.case + ')' if cfg.case else ''} "
              f"seed={cfg.seed} qubits={size['n_qubits']} layers={size['n_layers']} params={n_par} device={device}")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg.lr_steps)) if cfg.lr_steps else None
    hist, best, ep_times, status = [], (float("inf"), -1), [], "ok"
    comp_names, val_names = None, None
    t_start = time.perf_counter()
    start_ep, t_offset = 0, 0.0
    ckpt_path = os.path.join(rd, "checkpoint.pt")
    if os.path.exists(ckpt_path) and not cfg.overwrite:            # resume an interrupted run
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        hist, best, start_ep, t_offset = ck["hist"], tuple(ck["best"]), ck["epoch"] + 1, ck["time_s"]
        comp_names, val_names = ck["comp_names"], ck["val_names"]
        torch.set_rng_state(ck["rng_torch"].cpu().to(torch.uint8)); np.random.set_state(ck["rng_np"])
        if cfg.verbose:
            print(f"  [resume] from epoch {start_ep} (checkpoint.pt)")

    def write_history():
        cols = ["epoch", "loss"] + (comp_names or []) + ["lr", "time_s"] + (val_names or [])
        tmp = os.path.join(rd, "history.csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(hist)
        os.replace(tmp, os.path.join(rd, "history.csv"))

    def save_checkpoint(ep):
        ck = {"model": net.model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict() if sched else None, "epoch": ep, "hist": hist, "best": list(best),
              "time_s": t_offset + time.perf_counter() - t_start, "comp_names": comp_names,
              "val_names": val_names, "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state()}
        torch.save(ck, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        write_history()

    def validate(ep, row):
        nonlocal best, val_names
        m, _ = P.evaluate(net, device)
        val_names = val_names or list(m)
        row.update(m)
        if m["relL2"] < best[0]:
            best = (m["relL2"], ep)
        return m

    # ---------------------------------------------------------------- Adam
    ep = done_ep = start_ep - 1
    try:
        for ep in range(start_ep, cfg.epochs):
            t0 = time.perf_counter()
            opt.zero_grad()
            loss, comps = P.loss(net, pts, cfg.method)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            ep_times.append(time.perf_counter() - t0)
            comp_names = comp_names or list(comps)
            row = {"epoch": ep, "loss": loss.item(), **{k: v.item() for k, v in comps.items()},
                   "lr": opt.param_groups[0]["lr"], "time_s": t_offset + time.perf_counter() - t_start}
            if not np.isfinite(row["loss"]):
                status = f"diverged at epoch {ep}"
                hist.append(row)
                break
            if ep % cfg.val_every == 0 or ep == cfg.epochs - 1:
                m = validate(ep, row)
                if cfg.verbose and (ep % cfg.log_every == 0 or ep == cfg.epochs - 1):
                    print(f"  ep {ep:6d}  loss {row['loss']:.3e}  relL2 {m['relL2']:.3e}  {row['time_s']:.0f}s")
            hist.append(row)
            done_ep = ep
            if cfg.ckpt_every and (ep + 1) % cfg.ckpt_every == 0:
                save_checkpoint(ep)
    except KeyboardInterrupt:
        if done_ep >= start_ep:
            save_checkpoint(done_ep)
        print(f"\n  [interrupted] saved checkpoint.pt after epoch {done_ep} -> {rd}\n"
              f"  run the same command again to resume.")
        raise

    # ---------------------------------------------------------------- L-BFGS refinement
    if cfg.lbfgs_iters > 0 and status == "ok":
        before_state = copy.deepcopy(net.state_dict())
        before_loss = P.loss(net, pts, cfg.method)[0].item()
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters, history_size=50,
                               tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l, _ = P.loss(net, pts, cfg.method)
            l.backward()
            return l
        try:
            lb.step(closure)
            after_loss = P.loss(net, pts, cfg.method)[0].item()
        except RuntimeError as exc:
            after_loss = float("nan")
            print(f"  [lbfgs] failed ({exc}); keeping the Adam result")
        if not np.isfinite(after_loss) or after_loss > before_loss:
            net.load_state_dict(before_state)             # L-BFGS may never make the result worse
            if cfg.verbose:
                print(f"  [lbfgs] loss {before_loss:.3e} -> {after_loss:.3e}: kept the Adam weights")
        elif cfg.verbose:
            print(f"  [lbfgs] loss {before_loss:.3e} -> {after_loss:.3e}")
        loss, comps = P.loss(net, pts, cfg.method)
        row = {"epoch": cfg.epochs, "loss": loss.item(), **{k: v.item() for k, v in comps.items()},
               "lr": 1.0, "time_s": t_offset + time.perf_counter() - t_start}
        validate(cfg.epochs, row)
        hist.append(row)

    # ---------------------------------------------------------------- save
    total = t_offset + time.perf_counter() - t_start
    final, arrays = P.evaluate(net, device)
    steady = ep_times[5:] if len(ep_times) > 10 else ep_times
    metrics = {"example": cfg.example, "method": METHODS[cfg.method], "case": cfg.case, "seed": cfg.seed,
               "n_params": n_par, "n_qubits": size["n_qubits"], "n_layers": size["n_layers"],
               "epochs": cfg.epochs, "lbfgs_iters": cfg.lbfgs_iters, "status": status, **final,
               "best_relL2": best[0], "best_epoch": best[1], "final_loss": hist[-1]["loss"],
               "sec_per_epoch": float(np.mean(steady)) if steady else float("nan"), "total_sec": total,
               "run_dir": rd, "finished": datetime.now().isoformat(timespec="seconds")}
    write_history()
    np.savez_compressed(os.path.join(rd, "predictions.npz"), **arrays)
    if cfg.save_model:
        torch.save(net.model.state_dict(), os.path.join(rd, "model.pt"))
    if cfg.save_plots:
        try:
            from .plotting import plot_run
            plot_run(rd, P, arrays)
        except Exception as exc:          # plotting must never end a long run
            print(f"  [plot skipped: {exc}]")
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)
    _append_index(os.path.join(cfg.out, "index.csv"), {k: metrics[k] for k in
                  ["example", "method", "case", "seed", "n_params", "n_qubits", "n_layers", "status", "relL2",
                   "best_relL2", "final_loss", "sec_per_epoch", "total_sec", "run_dir", "finished"]})
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if cfg.verbose:
        print(f"[done] relL2={final['relL2']:.3e}  {metrics['sec_per_epoch'] * 1e3:.1f} ms/epoch  -> {rd}")
    return metrics
