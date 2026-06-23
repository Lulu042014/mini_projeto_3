"""
experiments/runner.py
=====================
Executa os experimentos comparativos entre GA, PSO, BAS e Híbrido.
Função objetivo: erro de posição + erro de orientação (Fanuc 200i).
"""

import time
import numpy as np
from functools import partial

from robot.fanuc import inverse_kinematics_error, random_target
from algorithms.bas import bas
from algorithms.pso import pso
from algorithms.ga import ga
from algorithms.hybrid import pso_bas_hybrid

DEFAULT_CONFIG = {
    "max_iter":    500,
    "n_runs":      20,
    "n_particles": 30,
    "pop_size":    50,
}


def _run_single(algo_fn, func, seed, **kwargs):
    t0 = time.perf_counter()
    result = algo_fn(func, seed=seed, **kwargs)
    elapsed = time.perf_counter() - t0
    return {
        "best_f":     result["best_f"],
        "best_x":     result["best_x"],
        "history":    result["history"],
        "eval_count": result["eval_count"],
        "time":       elapsed,
    }


def run_experiments(
    target_pos=None,
    target_rot=None,
    n_runs=DEFAULT_CONFIG["n_runs"],
    max_iter=DEFAULT_CONFIG["max_iter"],
    verbose=True,
):
    # ── Alvo ──────────────────────────────────────────────────
    if target_pos is None or target_rot is None:
        target_pos, target_rot = random_target(seed=42)

    func = partial(
        inverse_kinematics_error,
        target_pos=target_pos,
        target_rot=target_rot,
    )

    if verbose:
        print(f"\n  Posição alvo: x={target_pos[0]:.4f}m, "
              f"y={target_pos[1]:.4f}m, z={target_pos[2]:.4f}m")
        print(f"  Objetivo: erro posição + 0.3 × erro orientação")
        print(f"  Execuções: {n_runs} | Iterações: {max_iter}\n")

    algorithms = {
        "BAS": (bas, dict(max_iter=max_iter)),
        "PSO": (pso, dict(max_iter=max_iter,
                          n_particles=DEFAULT_CONFIG["n_particles"])),
        "GA":  (ga,  dict(max_gen=max_iter,
                          pop_size=DEFAULT_CONFIG["pop_size"])),
        "Híbrido (PSO→BAS)": (pso_bas_hybrid,
                               dict(total_iter=max_iter,
                                    n_particles=DEFAULT_CONFIG["n_particles"])),
    }

    results = {}
    for name, (algo_fn, kwargs) in algorithms.items():
        if verbose:
            print(f"  Executando {name}...", end=" ", flush=True)

        runs = [_run_single(algo_fn, func, seed=s, **kwargs) for s in range(n_runs)]

        vals  = [r["best_f"]     for r in runs]
        times = [r["time"]       for r in runs]
        evals = [r["eval_count"] for r in runs]
        hists = [r["history"]    for r in runs]

        results[name] = {
            "vals": vals, "times": times,
            "evals": evals, "histories": hists,
            "summary": {
                "best":      float(np.min(vals)),
                "worst":     float(np.max(vals)),
                "mean":      float(np.mean(vals)),
                "std":       float(np.std(vals)),
                "time_mean": float(np.mean(times)),
                "eval_mean": float(np.mean(evals)),
            },
        }

        if verbose:
            s = results[name]["summary"]
            print(f"melhor={s['best']:.4e}  média={s['mean']:.4e}  "
                  f"dp={s['std']:.4e}  t={s['time_mean']*1000:.0f}ms  "
                  f"aval≈{s['eval_mean']:.0f}")

    results["_meta"] = {
        "target_pos": target_pos,
        "target_rot": target_rot,
        "n_runs": n_runs,
        "max_iter": max_iter,
    }
    return results


def print_summary_table(results):
    meta = results.get("_meta", {})
    print("\n" + "=" * 80)
    print("  TABELA COMPARATIVA – Cinemática Inversa Fanuc 200i")
    print(f"  Runs: {meta.get('n_runs','?')}  |  Iter: {meta.get('max_iter','?')}")
    print("=" * 80)
    header = (f"  {'Algoritmo':<22} {'Melhor':>12} {'Pior':>12} "
              f"{'Média':>12} {'DP':>12} {'Tempo(ms)':>10} {'Aval.':>8}")
    print(header)
    print("  " + "-" * 90)
    for name, data in results.items():
        if name.startswith("_"):
            continue
        s = data["summary"]
        print(f"  {name:<22} {s['best']:>12.4e} {s['worst']:>12.4e} "
              f"{s['mean']:>12.4e} {s['std']:>12.4e} "
              f"{s['time_mean']*1000:>10.1f} {s['eval_mean']:>8.0f}")
    print("=" * 80)
