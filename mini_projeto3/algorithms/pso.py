

import numpy as np
from robot.fanuc import LOWER_BOUNDS, UPPER_BOUNDS


def pso(
    func,
    dim: int = 6,
    lower: np.ndarray = LOWER_BOUNDS,
    upper: np.ndarray = UPPER_BOUNDS,
    n_particles: int = 30,
    max_iter: int = 500,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int | None = None,
) -> dict:
    """
    PSO clássico (gbest) para minimização.

    Parâmetros
    ----------
    func        : callable(q) → float
    dim         : número de variáveis
    lower/upper : limites articulares
    n_particles : tamanho do enxame
    max_iter    : iterações máximas
    w           : inércia
    c1          : coeficiente cognitivo (pbest)
    c2          : coeficiente social (gbest)
    seed        : semente aleatória

    Retorna dict com:
        best_x    – melhor solução
        best_f    – melhor fitness
        history   – evolução do melhor fitness por iteração
        eval_count– total de avaliações da função objetivo
        iterations– total de iterações
    """
    rng = np.random.default_rng(seed)

    # ── Inicialização ──────────────────────────────────────────
    pos = rng.uniform(lower, upper, (n_particles, dim))
    vel = rng.uniform(-(upper - lower), (upper - lower), (n_particles, dim))

    pbest_pos = pos.copy()
    pbest_val = np.array([func(p) for p in pos])
    eval_count = n_particles

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [gbest_val]

    for _ in range(max_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))

        pos = np.clip(pos + vel, lower, upper)

        vals = np.array([func(p) for p in pos])
        eval_count += n_particles

        improved = vals < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = vals[improved]

        new_best_idx = np.argmin(pbest_val)
        if pbest_val[new_best_idx] < gbest_val:
            gbest_val = pbest_val[new_best_idx]
            gbest_pos = pbest_pos[new_best_idx].copy()

        history.append(gbest_val)

    return {
        "best_x":     gbest_pos,
        "best_f":     gbest_val,
        "history":    np.array(history),
        "eval_count": eval_count,
        "iterations": max_iter,
    }
