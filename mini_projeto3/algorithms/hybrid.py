
import numpy as np
from algorithms.pso import pso
from algorithms.bas import bas
from robot.fanuc import LOWER_BOUNDS, UPPER_BOUNDS


def pso_bas_hybrid(
    func,
    dim: int = 6,
    lower: np.ndarray = LOWER_BOUNDS,
    upper: np.ndarray = UPPER_BOUNDS,
    total_iter: int = 500,
    fase1_ratio: float = 0.6,
    n_particles: int = 30,
    pso_w: float = 0.7,
    pso_c1: float = 1.5,
    pso_c2: float = 1.5,
    bas_step0: float = 0.2,
    bas_decay: float = 0.97,
    seed: int | None = None,
) -> dict:
    """
    Híbrido PSO → BAS.

    Parâmetros
    ----------
    func         : callable(q) → float
    dim          : número de variáveis
    lower/upper  : limites articulares
    total_iter   : orçamento total de iterações
    fase1_ratio  : fração usada pelo PSO (0 < fase1_ratio < 1)
    n_particles  : partículas do PSO
    pso_*        : hiperparâmetros do PSO
    bas_*        : hiperparâmetros do BAS (refinamento)
    seed         : semente aleatória

    Retorna dict com:
        best_x       – melhor solução final
        best_f       – melhor fitness final
        history      – histórico concatenado (PSO + BAS)
        eval_count   – total de avaliações da função objetivo
        pso_iters    – iterações usadas pelo PSO
        bas_iters    – iterações usadas pelo BAS
        pso_best_f   – melhor fitness ao final do PSO
        iterations   – total de iterações
    """
    pso_iters = int(total_iter * fase1_ratio)
    bas_iters = total_iter - pso_iters

    # ── Fase 1: PSO (exploração global) ───────────────────────
    pso_result = pso(
        func,
        dim=dim,
        lower=lower,
        upper=upper,
        n_particles=n_particles,
        max_iter=pso_iters,
        w=pso_w,
        c1=pso_c1,
        c2=pso_c2,
        seed=seed,
    )

    pso_best_x = pso_result["best_x"]
    pso_best_f = pso_result["best_f"]
    eval_count = pso_result["eval_count"]

    # ── Fase 2: BAS (refinamento local) ───────────────────────
    # O BAS parte do melhor ponto do PSO, com passo menor
    bas_result = bas(
        func,
        dim=dim,
        lower=lower,
        upper=upper,
        max_iter=bas_iters,
        step0=bas_step0,
        step_decay=bas_decay,
        seed=seed,
    )

    # O BAS começa de posição aleatória internamente, mas vamos
    # usar o melhor do PSO como ponto de partida via wrapper
    bas_result_seeded = _bas_from_init(
        func=func,
        x_init=pso_best_x,
        dim=dim,
        lower=lower,
        upper=upper,
        max_iter=bas_iters,
        step0=bas_step0,
        step_decay=bas_decay,
        seed=seed,
    )

    eval_count += bas_result_seeded["eval_count"]

    # ── Histórico unificado ────────────────────────────────────
    # Garante continuidade: o BAS começa do melhor do PSO
    bas_history = bas_result_seeded["history"]
    bas_history = np.minimum.accumulate(
        np.concatenate([[pso_best_f], bas_history])
    )

    history = np.concatenate([pso_result["history"], bas_history[1:]])

    final_best_f = bas_result_seeded["best_f"]
    final_best_x = bas_result_seeded["best_x"]

    # Compara com melhor do PSO (caso BAS não melhore)
    if pso_best_f < final_best_f:
        final_best_f = pso_best_f
        final_best_x = pso_best_x

    return {
        "best_x":     final_best_x,
        "best_f":     final_best_f,
        "history":    history,
        "eval_count": eval_count,
        "pso_iters":  pso_iters,
        "bas_iters":  bas_iters,
        "pso_best_f": pso_best_f,
        "iterations": total_iter,
    }


def _bas_from_init(
    func,
    x_init: np.ndarray,
    dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int,
    step0: float,
    step_decay: float,
    seed: int | None,
) -> dict:
    """BAS com ponto de partida definido externamente."""
    rng = np.random.default_rng(seed)

    x = np.clip(x_init.copy(), lower, upper)
    best_x = x.copy()
    best_f = func(x)
    eval_count = 1

    history = [best_f]
    step = step0
    d = step / 5

    for _ in range(max_iter):
        b = rng.standard_normal(dim)
        b /= (np.linalg.norm(b) + 1e-12)

        x_right = np.clip(x + d / 2 * b, lower, upper)
        x_left  = np.clip(x - d / 2 * b, lower, upper)

        f_right = func(x_right)
        f_left  = func(x_left)
        eval_count += 2

        sign = np.sign(f_right - f_left)
        x = np.clip(x - step * sign * b, lower, upper)

        f_current = func(x)
        eval_count += 1

        if f_current < best_f:
            best_f = f_current
            best_x = x.copy()

        step *= step_decay
        d = step / 5
        history.append(best_f)

    return {
        "best_x":     best_x,
        "best_f":     best_f,
        "history":    np.array(history),
        "eval_count": eval_count,
    }
