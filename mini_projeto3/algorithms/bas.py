
import numpy as np
from robot.fanuc import repair_solution, LOWER_BOUNDS, UPPER_BOUNDS


def bas(
    func,
    dim: int = 6,
    lower: np.ndarray = LOWER_BOUNDS,
    upper: np.ndarray = UPPER_BOUNDS,
    max_iter: int = 500,
    step0: float = 0.5,
    step_decay: float = 0.95,
    d0: float = 0.1,
    seed: int | None = None,
) -> dict:
    """
    Beetle Antennae Search para minimização.

    Parâmetros
    ----------
    func       : callable(q) → float
    dim        : número de variáveis (6 para o robô)
    lower/upper: limites articulares (arrays dim,)
    max_iter   : iterações máximas
    step0      : passo inicial
    step_decay : fator de decaimento do passo (eta)
    d0         : distância inicial entre antenas
    seed       : semente aleatória

    Retorna dict com:
        best_x    – melhor solução encontrada
        best_f    – melhor fitness
        history   – array (max_iter+1,) com evolução do melhor fitness
        eval_count– número total de avaliações da função objetivo
        iterations– total de iterações
    """
    rng = np.random.default_rng(seed)

    # ── Inicialização ──────────────────────────────────────────
    x = rng.uniform(lower, upper)
    best_x = x.copy()
    best_f = func(x)
    eval_count = 1

    history = [best_f]
    step = step0
    d = d0

    for _ in range(max_iter):
        # Direção aleatória normalizada
        b = rng.standard_normal(dim)
        b /= (np.linalg.norm(b) + 1e-12)

        # Posições das antenas
        x_right = repair_solution(x + d / 2 * b)
        x_left  = repair_solution(x - d / 2 * b)

        # Avaliação
        f_right = func(x_right)
        f_left  = func(x_left)
        eval_count += 2

        # Atualização da posição
        sign = np.sign(f_right - f_left)
        x = repair_solution(x - step * sign * b)

        # Avalia nova posição
        f_current = func(x)
        eval_count += 1

        if f_current < best_f:
            best_f = f_current
            best_x = x.copy()

        # Decaimento
        step *= step_decay
        d = step / 5

        history.append(best_f)

    return {
        "best_x":     best_x,
        "best_f":     best_f,
        "history":    np.array(history),
        "eval_count": eval_count,
        "iterations": max_iter,
    }
