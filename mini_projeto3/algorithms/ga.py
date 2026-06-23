

import numpy as np
from robot.fanuc import LOWER_BOUNDS, UPPER_BOUNDS


def ga(
    func,
    dim: int = 6,
    lower: np.ndarray = LOWER_BOUNDS,
    upper: np.ndarray = UPPER_BOUNDS,
    pop_size: int = 50,
    max_gen: int = 500,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_sigma: float = 0.2,
    tournament_size: int = 3,
    alpha_blx: float = 0.5,
    seed: int | None = None,
) -> dict:
    """
    Algoritmo Genético de valor real para minimização.

    Parâmetros
    ----------
    func           : callable(q) → float
    dim            : número de variáveis (6 para o robô)
    lower/upper    : limites articulares
    pop_size       : tamanho da população
    max_gen        : número máximo de gerações
    crossover_rate : probabilidade de crossover (Pc)
    mutation_rate  : probabilidade de mutação por gene (Pm)
    mutation_sigma : desvio padrão da mutação gaussiana
                     (em fração do range articular)
    tournament_size: tamanho do torneio de seleção
    alpha_blx      : parâmetro do BLX-α crossover
    seed           : semente aleatória

    Retorna dict com:
        best_x    – melhor solução encontrada
        best_f    – melhor fitness
        history   – evolução do melhor fitness por geração
        eval_count– total de avaliações da função objetivo
        iterations– total de gerações
    """
    rng = np.random.default_rng(seed)
    ranges = upper - lower

    # ── Inicialização ──────────────────────────────────────────
    pop = rng.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    eval_count = pop_size

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f]

    # ── Operadores internos ────────────────────────────────────

    def tournament_select() -> int:
        """Seleção por torneio: retorna índice do vencedor."""
        candidates = rng.integers(0, pop_size, tournament_size)
        return int(candidates[np.argmin(fitness[candidates])])

    def blx_alpha_crossover(p1: np.ndarray, p2: np.ndarray):
        """BLX-α: gera dois filhos explorando além do intervalo dos pais."""
        d = np.abs(p2 - p1)
        lo_blend = np.minimum(p1, p2) - alpha_blx * d
        hi_blend = np.maximum(p1, p2) + alpha_blx * d
        c1 = rng.uniform(lo_blend, hi_blend)
        c2 = rng.uniform(lo_blend, hi_blend)
        return (np.clip(c1, lower, upper),
                np.clip(c2, lower, upper))

    def mutate(individual: np.ndarray) -> np.ndarray:
        """Mutação gaussiana: cada gene muta com probabilidade mutation_rate."""
        mask = rng.random(dim) < mutation_rate
        noise = rng.normal(0, mutation_sigma * ranges, dim)
        individual = individual + mask * noise
        return np.clip(individual, lower, upper)

    # ── Loop geracional ────────────────────────────────────────
    for _ in range(max_gen):
        new_pop = []

        # Elitismo: preserva o melhor
        new_pop.append(best_x.copy())

        while len(new_pop) < pop_size:
            # Seleção
            i1 = tournament_select()
            i2 = tournament_select()

            p1 = pop[i1].copy()
            p2 = pop[i2].copy()

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = blx_alpha_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutação
            c1 = mutate(c1)
            c2 = mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop)
        fitness = np.array([func(ind) for ind in pop])
        eval_count += pop_size

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_f:
            best_f = fitness[best_idx]
            best_x = pop[best_idx].copy()

        history.append(best_f)

    return {
        "best_x":     best_x,
        "best_f":     best_f,
        "history":    np.array(history),
        "eval_count": eval_count,
        "iterations": max_gen,
    }
