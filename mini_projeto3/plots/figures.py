"""
plots/figures.py
================
Gera todas as figuras exigidas pelo Mini-Projeto 3.

Figuras produzidas:
  fig1 – Convergência por iterações (média ± std)
  fig2 – Convergência por avaliações da função objetivo  ← principal
  fig3 – Boxplots de qualidade da solução final
  fig4 – Custo computacional (tempo médio de execução)
  fig5 – Tabela comparativa visual
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Estilo global ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         120,
})

COLORS = {
    "BAS":               "#E05C2F",
    "PSO":               "#2E86AB",
    "GA":                "#3BB273",
    "Híbrido (PSO→BAS)": "#9B59B6",
}

LINE_STYLES = {
    "BAS":               "-",
    "PSO":               "--",
    "GA":                "-.",
    "Híbrido (PSO→BAS)": ":",
}

OUTPUT_DIR = "outputs"


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _algo_names(results: dict) -> list[str]:
    return [k for k in results if not k.startswith("_")]


# ──────────────────────────────────────────────────────────────
#  FIG 1 – Convergência por iterações
# ──────────────────────────────────────────────────────────────

def fig_convergence_by_iterations(results: dict, save: bool = True) -> plt.Figure:
    """Curvas de convergência médias por número de iterações."""
    names = _algo_names(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Convergência por Iterações – Fanuc 200i (média de 20 execuções)",
        fontsize=13, fontweight="bold"
    )

    for name in names:
        hists = np.array(results[name]["histories"])
        # Uniformiza comprimento (corta pelo menor)
        min_len = min(len(h) for h in hists)
        hists = np.array([h[:min_len] for h in hists])

        mean_h = np.mean(hists, axis=0)
        std_h  = np.std(hists, axis=0)
        iters  = np.arange(min_len)

        color = COLORS.get(name, "gray")
        ls    = LINE_STYLES.get(name, "-")

        ax.semilogy(iters, mean_h + 1e-10, color=color, lw=2,
                    linestyle=ls, label=name)
        ax.fill_between(
            iters,
            np.maximum(mean_h - std_h, 1e-10),
            mean_h + std_h,
            alpha=0.15, color=color
        )

    ax.set_xlabel("Iterações / Gerações", fontsize=11)
    ax.set_ylabel("Melhor fitness – erro IK (m) [log]", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save:
        _ensure_output()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig1_convergencia_iteracoes.png"),
                    dpi=200, bbox_inches="tight")
        print("  ✓ fig1_convergencia_iteracoes.png")

    return fig


# ──────────────────────────────────────────────────────────────
#  FIG 2 – Convergência por avaliações (PRINCIPAL)
# ──────────────────────────────────────────────────────────────

def fig_convergence_by_evals(results: dict, save: bool = True) -> plt.Figure:
    """
    Gráfico principal: Fitness vs Número de Avaliações da Função Objetivo.
    Permite comparação justa entre algoritmos com custos por iteração distintos.
    """
    names = _algo_names(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Convergência por Avaliações da Função Objetivo (comparação justa)\n"
        "Fanuc 200i – média de 20 execuções",
        fontsize=12, fontweight="bold"
    )

    for name in names:
        hists      = results[name]["histories"]
        evals_list = results[name]["evals"]

        color = COLORS.get(name, "gray")
        ls    = LINE_STYLES.get(name, "-")

        # Interpola cada histórico para eixo de avaliações uniforme
        max_evals = int(np.max(evals_list))
        eval_axis = np.linspace(0, max_evals, 300)

        interp_hists = []
        for hist, total_evals in zip(hists, evals_list):
            # Cria eixo de avaliações proporcional ao histórico
            n = len(hist)
            eval_pts = np.linspace(0, total_evals, n)
            interp = np.interp(eval_axis, eval_pts, hist)
            interp_hists.append(interp)

        interp_hists = np.array(interp_hists)
        mean_h = np.mean(interp_hists, axis=0)
        std_h  = np.std(interp_hists, axis=0)

        ax.semilogy(eval_axis, mean_h + 1e-10, color=color, lw=2.5,
                    linestyle=ls, label=name)
        ax.fill_between(
            eval_axis,
            np.maximum(mean_h - std_h, 1e-10),
            mean_h + std_h,
            alpha=0.15, color=color
        )

    ax.set_xlabel("Número acumulado de avaliações da função objetivo", fontsize=11)
    ax.set_ylabel("Melhor fitness – erro IK (m) [log]", fontsize=11)
    ax.legend(fontsize=10)

    # Anotação explicativa
    ax.annotate(
        "BAS: ~3 aval./iter.\nPSO: ~30 aval./iter.\nGA: ~50 aval./iter.",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=8, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

    fig.tight_layout()

    if save:
        _ensure_output()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig2_convergencia_avaliacoes.png"),
                    dpi=200, bbox_inches="tight")
        print("  ✓ fig2_convergencia_avaliacoes.png  ← GRÁFICO PRINCIPAL")

    return fig


# ──────────────────────────────────────────────────────────────
#  FIG 3 – Boxplots de qualidade
# ──────────────────────────────────────────────────────────────

def fig_boxplot_quality(results: dict, save: bool = True) -> plt.Figure:
    """Boxplots do fitness final por algoritmo."""
    names = _algo_names(results)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "Qualidade da Solução Final – Fanuc 200i (20 execuções)",
        fontsize=13, fontweight="bold"
    )

    data_box = [results[n]["vals"] for n in names]
    colors   = [COLORS.get(n, "gray") for n in names]

    bp = ax.boxplot(
        data_box,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=2.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker, color in zip(bp["whiskers"], [c for c in colors for _ in range(2)]):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"], [c for c in colors for _ in range(2)]):
        cap.set_color(color)
    for flier, color in zip(bp["fliers"], colors):
        flier.set(marker="o", color=color, alpha=0.4)

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, fontsize=10)
    ax.set_ylabel("Fitness final – erro IK (m) [log]", fontsize=11)
    ax.set_title("Distribuição do erro de posicionamento final")

    fig.tight_layout()

    if save:
        _ensure_output()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig3_boxplot_qualidade.png"),
                    dpi=200, bbox_inches="tight")
        print("  ✓ fig3_boxplot_qualidade.png")

    return fig


# ──────────────────────────────────────────────────────────────
#  FIG 4 – Custo computacional
# ──────────────────────────────────────────────────────────────

def fig_computational_cost(results: dict, save: bool = True) -> plt.Figure:
    """Barras de tempo médio de execução e avaliações médias."""
    names = _algo_names(results)
    colors = [COLORS.get(n, "gray") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Custo Computacional – Fanuc 200i",
        fontsize=13, fontweight="bold"
    )

    # Tempo médio
    ax = axes[0]
    times = [results[n]["summary"]["time_mean"] * 1000 for n in names]
    bars = ax.bar(names, times, color=colors, alpha=0.8, edgecolor="white")
    ax.bar_label(bars, fmt="%.0f ms", fontsize=9, padding=3)
    ax.set_ylabel("Tempo médio de execução (ms)")
    ax.set_title("Tempo médio por execução")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, fontsize=9)

    # Avaliações médias
    ax2 = axes[1]
    evals = [results[n]["summary"]["eval_mean"] for n in names]
    bars2 = ax2.bar(names, evals, color=colors, alpha=0.8, edgecolor="white")
    ax2.bar_label(bars2, fmt="%.0f", fontsize=9, padding=3)
    ax2.set_ylabel("Média de avaliações da função objetivo")
    ax2.set_title("Avaliações por execução")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, fontsize=9)

    fig.tight_layout()

    if save:
        _ensure_output()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig4_custo_computacional.png"),
                    dpi=200, bbox_inches="tight")
        print("  ✓ fig4_custo_computacional.png")

    return fig


# ──────────────────────────────────────────────────────────────
#  FIG 5 – Tabela visual comparativa
# ──────────────────────────────────────────────────────────────

def fig_summary_table(results: dict, save: bool = True) -> plt.Figure:
    """Tabela comparativa visual com todos os indicadores."""
    names = _algo_names(results)

    col_labels = ["Algoritmo", "Melhor", "Pior", "Média", "DP", "Tempo (ms)", "Aval. médias"]
    rows = []
    for name in names:
        s = results[name]["summary"]
        rows.append([
            name,
            f"{s['best']:.4e}",
            f"{s['worst']:.4e}",
            f"{s['mean']:.4e}",
            f"{s['std']:.4e}",
            f"{s['time_mean']*1000:.1f}",
            f"{s['eval_mean']:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 2.5 + 0.5 * len(names)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Cabeçalho colorido
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Linhas alternadas
    for i, name in enumerate(names, start=1):
        row_color = COLORS.get(name, "gray") + "22"  # alpha hex
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(row_color)

    fig.suptitle(
        "Tabela Comparativa – Mini-Projeto 3 | Fanuc 200i",
        fontsize=12, fontweight="bold", y=0.98
    )
    fig.tight_layout()

    if save:
        _ensure_output()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig5_tabela_comparativa.png"),
                    dpi=200, bbox_inches="tight")
        print("  ✓ fig5_tabela_comparativa.png")

    return fig


# ──────────────────────────────────────────────────────────────
#  GERAR TODAS AS FIGURAS
# ──────────────────────────────────────────────────────────────

def generate_all_figures(results: dict):
    """Gera e salva todas as figuras do projeto."""
    print("\n  Gerando figuras...")
    fig_convergence_by_iterations(results)
    fig_convergence_by_evals(results)
    fig_boxplot_quality(results)
    fig_computational_cost(results)
    fig_summary_table(results)
    print(f"\n  Todas as figuras salvas em /{OUTPUT_DIR}/")
