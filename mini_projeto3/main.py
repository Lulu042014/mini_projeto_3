import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



import numpy as np
from robot.fanuc import print_robot_info, random_target
from experiments.runner import run_experiments, print_summary_table
from plots.figures import generate_all_figures


def main():
    print("=" * 65)
    print("  Mini-Projeto 3 – GA vs PSO vs BAS vs Híbrido")
    print("  UFC Itapajé | Computação Evolutiva 2026.1")
    print("=" * 65)

    print_robot_info()

    target_pos, target_rot = random_target(seed=42)
    print(f"\n  Posição alvo (FK de q aleatório válido):")
    print(f"    x = {target_pos[0]:.4f} m")
    print(f"    y = {target_pos[1]:.4f} m")
    print(f"    z = {target_pos[2]:.4f} m")

    print("\n" + "─" * 65)
    print("  Iniciando experimentos (20 runs × 4 algoritmos)...")
    print("─" * 65)

    results = run_experiments(
        target_pos=target_pos,
        target_rot=target_rot,
        n_runs=20,
        max_iter=500,
        verbose=True,
    )

    print_summary_table(results)
    generate_all_figures(results)

    print("\n" + "=" * 65)
    print("  Concluído! Verifique a pasta outputs/ para os gráficos.")
    print("=" * 65)


if __name__ == "__main__":
    main()
