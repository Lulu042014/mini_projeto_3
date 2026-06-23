# 🤖 Mini-Projeto 3 — Comparação de Metaheurísticas

**Computação Evolutiva 2026.1 · UFC Itapajé — Campus Jardins de Anita**  
**Autoras:** Luíza Nogueira & Ana Julia Freitas

---

## 📌 Sobre o Projeto

Comparação experimental entre três metaheurísticas aplicadas ao problema de **cinemática inversa** de um manipulador robótico **Fanuc R-2000iA/200i (6-DOF)**:

| Algoritmo | Tipo | Avaliações/iter |
|---|---|---|
| 🧬 GA — Algoritmo Genético | Populacional | ~50 |
| 🐦 PSO — Particle Swarm Optimization | Enxame | ~30 |
| 🪲 BAS — Beetle Antennae Search | Agente único | ~3 |
| 🔗 Híbrido PSO→BAS | Combinado | ~19 |

---

## 🎯 Problema

Dado um ponto no espaço 3D, encontrar os **6 ângulos articulares** do robô que minimizam o erro entre a pose desejada e a calculada pelo modelo cinemático direto (DH clássico).

**Função objetivo:**

**Restrições:** limites articulares reais do Fanuc 200i. Estratégia de reparo: **clipping**.

---

## 📊 Resultados (20 execuções independentes)

| Algoritmo | Melhor | Média | DP | Tempo | Avaliações |
|---|---|---|---|---|---|
| BAS | 1.39e-01 | 5.33e-01 | 2.21e-01 | 40 ms | 1.501 |
| PSO | 3.10e-02 | 1.59e-01 | 1.03e-01 | 367 ms | 15.030 |
| **GA** | **4.49e-04** | **4.40e-02** | **5.73e-02** | 1080 ms | 25.050 |
| Híbrido | 1.55e-02 | 1.46e-01 | 1.02e-01 | 253 ms | 9.631 |

> 📈 O GA obteve a melhor qualidade. O BAS foi 27× mais rápido. O Híbrido superou o PSO puro em todas as métricas com 36% menos avaliações.

---

## 🗂️ Estrutura do Projeto
---

## 🚀 Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/Lulu042014/mini_projeto_3.git
cd mini_projeto_3/mini_projeto3

# 2. Crie o ambiente virtual e instale dependências
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Execute
python main.py
```

---

## 🔬 Metodologia

O critério principal de comparação é o **número acumulado de avaliações da função objetivo** (não iterações), para garantir uma comparação justa entre algoritmos com custos por iteração distintos.

### Bônus — Estratégia Híbrida PSO→BAS
- **Fase 1 (60%):** PSO explora globalmente o espaço 6D
- **Fase 2 (40%):** BAS refina localmente a partir do melhor ponto do PSO
- Resultado: 2× melhor fitness, 36% menos avaliações que PSO puro

---

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-2.4-blue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11-blue)
