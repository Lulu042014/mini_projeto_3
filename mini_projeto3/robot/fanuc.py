"""
robot/fanuc.py
==============
Cinemática do manipulador Fanuc 200i (6-DOF).
Convenção DH clássica (Denavit-Hartenberg padrão).
Função objetivo: erro de posição (m) + 0.3 * erro de orientação (rad).
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
#  LIMITES ARTICULARES (graus → radianos)
# ──────────────────────────────────────────────────────────────

JOINT_LIMITS_DEG = [
    (-180, 180),
    (-100, 110),
    (-210,  70),
    (-190, 190),
    (-120, 120),
    (-360, 360),
]

_lo_rad = [np.deg2rad(lo) for lo, hi in JOINT_LIMITS_DEG]
_hi_rad = [np.deg2rad(hi) for lo, hi in JOINT_LIMITS_DEG]

LOWER_BOUNDS = np.array(_lo_rad)
UPPER_BOUNDS = np.array(_hi_rad)

# ──────────────────────────────────────────────────────────────
#  PARÂMETROS DH – Fanuc R-2000iA/200F
#  [a (m), alpha (rad), d (m), theta_offset (rad)]
# ──────────────────────────────────────────────────────────────

DH_PARAMS = np.array([
    [0.150,  0.0,           0.525,  0.0],
    [0.790, -np.pi / 2,    0.0,    0.0],
    [0.150,  0.0,           0.0,    0.0],
    [0.0,    np.pi / 2,    0.860,  0.0],
    [0.0,   -np.pi / 2,    0.0,    0.0],
    [0.0,    np.pi / 2,    0.100,  0.0],
])


def _dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Matriz de transformação DH clássica."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct,      -st * ca,   st * sa,   a * ct],
        [st,       ct * ca,  -ct * sa,   a * st],
        [0.0,      sa,        ca,        d     ],
        [0.0,      0.0,       0.0,       1.0   ],
    ])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Cinemática direta: retorna matriz homogênea 4x4."""
    T = np.eye(4)
    for i in range(6):
        a, alpha, d, theta_off = DH_PARAMS[i]
        T = T @ _dh_matrix(a, alpha, d, q[i] + theta_off)
    return T


def end_effector_pose(q: np.ndarray):
    """Retorna (posição XYZ (3,), rotação (3,3))."""
    T = forward_kinematics(q)
    return T[:3, 3].copy(), T[:3, :3].copy()


def end_effector_position(q: np.ndarray) -> np.ndarray:
    return forward_kinematics(q)[:3, 3]


def rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> float:
    """Erro de orientação em radianos via ângulo da rotação relativa."""
    R_rel = R_current.T @ R_target
    cos_val = (np.trace(R_rel) - 1.0) / 2.0
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.arccos(cos_val))


def inverse_kinematics_error(
    q: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    w_pos: float = 1.0,
    w_rot: float = 0.3,
) -> float:
    """
    Função objetivo combinada:
        f = w_pos * ||pos - target_pos|| + w_rot * erro_orientacao

    Parâmetros
    ----------
    q          : ângulos articulares (6,) em radianos
    target_pos : posição alvo (3,) em metros
    target_rot : rotação alvo (3,3)
    w_pos      : peso do erro de posição
    w_rot      : peso do erro de orientação
    """
    pos, rot = end_effector_pose(q)
    err_pos = float(np.linalg.norm(pos - target_pos))
    err_rot = rotation_error(rot, target_rot)
    return w_pos * err_pos + w_rot * err_rot


def repair_solution(q: np.ndarray) -> np.ndarray:
    """Projeta q para dentro dos limites articulares (clipping)."""
    return np.clip(q, LOWER_BOUNDS, UPPER_BOUNDS)


def random_target(seed: int = 42):
    """
    Gera pose alvo alcançável via FK de configuração articular aleatória.
    Retorna (pos (3,), rot (3,3)).
    """
    rng = np.random.default_rng(seed)
    q_rand = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
    T = forward_kinematics(q_rand)
    return T[:3, 3].copy(), T[:3, :3].copy()


def print_robot_info():
    print("=" * 58)
    print("  Robô: Fanuc R-2000iA/200i  |  6-DOF")
    print("=" * 58)
    print(f"  {'Junta':<8} {'Min (°)':>10} {'Max (°)':>10}")
    print("  " + "-" * 30)
    for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
        print(f"  J{i + 1:<7} {lo:>10.1f} {hi:>10.1f}")
    print("  Objetivo: w_pos * ||Δpos|| + w_rot * ||Δori||")
    print("=" * 58)
