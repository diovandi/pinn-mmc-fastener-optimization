import numpy as np
from dataclasses import dataclass
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

E_BASE = 1.0
E_BOLT = 5.0


@dataclass
class DomainConfig:
    nelx: int = 40
    nely: int = 40
    void_cutoff_x: int = 16
    void_cutoff_y: int = 16


@dataclass
class ConstraintConfig:
    edge_margin: float = 1.0
    min_spacing: float = 2.0


@dataclass
class MMCConfig:
    n_screws: int = 2
    n_iters: int = 30
    lr: float = 0.5
    radius: float = 4.0
    beta: float = 5.0
    seed: int = 1234
    load_case: str = "horizontal_tip"


def get_ke():
    E, nu = 1.0, 0.3
    k = np.array([
        1/2-nu/6,  1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
       -1/4+nu/12, -1/8-nu/8,  nu/6,       1/8-3*nu/8
    ])
    KE = E/(1-nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[2], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[2]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[2], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[2], k[3], k[6], k[5], k[0]],
    ])
    return KE


def build_passive_mask(domain: DomainConfig):
    passive = np.zeros((domain.nely, domain.nelx), dtype=int)
    for i in range(domain.nelx):
        for j in range(domain.nely):
            if i > domain.void_cutoff_x and j > domain.void_cutoff_y:
                passive[j, i] = 1
    return passive


def update_design(design_vars, domain: DomainConfig, radius=4.0, beta=5.0):
    n_screws = len(design_vars) // 2
    grid_x, grid_y = np.meshgrid(np.arange(domain.nelx), np.arange(domain.nely))

    phi = -1.0 * np.ones((domain.nely, domain.nelx), dtype=float)
    for i in range(n_screws):
        cx = design_vars[2 * i]
        cy = design_vars[2 * i + 1]
        dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        phi = np.maximum(phi, radius - dist)

    return 1.0 / (1.0 + np.exp(-beta * phi))


def fe_solve(x_phys, ke, passive, domain: DomainConfig, load_case: str = "horizontal_tip"):
    ndof = 2 * (domain.nelx + 1) * (domain.nely + 1)
    x_eff = x_phys.copy()
    x_eff[passive == 1] = 0.0

    e_elem = E_BASE + x_eff * (E_BOLT - E_BASE)
    I, J, V = [], [], []
    for elx in range(domain.nelx):
        for ely in range(domain.nely):
            e_idx = elx * domain.nely + ely
            n1 = (domain.nely + 1) * elx + ely
            n2 = (domain.nely + 1) * (elx + 1) + ely
            nodes = [n1, n2, n2 + 1, n1 + 1]
            dofs = np.array([2 * n for n in nodes] + [2 * n + 1 for n in nodes])
            k_scaled = ke * e_elem[ely, elx]
            for i in range(8):
                for j in range(8):
                    I.append(dofs[i])
                    J.append(dofs[j])
                    V.append(k_scaled[i, j])

    K = coo_matrix((V, (I, J)), shape=(ndof, ndof)).tocsc()

    fixed_dofs = []
    for x in range(domain.void_cutoff_x + 1):
        n = x * (domain.nely + 1) + domain.nely
        fixed_dofs.extend([2 * n, 2 * n + 1])

    force = np.zeros(ndof)
    if load_case == "horizontal_tip":
        load_node = domain.nelx * (domain.nely + 1) + int(domain.void_cutoff_y / 2)
        load_dof = 2 * load_node + 1
        force[load_dof] = -1.0
    elif load_case == "vertical_tip":
        load_node = int(domain.void_cutoff_x / 2) * (domain.nely + 1)
        load_dof = 2 * load_node
        force[load_dof] = 1.0
    else:
        raise ValueError(f"Unknown load case: {load_case}")

    K_diag = K.diagonal().copy()
    K_diag[fixed_dofs] = 1e9
    K.setdiag(K_diag)

    u = spsolve(K, force)
    return u, force


def compliance(u, F):
    return float(F @ u)


def apply_constraints(design, domain: DomainConfig, constraints: ConstraintConfig):
    design = design.copy()
    n_screws = len(design) // 2
    min_x = constraints.edge_margin
    max_x = domain.nelx - constraints.edge_margin
    min_y = constraints.edge_margin
    max_y = domain.nely - constraints.edge_margin

    for i in range(n_screws):
        design[2 * i] = np.clip(design[2 * i], min_x, max_x)
        design[2 * i + 1] = np.clip(design[2 * i + 1], min_y, max_y)

    for i in range(n_screws):
        for j in range(i + 1, n_screws):
            dx = design[2 * j] - design[2 * i]
            dy = design[2 * j + 1] - design[2 * i + 1]
            dist = np.hypot(dx, dy)
            if 0 < dist < constraints.min_spacing:
                shift = 0.5 * (constraints.min_spacing - dist)
                direction = np.array([dx, dy]) / (dist + 1e-6)
                design[2 * i:2 * i + 2] -= direction * shift
                design[2 * j:2 * j + 2] += direction * shift

                design[2 * i] = np.clip(design[2 * i], min_x, max_x)
                design[2 * i + 1] = np.clip(design[2 * i + 1], min_y, max_y)
                design[2 * j] = np.clip(design[2 * j], min_x, max_x)
                design[2 * j + 1] = np.clip(design[2 * j + 1], min_y, max_y)

    return design


def run_mmc_lbracket(
    config: MMCConfig,
    domain: DomainConfig = DomainConfig(),
    constraints: ConstraintConfig = ConstraintConfig(),
):
    np.random.seed(config.seed)
    passive = build_passive_mask(domain)
    ke = get_ke()

    base_positions = np.linspace(
        constraints.edge_margin,
        max(constraints.edge_margin, domain.void_cutoff_x - constraints.edge_margin),
        config.n_screws,
    )
    design = np.zeros(2 * config.n_screws, dtype=float)
    for idx in range(config.n_screws):
        design[2 * idx] = base_positions[idx]
        design[2 * idx + 1] = constraints.edge_margin + idx * 2
    design = apply_constraints(design, domain, constraints)
    history = []

    for it in range(config.n_iters + 1):
        x_phys = update_design(design, domain, radius=config.radius, beta=config.beta)
        u, F = fe_solve(x_phys, ke, passive, domain, load_case=config.load_case)
        comp = compliance(u, F)

        record = {
            "iter": it,
            "compliance": comp,
            "x1": design[0],
            "y1": design[1],
            "x2": design[2],
            "y2": design[3],
        }
        history.append(record)

        if it == config.n_iters:
            break

        target = np.array(
            [domain.void_cutoff_x, domain.void_cutoff_y] * config.n_screws, dtype=float
        )
        grad_mock = target - design
        design = design + config.lr * 0.05 * grad_mock
        design = apply_constraints(design, domain, constraints)

    return history

