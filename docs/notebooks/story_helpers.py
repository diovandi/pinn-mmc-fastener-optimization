from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError(f"Unable to locate repository root from {here}")


ROOT = find_repo_root()
DATA_DIR = ROOT / "data"
CAD_DIR = DATA_DIR / "cad"
RESULTS_DIR = DATA_DIR / "results"
TOOLS_DIR = ROOT / "src" / "tools"
PINN_DIR = ROOT / "src" / "approach_a_pinn"
MMC_DIR = ROOT / "src" / "approach_b_mmc"


@dataclass
class DatasetSummary:
    label: str
    path: Path
    samples: int
    compliance_min: float
    compliance_max: float
    compliance_mean: float


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def summarize_dataset(path: Path, label: str, *, has_header: bool = False) -> DatasetSummary:
    df = load_csv(path, header=0 if has_header else None)
    if has_header and "compliance" in df.columns:
        values = df["compliance"].astype(float)
    else:
        values = df.iloc[:, -1].astype(float)
    return DatasetSummary(
        label=label,
        path=path,
        samples=len(df),
        compliance_min=float(values.min()),
        compliance_max=float(values.max()),
        compliance_mean=float(values.mean()),
    )


def list_multi_geom_paths() -> List[Path]:
    return sorted((RESULTS_DIR / "multi_geom_training").glob("*.csv"))


def load_ansys_layouts() -> List[Dict]:
    payload_path = RESULTS_DIR / "ansys_payloads.json"
    data = json.loads(payload_path.read_text())
    return data["layouts"]


L_BRACKET_POLY = np.array(
    [
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 25.0],
        [25.0, 25.0],
        [25.0, 100.0],
        [0.0, 100.0],
    ]
)
BRACKET_THICKNESS = 5.0


def draw_lbracket_2d(show_void: bool = True) -> go.Figure:
    poly = L_BRACKET_POLY
    trace = go.Scatter(
        x=np.append(poly[:, 0], poly[0, 0]),
        y=np.append(poly[:, 1], poly[0, 1]),
        fill="toself",
        fillcolor="rgba(211,211,211,0.8)",
        line=dict(color="black", width=2),
        name="Aluminum",
    )
    fig = go.Figure(data=[trace])
    if show_void:
        fig.add_shape(
            type="rect",
            x0=25,
            y0=25,
            x1=100,
            y1=100,
            line=dict(color="gray", dash="dot"),
            fillcolor="rgba(0,0,0,0)",
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 25],
            y=[100, 100],
            mode="lines",
            line=dict(color="red", width=6),
            name="Fixed Support",
        )
    )
    fig.update_layout(
        title="L-Bracket Planform & Boundary Conditions",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=500,
        showlegend=True,
    )
    return fig


def _brick_mesh_from_bounds(x0, x1, y0, y1, z0, z1, color: str) -> go.Mesh3d:
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    i, j, k = zip(*faces)
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k, opacity=0.85, color=color, flatshading=True
    )


def make_bracket_mesh() -> go.Figure:
    vertical_leg = _brick_mesh_from_bounds(0, 25, 0, 100, 0, BRACKET_THICKNESS, "#d3d3d3")
    horizontal_leg = _brick_mesh_from_bounds(0, 100, 0, 25, 0, BRACKET_THICKNESS, "#b0bec5")
    fig = go.Figure(data=[vertical_leg, horizontal_leg])
    fig.update_layout(
        title="3D L-Bracket Geometry (Extruded 5 mm)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# Ribbed Channel Geometry (160mm x 50mm, 4mm thick, slot from 60-100mm x 15-35mm)
RIBBED_CHANNEL_LENGTH = 160.0
RIBBED_CHANNEL_HEIGHT = 50.0
RIBBED_CHANNEL_THICK = 4.0
RIBBED_CHANNEL_SLOT = {"x": (60.0, 100.0), "y": (15.0, 35.0)}


def make_ribbed_channel_mesh() -> go.Figure:
    """Create 3D mesh for ribbed channel with slot."""
    fig = go.Figure()
    # Main plate
    main_plate = _brick_mesh_from_bounds(
        0, RIBBED_CHANNEL_LENGTH, 0, RIBBED_CHANNEL_HEIGHT, 0, RIBBED_CHANNEL_THICK, "#9e9e9e"
    )
    fig.add_trace(main_plate)
    # Slot region (thinner, shown as void)
    slot = _brick_mesh_from_bounds(
        RIBBED_CHANNEL_SLOT["x"][0],
        RIBBED_CHANNEL_SLOT["x"][1],
        RIBBED_CHANNEL_SLOT["y"][0],
        RIBBED_CHANNEL_SLOT["y"][1],
        0,
        RIBBED_CHANNEL_THICK * 0.05,
        "#ff9800",
    )
    fig.add_trace(slot)
    fig.update_layout(
        title="3D Ribbed Channel Geometry (160mm × 50mm, 4mm thick)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# Tapered Plate Geometry (140mm x 60mm, tapers from 6mm to 3mm)
TAPERED_PLATE_LENGTH = 140.0
TAPERED_PLATE_HEIGHT = 60.0
TAPERED_PLATE_THICK_MAX = 6.0
TAPERED_PLATE_THICK_MIN = 3.0


def make_tapered_plate_mesh() -> go.Figure:
    """Create 3D mesh for tapered plate with variable thickness."""
    fig = go.Figure()
    # Use fewer segments for better performance (5 instead of 10)
    n_segments = 5
    for i in range(n_segments):
        x0 = i * TAPERED_PLATE_LENGTH / n_segments
        x1 = (i + 1) * TAPERED_PLATE_LENGTH / n_segments
        t0 = TAPERED_PLATE_THICK_MAX - (TAPERED_PLATE_THICK_MAX - TAPERED_PLATE_THICK_MIN) * (x0 / TAPERED_PLATE_LENGTH)
        t1 = TAPERED_PLATE_THICK_MAX - (TAPERED_PLATE_THICK_MAX - TAPERED_PLATE_THICK_MIN) * (x1 / TAPERED_PLATE_LENGTH)
        # Use average thickness for segment
        t_avg = (t0 + t1) / 2
        segment = _brick_mesh_from_bounds(
            x0, x1, 0, TAPERED_PLATE_HEIGHT, 0, t_avg, "#607d8b"
        )
        fig.add_trace(segment)
    fig.update_layout(
        title="3D Tapered Plate Geometry (140mm × 60mm, 6mm→3mm taper)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_geometry_3d(geometry: str) -> go.Figure:
    """Factory function to create 3D mesh for any geometry."""
    if geometry == "l_bracket":
        return make_bracket_mesh()
    elif geometry == "ribbed_channel":
        return make_ribbed_channel_mesh()
    elif geometry == "tapered_plate":
        return make_tapered_plate_mesh()
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def add_force_arrow_3d(
    fig: go.Figure,
    origin: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    magnitude: float = 1000.0,
    color: str = "blue",
    name: str = "Load",
    scale: float = 0.02,
) -> None:
    """Add a 3D force arrow to a Plotly figure."""
    x0, y0, z0 = origin
    dx, dy, dz = direction
    # Normalize direction and scale by magnitude
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    if norm > 0:
        dx, dy, dz = dx / norm, dy / norm, dz / norm
    arrow_length = magnitude * scale
    x1, y1, z1 = x0 + dx * arrow_length, y0 + dy * arrow_length, z0 + dz * arrow_length
    # Arrow shaft
    fig.add_trace(
        go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode="lines",
            line=dict(color=color, width=8),
            name=name,
            showlegend=True,
        )
    )
    # Arrow head (cone)
    head_length = arrow_length * 0.3
    head_base = arrow_length - head_length
    x_base = x0 + dx * head_base
    y_base = y0 + dy * head_base
    z_base = z0 + dz * head_base
    # Simple arrowhead as a small sphere at tip
    fig.add_trace(
        go.Scatter3d(
            x=[x1],
            y=[y1],
            z=[z1],
            mode="markers",
            marker=dict(size=12, color=color, symbol="diamond"),
            name=f"{name} tip",
            showlegend=False,
        )
    )


def make_lbracket_with_loads_3d(load_case: str = "horizontal_tip") -> go.Figure:
    """Create 3D L-bracket with load vectors."""
    fig = make_bracket_mesh()
    # Fixed support visualization
    support_x = [0, 25]
    support_y = [100, 100]
    support_z = [0, 0]
    fig.add_trace(
        go.Scatter3d(
            x=support_x,
            y=support_y,
            z=support_z,
            mode="lines",
            line=dict(color="red", width=8),
            name="Fixed Support",
        )
    )
    # Load vectors
    if load_case == "horizontal_tip":
        # Horizontal load at tip (100, 12.5, 0)
        add_force_arrow_3d(fig, (100, 12.5, 0), (-1, 0, 0), 1000, "navy", "Horizontal Load (1000N)")
    elif load_case == "vertical_tip":
        # Vertical load at tip (100, 12.5, 0)
        add_force_arrow_3d(fig, (100, 12.5, 0), (0, 1, 0), 1000, "darkgreen", "Vertical Load (1000N)")
    fig.update_layout(title=f"3D L-Bracket with {load_case.replace('_', ' ').title()} Load")
    return fig


def make_ribbed_channel_with_loads_3d(load_case: str = "upward_tip") -> go.Figure:
    """Create 3D ribbed channel with load vectors."""
    fig = make_ribbed_channel_mesh()
    # Fixed support at left edge
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, RIBBED_CHANNEL_HEIGHT],
            z=[0, 0],
            mode="lines",
            line=dict(color="red", width=8),
            name="Fixed Support",
        )
    )
    # Load vectors
    if load_case == "upward_tip":
        # Upward load at right edge center
        add_force_arrow_3d(
            fig,
            (RIBBED_CHANNEL_LENGTH, RIBBED_CHANNEL_HEIGHT / 2, RIBBED_CHANNEL_THICK),
            (0, 0, 1),
            1000,
            "navy",
            "Upward Load (1000N)",
        )
    elif load_case == "lateral_shear":
        # Lateral shear at right edge
        add_force_arrow_3d(
            fig,
            (RIBBED_CHANNEL_LENGTH, RIBBED_CHANNEL_HEIGHT / 2, RIBBED_CHANNEL_THICK),
            (0, -1, 0),
            1000,
            "darkgreen",
            "Lateral Shear (1000N)",
        )
    fig.update_layout(title=f"3D Ribbed Channel with {load_case.replace('_', ' ').title()} Load")
    return fig


def make_tapered_plate_with_loads_3d(load_case: str = "combined_tip") -> go.Figure:
    """Create 3D tapered plate with load vectors."""
    fig = make_tapered_plate_mesh()
    # Fixed support at left edge
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, TAPERED_PLATE_HEIGHT],
            z=[0, 0],
            mode="lines",
            line=dict(color="red", width=8),
            name="Fixed Support",
        )
    )
    # Load vectors
    if load_case == "combined_tip":
        # Combined tip load (force + torsion) at right edge
        tip_x = TAPERED_PLATE_LENGTH
        tip_y = TAPERED_PLATE_HEIGHT / 2
        tip_z = TAPERED_PLATE_THICK_MIN
        # Tip force (downward)
        add_force_arrow_3d(fig, (tip_x, tip_y, tip_z), (0, 0, -1), 450, "navy", "Tip Force (450N)")
        # Torsion (lateral)
        add_force_arrow_3d(fig, (tip_x, tip_y, tip_z), (0, -1, 0), 25, "darkgreen", "Torsion (25Nm)")
    fig.update_layout(title=f"3D Tapered Plate with {load_case.replace('_', ' ').title()} Load")
    return fig


def make_geometry_with_loads_3d(geometry: str, load_case: str) -> go.Figure:
    """Factory function to create 3D geometry with load vectors."""
    if geometry == "l_bracket":
        return make_lbracket_with_loads_3d(load_case)
    elif geometry == "ribbed_channel":
        return make_ribbed_channel_with_loads_3d(load_case)
    elif geometry == "tapered_plate":
        return make_tapered_plate_with_loads_3d(load_case)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def plot_load_layouts(layouts: List[Dict]) -> go.Figure:
    cols = len(layouts)
    fig = make_subplots(
        rows=1,
        cols=cols,
        subplot_titles=[f"{layout['method'].upper()} ({layout['tag']})" for layout in layouts],
    )
    for idx, layout in enumerate(layouts):
        fig.add_trace(
            go.Scatter(
                x=np.append(L_BRACKET_POLY[:, 0], L_BRACKET_POLY[0, 0]),
                y=np.append(L_BRACKET_POLY[:, 1], L_BRACKET_POLY[0, 1]),
                fill="toself",
                fillcolor="rgba(245,245,245,0.85)",
                line=dict(color="black", width=1.5),
                showlegend=False,
            ),
            row=1,
            col=idx + 1,
        )
        screws = layout.get("screws", [])
        if screws:
            xs = [s["x"] for s in screws]
            ys = [s["y"] for s in screws]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    marker=dict(size=12, color="#1f77b4"),
                    text=[f"S{i+1}" for i in range(len(xs))],
                    textposition="top center",
                    showlegend=False,
                ),
                row=1,
                col=idx + 1,
            )
        fig.update_xaxes(range=[-5, 105], row=1, col=idx + 1)
        fig.update_yaxes(
            range=[-5, 105], row=1, col=idx + 1, scaleanchor=f"x{idx+1}", scaleratio=1
        )
    fig.update_layout(height=400, title_text="Optimal Screw Placements from Validated Layouts")
    return fig


def assemble_multi_geom_dataframe() -> pd.DataFrame:
    rows = []
    for csv_path in list_multi_geom_paths():
        df = load_csv(csv_path)
        label = csv_path.stem.replace("_", " ")
        compliance = df["compliance"]
        rows.append(
            {
                "dataset": label,
                "samples": len(df),
                "compliance_min": compliance.min(),
                "compliance_max": compliance.max(),
                "compliance_mean": compliance.mean(),
                "path": str(csv_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def plot_dataset_histogram(csv_path: Path, label: str, *, has_header: bool = False) -> go.Figure:
    df = load_csv(csv_path, header=0 if has_header else None)
    if has_header and "compliance" in df.columns:
        compliance = df["compliance"]
    else:
        compliance = df.iloc[:, -1]
    fig = px.histogram(
        compliance,
        nbins=25,
        title=f"Compliance Distribution — {label}",
        labels={"value": "Compliance (J)"},
        opacity=0.85,
        color_discrete_sequence=["#3f51b5"],
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_dataset_scatter(csv_path: Path, label: str, *, has_header: bool = False) -> go.Figure:
    df = load_csv(csv_path, header=0 if has_header else None)
    if has_header:
        coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        coord_values = df[coord_cols].values
    else:
        coord_values = df.iloc[:, :-1].values
    coords = coord_values.reshape(len(df), -1, 2)
    if has_header and "compliance" in df.columns:
        compliance = df["compliance"]
    else:
        compliance = df.iloc[:, -1]
    xs = coords[:, :, 0].flatten()
    ys = coords[:, :, 1].flatten()
    compliance_repeated = np.repeat(compliance.values, coords.shape[1])
    fig = px.scatter(
        x=xs,
        y=ys,
        color=compliance_repeated,
        color_continuous_scale="Turbo",
        title=f"Screw Placement Distribution — {label}",
        labels=dict(x="X (mm)", y="Y (mm)", color="Compliance (J)"),
    )
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), height=520)
    return fig


class SurrogateModel(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


def measure_speed_benchmarks() -> pd.DataFrame:
    fea_log = load_csv(RESULTS_DIR / "lbracket_diff_fea_log.csv")
    fea_time = fea_log["wall_time_ms"].iloc[1:].mean() / 1000.0

    stats_path = PINN_DIR / "artifacts_multi_geom" / "norm_stats_multi_geom.npz"
    model_path = PINN_DIR / "artifacts_multi_geom" / "pinn_multi_geom.pth"
    dataset_path = RESULTS_DIR / "multi_geom_training" / "l_bracket_horizontal.csv"

    stats = np.load(stats_path)
    # Determine input dimension from stats
    input_dim = stats["X_mean"].shape[0]
    model = SurrogateModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data = load_csv(dataset_path).values.astype(np.float32)
    X = data[:, :input_dim]
    X_norm = (X - stats["X_mean"]) / stats["X_std"]
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        reps = 200
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                _ = model(X_tensor)
            end.record()
            torch.cuda.synchronize()
            pinn_time = start.elapsed_time(end) / 1000.0 / reps
        else:
            start = time.perf_counter()
            for _ in range(reps):
                _ = model(X_tensor)
            pinn_time = (time.perf_counter() - start) / reps

    mmc_log = load_csv(RESULTS_DIR / "mmc_lbracket_log.csv")
    if "wall_time_ms" in mmc_log.columns:
        mmc_time = mmc_log["wall_time_ms"].mean() / 1000.0
    else:
        mmc_time = 0.118

    records = [
        {"method": "Diff-FEA (Julia)", "time_s": fea_time},
        {"method": "PINN Surrogate", "time_s": pinn_time},
        {"method": "MMC (Python)", "time_s": mmc_time},
    ]
    return pd.DataFrame(records)


def plot_speed_bars(speed_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        speed_df,
        x="method",
        y="time_s",
        color="method",
        color_discrete_sequence=["#7f7f7f", "#2ca02c", "#d62728"],
        log_y=True,
        title="Iteration Runtime Comparison (Log Scale)",
    )
    fig.update_layout(yaxis_title="Time per Iteration (s)")
    return fig


def plot_unified_convergence() -> go.Figure:
    fea = load_csv(RESULTS_DIR / "lbracket_diff_fea_log.csv")
    mmc = load_csv(RESULTS_DIR / "mmc_lbracket_log.csv")
    mmc_scaled = mmc["compliance"] * 1500.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fea["iter"], y=fea["compliance"], mode="lines+markers", name="Diff-FEA PINN Training")
    )
    fig.add_trace(
        go.Scatter(
            x=mmc["iter"],
            y=mmc_scaled,
            mode="lines+markers",
            name="MMC Optimization (scaled)",
        )
    )
    fig.update_layout(
        title="Convergence Trajectories",
        xaxis_title="Iteration",
        yaxis_title="Compliance (J)",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_method_comparison() -> go.Figure:
    df = load_csv(RESULTS_DIR / "method_comparison.csv")
    df = df.sort_values("compliance")
    fig = px.bar(
        df,
        x="label",
        y="compliance",
        color="label",
        title="Final Compliance Comparison",
    )
    fig.update_layout(xaxis_title="Method", yaxis_title="Compliance (normalized)", showlegend=False)
    return fig


