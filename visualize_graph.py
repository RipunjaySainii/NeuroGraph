"""
visualize_graph.py

Publication-quality EEG network visualizer for NeuroGraph.
Resolves node/edge overlap and clustering by implementing:
1. A 64-node Circular Connectome layout with non-overlapping radial electrode labels
2. Anatomical 10-20 Scalp Topography mapping real cortical electrode coordinates
3. Dynamic shortest-path routing highlighting information flow via Dijkstra
4. Glowing callouts for top degree-centrality hubs
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd
import mne
from mne.datasets import eegbci


# Official 64 standardized channel names for PhysioNet EEGBCI
CHANNEL_NAMES = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7",
    "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8",
    "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz"
]


def load_dataset(project_dir: Path, threshold: float = 0.5):
    """Load adjacency matrix and topological metrics from CSV files."""
    adj_path = project_dir / "eeg_adjacency_matrix.csv"
    metrics_path = project_dir / "topological_metrics.csv"

    if not adj_path.exists():
        raise FileNotFoundError(f"Missing {adj_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    adj_matrix = np.loadtxt(adj_path, delimiter=",")
    metrics_df = pd.read_csv(metrics_path)

    # Ensure integer node_id
    metrics_df["node_id"] = metrics_df["node_id"].astype(int)

    # Build NetworkX directed graph
    G = nx.DiGraph()
    for i in range(len(CHANNEL_NAMES)):
        row = metrics_df[metrics_df["node_id"] == i]
        cent = row["degree_centrality"].values[0] if not row.empty else 0.0
        deg = row["degree"].values[0] if not row.empty else 0
        w_deg = row["weighted_degree"].values[0] if not row.empty else 0.0
        avg_sp = row["avg_shortest_path"].values[0] if not row.empty else 0.0

        G.add_node(
            i,
            channel=CHANNEL_NAMES[i],
            degree_centrality=cent,
            degree=deg,
            weighted_degree=w_deg,
            avg_shortest_path=avg_sp,
        )

    for i in range(len(CHANNEL_NAMES)):
        for j in range(len(CHANNEL_NAMES)):
            if i != j and adj_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=float(adj_matrix[i, j]))

    return G, metrics_df, adj_matrix


def get_scalp_positions():
    """Retrieve 2D scalp positions for the 64 electrodes via MNE standard 10-05 montage."""
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        pos_dict = montage.get_positions()["ch_pos"]
        scalp_pos = {}

        # Extract 2D projection
        for i, ch in enumerate(CHANNEL_NAMES):
            if ch in pos_dict:
                x, y, _ = pos_dict[ch]
                scalp_pos[i] = (x, y)
            else:
                scalp_pos[i] = (0.0, 0.0)

        # Normalize coordinates to unit circle
        coords = np.array([scalp_pos[i] for i in range(len(CHANNEL_NAMES))])
        max_r = np.max(np.linalg.norm(coords, axis=1))
        if max_r > 0:
            for i in scalp_pos:
                scalp_pos[i] = (scalp_pos[i][0] / max_r * 0.88, scalp_pos[i][1] / max_r * 0.88)
        return scalp_pos
    except Exception as e:
        print(f"Notice: Using fallback circular layout for scalp ({e})")
        return nx.circular_layout(range(len(CHANNEL_NAMES)))


def draw_head_outline(ax, radius=0.98, color="#475569"):
    """Draw a biological head perimeter with nose and ears."""
    # Head circle
    head = plt.Circle((0, 0), radius, color=color, fill=False, linewidth=2.0, linestyle="--", alpha=0.7)
    ax.add_patch(head)

    # Nose
    nose = plt.Polygon(
        [[-0.09 * radius, radius], [0.0, radius + 0.12 * radius], [0.09 * radius, radius]],
        color=color, fill=False, linewidth=2.0, alpha=0.8
    )
    ax.add_patch(nose)

    # Left ear
    l_ear = plt.Polygon(
        [[-radius, 0.12 * radius], [-radius - 0.08 * radius, 0.0], [-radius, -0.12 * radius]],
        color=color, fill=False, linewidth=2.0, alpha=0.8
    )
    ax.add_patch(l_ear)

    # Right ear
    r_ear = plt.Polygon(
        [[radius, 0.12 * radius], [radius + 0.08 * radius, 0.0], [radius, -0.12 * radius]],
        color=color, fill=False, linewidth=2.0, alpha=0.8
    )
    ax.add_patch(r_ear)


def generate_visualization(G: nx.DiGraph, metrics_df: pd.DataFrame, adj_matrix: np.ndarray, output_path: Path):
    """Render publication-grade visualization separating nodes and highlighting information flow."""
    num_nodes = len(CHANNEL_NAMES)

    # Identify top hubs
    sorted_df = metrics_df.sort_values(by="degree_centrality", ascending=False)
    top_hubs = sorted_df.head(5)["node_id"].tolist()
    primary_hub = top_hubs[0]  # Node 4 (FC2)

    # Select representative flow paths
    peripheral_nodes = sorted_df[sorted_df["degree"] > 0].tail(4)["node_id"].tolist()
    distant_targets = [60, 63, 55, 18]

    flow_paths = []
    # 1. Peripheral convergence into primary hub
    for p in peripheral_nodes:
        if nx.has_path(G, p, primary_hub):
            flow_paths.append(nx.shortest_path(G, source=p, target=primary_hub, weight="weight"))

    # 2. Transmission from hub to distant brain lobes
    for t in distant_targets:
        if nx.has_path(G, primary_hub, t):
            flow_paths.append(nx.shortest_path(G, source=primary_hub, target=t, weight="weight"))

    # 3. Trans-network longest route
    if nx.has_path(G, 28, 63):
        flow_paths.append(nx.shortest_path(G, source=28, target=63, weight="weight"))

    # Extract highlighted edges
    flow_edges = set()
    for path in flow_paths:
        for u, v in zip(path[:-1], path[1:]):
            flow_edges.add((u, v))

    # Top backbone functional edges for subtle structural context
    strong_edges = []
    threshold_backbone = 0.88
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj_matrix[i, j] >= threshold_backbone and (i, j) not in flow_edges:
                strong_edges.append((i, j))

    # -------------------------------------------------------------
    # Setup Figure (24 x 12 inches for maximum clarity)
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(24, 12), facecolor="#0b0f19")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.12)

    # -------------------------------------------------------------
    # Panel 1: Circular Connectome (Zero-Overlap Architecture)
    # -------------------------------------------------------------
    ax_circ = fig.add_subplot(gs[0], facecolor="#0b0f19")
    ax_circ.set_title(
        "Circular Connectome (All 64 Nodes Separated Without Overlap)\n"
        "Vibrant Shortest Paths Highlight Dijkstra Information Flow to Hubs",
        color="#f8fafc",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )

    # Circular Layout Coordinates (Equidistant on radius R=1.0)
    R = 1.0
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    # Start at top (pi/2) and go clockwise for intuitive reading
    angles = np.pi / 2 - angles
    pos_circ = {i: np.array([R * np.cos(a), R * np.sin(a)]) for i, a in enumerate(angles)}

    # 1. Subtle background backbone edges (light, curved, non-intrusive)
    nx.draw_networkx_edges(
        G,
        pos_circ,
        edgelist=strong_edges,
        ax=ax_circ,
        edge_color="#334155",
        alpha=0.18,
        width=0.8,
        connectionstyle="arc3,rad=0.12",
        arrows=False,
    )

    # 2. Highlighted Shortest Paths (Neon Crimson & Glowing Cyan)
    nx.draw_networkx_edges(
        G,
        pos_circ,
        edgelist=list(flow_edges),
        ax=ax_circ,
        edge_color="#f43f5e",
        alpha=0.95,
        width=3.2,
        connectionstyle="arc3,rad=0.18",
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
    )

    # 3. Node rendering
    centrality_vals = [G.nodes[i]["degree_centrality"] for i in range(num_nodes)]
    node_sizes = [160 + 550 * c for c in centrality_vals]
    cmap = plt.cm.plasma

    scatter = ax_circ.scatter(
        [pos_circ[i][0] for i in range(num_nodes)],
        [pos_circ[i][1] for i in range(num_nodes)],
        c=centrality_vals,
        cmap=cmap,
        s=node_sizes,
        edgecolors="#1e293b",
        linewidths=1.5,
        zorder=5,
    )

    # 4. Highlight Top-5 Hubs with Golden Halo Rings
    for h in top_hubs:
        hx, hy = pos_circ[h]
        halo = plt.Circle(
            (hx, hy),
            0.052,
            color="#fbbf24",
            fill=False,
            linewidth=2.8,
            linestyle="-",
            zorder=6,
        )
        ax_circ.add_patch(halo)

    # 5. Non-overlapping Radial Labels for ALL 64 Nodes
    label_radius = 1.14
    for i in range(num_nodes):
        x, y = pos_circ[i]
        angle = np.arctan2(y, x)
        lx = label_radius * np.cos(angle)
        ly = label_radius * np.sin(angle)

        ha = "left" if x >= 0.01 else ("right" if x <= -0.01 else "center")
        va = "bottom" if y >= 0.01 else ("top" if y <= -0.01 else "center")

        is_hub = (i in top_hubs)
        color = "#fbbf24" if is_hub else "#cbd5e1"
        weight = "bold" if is_hub else "normal"
        fontsize = 8.5 if is_hub else 7.0

        label_text = f"#{i} {CHANNEL_NAMES[i]}"
        if is_hub:
            label_text = f"★ #{i} {CHANNEL_NAMES[i]}"

        ax_circ.text(
            lx, ly, label_text,
            ha=ha, va=va,
            color=color,
            fontsize=fontsize,
            fontweight=weight,
            fontfamily="sans-serif",
            zorder=7,
        )

    ax_circ.set_xlim(-1.42, 1.42)
    ax_circ.set_ylim(-1.42, 1.42)
    ax_circ.set_aspect("equal")
    ax_circ.axis("off")

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax_circ, orientation="horizontal", pad=0.03, shrink=0.55)
    cbar.set_label("Node Degree Centrality ($k_i / (N-1)$)", color="#e2e8f0", fontsize=11, labelpad=6)
    cbar.ax.xaxis.set_tick_params(color="#94a3b8", labelcolor="#e2e8f0")

    # -------------------------------------------------------------
    # Panel 2: Anatomical 10-20 Scalp Topography & Metrics Box
    # -------------------------------------------------------------
    ax_scalp = fig.add_subplot(gs[1], facecolor="#0b0f19")
    ax_scalp.set_title(
        "Anatomical 10-20 Scalp Topography & Cortical Signal Routing",
        color="#f8fafc",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )

    pos_scalp = get_scalp_positions()
    draw_head_outline(ax_scalp, radius=0.98, color="#334155")

    # Draw shortest paths across the cortical surface
    nx.draw_networkx_edges(
        G,
        pos_scalp,
        edgelist=list(flow_edges),
        ax=ax_scalp,
        edge_color="#38bdf8",
        alpha=0.90,
        width=2.8,
        connectionstyle="arc3,rad=0.14",
        arrows=True,
        arrowsize=16,
        arrowstyle="-|>",
    )

    # Draw nodes on scalp
    ax_scalp.scatter(
        [pos_scalp[i][0] for i in range(num_nodes)],
        [pos_scalp[i][1] for i in range(num_nodes)],
        c=centrality_vals,
        cmap=cmap,
        s=[120 + 350 * c for c in centrality_vals],
        edgecolors="#0f172a",
        linewidths=1.2,
        zorder=5,
    )

    # Highlight top hubs on scalp
    for h in top_hubs:
        hx, hy = pos_scalp[h]
        ax_scalp.add_patch(
            plt.Circle((hx, hy), 0.050, color="#fbbf24", fill=False, linewidth=2.5, zorder=6)
        )
        ax_scalp.text(
            hx, hy + 0.065, f"{CHANNEL_NAMES[h]} (Hub)",
            ha="center", va="bottom",
            color="#fef08a", fontsize=8.5, fontweight="bold",
            zorder=7,
        )

    # Label key landmark electrodes
    landmarks = [21, 23, 10, 60, 62, 40, 41, 43]
    for lm in landmarks:
        if lm not in top_hubs:
            lx, ly = pos_scalp[lm]
            ax_scalp.text(
                lx, ly - 0.055, CHANNEL_NAMES[lm],
                ha="center", va="top",
                color="#94a3b8", fontsize=7.0,
                zorder=7,
            )

    ax_scalp.set_xlim(-1.30, 1.30)
    ax_scalp.set_ylim(-1.30, 1.30)
    ax_scalp.set_aspect("equal")
    ax_scalp.axis("off")

    # -------------------------------------------------------------
    # Information & Routing Legend Box below Scalp
    # -------------------------------------------------------------
    top_hubs_desc = "\n".join([
        f"  ★ #{int(row['node_id']):02d} {CHANNEL_NAMES[int(row['node_id'])]} | Out-Deg: {int(row['degree']):2d} | Centrality: {row['degree_centrality']:.4f} | Avg Path: {row['avg_shortest_path']:.3f}"
        for _, row in sorted_df.head(5).iterrows()
    ])

    flow_desc = "\n".join([
        f"  • {' -> '.join(CHANNEL_NAMES[n] for n in p)} (Hops: {len(p)-1})"
        for p in flow_paths[:5]
    ])

    info_text = (
        "TOP MOTOR-CORTEX HUBS (PREMOTOR & SENSORIMOTOR CORE)\n"
        "──────────────────────────────────────────────────────────\n"
        f"{top_hubs_desc}\n\n"
        "KEY DIJKSTRA INFORMATION FLOW PATHS\n"
        "──────────────────────────────────────────────────────────\n"
        f"{flow_desc}\n\n"
        "VISUALIZATION KEY:\n"
        "• Left Panel: 64 circular nodes with zero overlap; every channel labeled radially.\n"
        "• Right Panel: Anatomical 10-20 sensor placement on scalp.\n"
        "• Pink/Cyan Bold Curves: Shortest paths transmitting neural signals."
    )

    ax_scalp.text(
        0.50, -0.22, info_text,
        transform=ax_scalp.transAxes,
        fontsize=9.2, fontfamily="monospace",
        color="#f1f5f9",
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#475569", alpha=0.95),
    )

    # Save high-res figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"High-resolution, non-overlapping visualization saved to: {output_path}")


def main():
    project_dir = Path(__file__).resolve().parent
    output_image = project_dir / "neurograph_visualization.png"

    print("Loading EEG network data...")
    G, metrics_df, adj_matrix = load_dataset(project_dir, threshold=0.5)

    print("Generating non-overlapping dual-panel visualization...")
    generate_visualization(G, metrics_df, adj_matrix, output_image)


if __name__ == "__main__":
    main()
