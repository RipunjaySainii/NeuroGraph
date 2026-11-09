"""
visualize_graph.py

Loads 'eeg_adjacency_matrix.csv' and 'topological_metrics.csv' from the NeuroGraph project.
Constructs a 64-node NetworkX directed graph, identifies top degree-centrality hubs,
calculates key shortest paths across the network, and creates an informative, publication-grade
visualization highlighting information flow and hub architecture using Matplotlib.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


def load_data(project_dir: Path, threshold: float = 0.5):
    """Load adjacency matrix and topological metrics from CSV files."""
    adj_path = project_dir / "eeg_adjacency_matrix.csv"
    metrics_path = project_dir / "topological_metrics.csv"

    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency matrix file not found: {adj_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    adj_matrix = np.loadtxt(adj_path, delimiter=",")
    metrics_df = pd.read_csv(metrics_path)

    # Build directed weighted graph
    G = nx.DiGraph()
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        row = metrics_df[metrics_df["node_id"] == i]
        deg_cent = row["degree_centrality"].values[0] if not row.empty else 0.0
        deg = row["degree"].values[0] if not row.empty else 0
        w_deg = row["weighted_degree"].values[0] if not row.empty else 0.0
        avg_sp = row["avg_shortest_path"].values[0] if not row.empty else 0.0
        closeness = row["closeness_centrality"].values[0] if not row.empty else 0.0

        G.add_node(
            i,
            degree_centrality=deg_cent,
            degree=deg,
            weighted_degree=w_deg,
            avg_shortest_path=avg_sp,
            closeness=closeness,
        )

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=float(adj_matrix[i, j]))

    return G, metrics_df, adj_matrix


def visualize_network(G: nx.DiGraph, metrics_df: pd.DataFrame, output_path: Path):
    """Generate high-resolution visualization of the graph topology and information flow."""
    # Compute reproducible spring layout
    pos = nx.spring_layout(G, seed=42, k=0.35, iterations=80)

    # Identify top hubs by degree centrality
    sorted_df = metrics_df.sort_values(by="degree_centrality", ascending=False)
    top_hubs = sorted_df.head(5)["node_id"].tolist()
    primary_hub = top_hubs[0]  # Highest degree centrality node (Node 4)

    # Select representative peripheral and target nodes to trace information flow
    peripheral_nodes = sorted_df[sorted_df["degree"] > 0].tail(4)["node_id"].tolist()
    distant_targets = [63, 60, 55, 18]

    # Calculate shortest paths demonstrating information flow through the network
    flow_paths = []
    # 1. Flow from peripheral nodes into the primary hub
    for p_node in peripheral_nodes:
        if nx.has_path(G, p_node, primary_hub):
            path = nx.shortest_path(G, source=p_node, target=primary_hub, weight="weight")
            flow_paths.append(path)

    # 2. Flow from primary hub out to distant target nodes
    for t_node in distant_targets:
        if nx.has_path(G, primary_hub, t_node):
            path = nx.shortest_path(G, source=primary_hub, target=t_node, weight="weight")
            flow_paths.append(path)

    # 3. Trans-network longest flow path
    if nx.has_path(G, 28, 63):
        flow_paths.append(nx.shortest_path(G, source=28, target=63, weight="weight"))

    # Extract all edges that participate in the highlighted shortest paths
    flow_edges = set()
    for path in flow_paths:
        for u, v in zip(path[:-1], path[1:]):
            flow_edges.add((u, v))

    background_edges = [e for e in G.edges() if e not in flow_edges]

    # Setup aesthetic dark/modern figure layout
    fig = plt.figure(figsize=(20, 10), facecolor="#0f172a")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 0.7], wspace=0.15)

    # Subplot 1: Main Network Graph
    ax_graph = fig.add_subplot(gs[0], facecolor="#0f172a")
    ax_graph.set_title(
        "NeuroGraph: 64-Node EEG Connectivity & Information Flow\n(PhysioNet Subject 1 - Motor Imagery)",
        color="#f8fafc",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    # Draw background edges (subdued)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=background_edges,
        ax=ax_graph,
        edge_color="#334155",
        alpha=0.12,
        width=0.6,
        arrows=False,
    )

    # Draw shortest path flow edges (vibrant crimson & neon gold arrows)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(flow_edges),
        ax=ax_graph,
        edge_color="#f43f5e",
        alpha=0.95,
        width=2.8,
        arrows=True,
        arrowsize=16,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
    )

    # Node styling by degree centrality
    centrality_values = [G.nodes[n]["degree_centrality"] for n in G.nodes()]
    node_sizes = [200 + 1200 * c for c in centrality_values]
    cmap = plt.cm.viridis

    # Draw normal nodes
    nodes_collection = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax_graph,
        node_color=centrality_values,
        cmap=cmap,
        node_size=node_sizes,
        alpha=0.92,
        edgecolors="#1e293b",
        linewidths=1.2,
    )

    # Highlight Top-5 Hub nodes with glowing halos
    hub_pos = {h: pos[h] for h in top_hubs}
    hub_sizes = [node_sizes[h] * 1.5 for h in top_hubs]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=top_hubs,
        ax=ax_graph,
        node_size=hub_sizes,
        node_color="none",
        edgecolors="#f59e0b",
        linewidths=3.0,
    )

    # Annotate Top Hubs with labels
    hub_labels = {
        h: f"Hub #{h}\n(k={metrics_df.loc[metrics_df['node_id'] == h, 'degree'].values[0]})"
        for h in top_hubs
    }
    nx.draw_networkx_labels(
        G,
        pos,
        labels=hub_labels,
        ax=ax_graph,
        font_size=9,
        font_color="#fef08a",
        font_weight="bold",
    )

    # Annotate disconnected / peripheral nodes
    labels_misc = {43: "Isolated #43", 37: "Periph #37"}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels_misc,
        ax=ax_graph,
        font_size=8,
        font_color="#94a3b8",
    )

    ax_graph.axis("off")

    # Add Colorbar for degree centrality
    cbar = plt.colorbar(nodes_collection, ax=ax_graph, orientation="horizontal", pad=0.04, shrink=0.6)
    cbar.set_label("Degree Centrality ($k_i / (N-1)$)", color="#e2e8f0", fontsize=11, labelpad=8)
    cbar.ax.xaxis.set_tick_params(color="#94a3b8", labelcolor="#e2e8f0")

    # Subplot 2: Metrics Dashboard & Information Flow Details
    ax_info = fig.add_subplot(gs[1], facecolor="#1e293b")
    ax_info.set_title("Network Analytics & Flow Routing", color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax_info.axis("off")

    # Build analytics text
    top_hubs_text = "\n".join([
        f"  #{int(row['node_id']):02d} | Out-Deg: {int(row['degree']):2d} | Centrality: {row['degree_centrality']:.4f} | Avg Path: {row['avg_shortest_path']:.3f}"
        for _, row in sorted_df.head(5).iterrows()
    ])

    sample_paths_text = "\n".join([
        f"  {' -> '.join(str(n) for n in p)}"
        for p in flow_paths[:6]
    ])

    summary_text = (
        "NETWORK OVERVIEW\n"
        "────────────────────────────────────────\n"
        f"• Total Electrodes (Nodes): {G.number_of_nodes()}\n"
        f"• Thresholded Directed Edges: {G.number_of_edges():,}\n"
        f"• Network Density: {nx.density(G):.3f}\n"
        f"• Primary Hub: Node #{primary_hub} (Centrality: {metrics_df.loc[primary_hub, 'degree_centrality']:.4f})\n\n"
        "TOP DEGREE CENTRALITY HUBS\n"
        "────────────────────────────────────────\n"
        f"{top_hubs_text}\n\n"
        "SAMPLE SHORTEST PATHS (INFORMATION FLOW)\n"
        "────────────────────────────────────────\n"
        f"{sample_paths_text}\n\n"
        "PATH HIGHLIGHTING LEGEND\n"
        "────────────────────────────────────────\n"
        "• Red Arrows (-->): Shortest paths computed by Dijkstra\n"
        "• Yellow Ring: Top Hubs orchestrating flow\n"
        "• Faint Gray Mesh: Background functional connections"
    )

    ax_info.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_info.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        color="#f1f5f9",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#0f172a", edgecolor="#475569", alpha=0.9),
    )

    # Save visualization to disk
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Visualization successfully generated and saved to: {output_path}")


def main():
    project_dir = Path(__file__).resolve().parent
    output_image = project_dir / "neurograph_visualization.png"

    print("Loading network data and metrics...")
    G, metrics_df, _ = load_data(project_dir, threshold=0.5)

    print(f"Graph initialized: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Generating network visualization with highlighted hubs and shortest paths...")
    visualize_network(G, metrics_df, output_image)


if __name__ == "__main__":
    main()
