"""
Neptune Graph Visualizer
========================
Converts AWS Neptune OpenCypher JSON responses (RETURN *)
into clean PNG graph images styled after AWS Graph Explorer.

Usage:
    from neptune_graph_visualizer import render_neptune_graph

    # From a Neptune JSON response dict
    render_neptune_graph(neptune_json, output_path="graph.png")

    # From a JSON file
    render_neptune_graph_from_file("response.json", output_path="graph.png")
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG generation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx


# ---------------------------------------------------------------------------
# Color palette – muted, professional tones inspired by AWS Graph Explorer
# ---------------------------------------------------------------------------
LABEL_COLORS: list[str] = [
    "#4A90D9",  # Steel blue
    "#50B88E",  # Teal green
    "#E8734A",  # Warm coral
    "#9B6FC3",  # Soft purple
    "#E6A544",  # Amber gold
    "#D95B7F",  # Rose pink
    "#5BC0BE",  # Cyan
    "#8B8B8B",  # Neutral gray
    "#6BB54A",  # Leaf green
    "#C75C5C",  # Muted red
    "#4ABBE8",  # Sky blue
    "#C49A44",  # Mustard
]

EDGE_COLOR = "#7A8A99"
EDGE_LABEL_COLOR = "#4A5568"
BG_COLOR = "#FAFBFC"
NODE_BORDER_COLOR_FACTOR = 0.75  # darken node color for border


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _darken(hex_color: str, factor: float = 0.75) -> str:
    """Return a darker shade of *hex_color*."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _truncate(text: str, max_len: int = 18) -> str:
    """Truncate text and add ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _pick_display_label(properties: dict, labels: list[str]) -> str:
    """Choose the best label to display inside a node."""
    # Prefer common human-readable keys
    for key in ("name", "title", "label", "code", "desc", "description", "username"):
        if key in properties:
            return _truncate(str(properties[key]))
    # Fall back to first property value
    if properties:
        first_val = next(iter(properties.values()))
        return _truncate(str(first_val))
    # Fall back to first label
    if labels:
        return _truncate(labels[0])
    return "?"


# ---------------------------------------------------------------------------
# Neptune JSON parser
# ---------------------------------------------------------------------------

def parse_neptune_json(data: dict[str, Any]) -> tuple[dict, dict]:
    """
    Parse a Neptune OpenCypher JSON response and extract nodes and edges.

    Supports the standard Neptune format where each result row may contain
    multiple variables, each being a node or relationship entity.

    Also supports the `RETURN *` pattern where nodes AND relationships
    may appear under different variable names in the same row.

    Returns
    -------
    nodes : dict
        Mapping of node_id -> {labels, properties}
    edges : dict
        Mapping of edge_id -> {start, end, type, properties}
    """
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    results = data.get("results", [])

    for row in results:
        for _var_name, entity in row.items():
            if not isinstance(entity, dict):
                continue  # skip scalar values (count, etc.)

            entity_type = entity.get("~entityType", "")

            if entity_type == "node":
                node_id = str(entity.get("~id", ""))
                if node_id and node_id not in nodes:
                    nodes[node_id] = {
                        "labels": entity.get("~labels", []),
                        "properties": entity.get("~properties", {}),
                    }

            elif entity_type == "relationship":
                edge_id = str(entity.get("~id", ""))
                if edge_id and edge_id not in edges:
                    edges[edge_id] = {
                        "start": str(entity.get("~start", "")),
                        "end": str(entity.get("~end", "")),
                        "type": entity.get("~type", ""),
                        "properties": entity.get("~properties", entity.get("properties", {})),
                    }

    return nodes, edges


# ---------------------------------------------------------------------------
# Graph renderer
# ---------------------------------------------------------------------------

def render_neptune_graph(
    neptune_json: dict[str, Any],
    output_path: str | Path = "graph.png",
    *,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int = 150,
    show_edge_labels: bool = True,
    show_properties: bool = False,
    node_size: int = 2200,
    font_size: int = 9,
) -> Path:
    """
    Render a Neptune OpenCypher JSON response as a clean PNG graph image.

    Parameters
    ----------
    neptune_json : dict
        The raw JSON response from Neptune (must contain a "results" key).
    output_path : str or Path
        Where to save the PNG image.
    title : str, optional
        Optional title drawn at the top of the image.
    figsize : tuple, optional
        Figure size in inches (width, height). Auto-calculated if None.
    dpi : int
        Dots per inch for the output image.
    show_edge_labels : bool
        Whether to draw relationship type labels on edges.
    show_properties : bool
        Whether to display node properties as sub-labels.
    node_size : int
        Base size for nodes.
    font_size : int
        Font size for node labels.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    nodes, edges = parse_neptune_json(neptune_json)

    if not nodes and not edges:
        raise ValueError("No graph entities found in the Neptune JSON response.")

    # -- Build NetworkX graph -----------------------------------------------
    G = nx.DiGraph()

    for node_id, info in nodes.items():
        G.add_node(node_id, **info)

    for _edge_id, info in edges.items():
        src, dst = info["start"], info["end"]
        # Ensure source/target nodes exist (they may not if the query
        # only returned relationships via RETURN r)
        if src not in G:
            G.add_node(src, labels=[], properties={})
            nodes[src] = {"labels": [], "properties": {}}
        if dst not in G:
            G.add_node(dst, labels=[], properties={})
            nodes[dst] = {"labels": [], "properties": {}}
        G.add_edge(src, dst, rel_type=info["type"], **info.get("properties", {}))

    # -- Assign colors by label --------------------------------------------
    label_color_map: dict[str, str] = {}
    color_idx = 0
    for node_id in G.nodes():
        node_labels = nodes.get(node_id, {}).get("labels", [])
        primary = node_labels[0] if node_labels else "__default__"
        if primary not in label_color_map:
            label_color_map[primary] = LABEL_COLORS[color_idx % len(LABEL_COLORS)]
            color_idx += 1

    node_colors = []
    node_edge_colors = []
    for node_id in G.nodes():
        nlabels = nodes.get(node_id, {}).get("labels", [])
        primary = nlabels[0] if nlabels else "__default__"
        c = label_color_map[primary]
        node_colors.append(c)
        node_edge_colors.append(_darken(c, NODE_BORDER_COLOR_FACTOR))

    # -- Layout -------------------------------------------------------------
    n = G.number_of_nodes()
    if figsize is None:
        side = max(9, min(5 + n * 1.3, 26))
        figsize = (side, side * 0.75)

    if n <= 2:
        pos = nx.spring_layout(G, seed=42, k=4.0, iterations=100)
    elif n <= 20:
        pos = nx.spring_layout(G, seed=42, k=4.5 / math.sqrt(n), iterations=150)
    else:
        pos = nx.kamada_kawai_layout(G)
        # Scale positions outward slightly so large graphs don't cluster
        pos = {k: v * 1.2 for k, v in pos.items()}

    # -- Draw ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=EDGE_COLOR,
        width=1.8,
        alpha=0.6,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=22,
        min_target_margin=22,
    )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=2.2,
        alpha=0.92,
    )

    # Node labels
    display_labels = {}
    for node_id in G.nodes():
        info = nodes.get(node_id, {})
        props = info.get("properties", {})
        lbls = info.get("labels", [])
        main = _pick_display_label(props, lbls)

        if show_properties and props:
            prop_lines = [f"{k}: {_truncate(str(v), 20)}" for k, v in list(props.items())[:3]]
            main = main + "\n" + "\n".join(prop_lines)

        display_labels[node_id] = main

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=display_labels,
        font_size=font_size,
        font_color="#1A202C",
        font_weight="bold",
        font_family="sans-serif",
    )

    # Edge labels (drawn last so they are never hidden under nodes)
    # Custom placement: for each edge, find the position along the edge
    # that is farthest from all node centers to avoid overlap.
    if show_edge_labels:
        import numpy as np

        node_positions = np.array([pos[n] for n in G.nodes()])
        # Approximate node radius in data coords from node_size
        # node_size is in points^2; convert to rough data-coord radius
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_w = xlim[1] - xlim[0]
        data_h = ylim[1] - ylim[0]
        fig_w, fig_h = fig.get_size_inches()
        pts_per_data = (fig_w * fig.dpi) / data_w if data_w > 0 else 1
        node_radius_data = math.sqrt(node_size) / pts_per_data * 0.6

        for u, v, d in G.edges(data=True):
            rel_type = d.get("rel_type", "")
            if not rel_type:
                continue

            p_src = np.array(pos[u])
            p_dst = np.array(pos[v])

            # Sample candidate positions along the edge (0.2 to 0.8)
            best_t = 0.5
            best_min_dist = -1.0
            for t in [i / 20.0 for i in range(4, 17)]:  # 0.20 to 0.80
                candidate = (1 - t) * p_src + t * p_dst
                # Minimum distance from this candidate to any node center
                dists = np.linalg.norm(node_positions - candidate, axis=1)
                min_dist = float(np.min(dists))
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_t = t

            label_pt = (1 - best_t) * p_src + best_t * p_dst

            # Compute rotation angle to align with the edge
            dx = p_dst[0] - p_src[0]
            dy = p_dst[1] - p_src[1]
            angle = math.degrees(math.atan2(dy, dx))
            # Keep text right-side up
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            ax.text(
                label_pt[0], label_pt[1], rel_type,
                fontsize=font_size - 3,
                color=EDGE_LABEL_COLOR,
                fontfamily="sans-serif",
                ha="center", va="center",
                rotation=angle,
                rotation_mode="anchor",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.9),
                zorder=10,
            )

    # -- Legend (label → color) ---------------------------------------------
    legend_handles = []
    for label_name, color in label_color_map.items():
        display = label_name if label_name != "__default__" else "Unknown"
        patch = mpatches.Patch(color=color, label=display)
        legend_handles.append(patch)

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(legend_handles), 6),
            frameon=True,
            fontsize=font_size,
            title="Node Types",
            title_fontsize=font_size + 1,
            facecolor="white",
            edgecolor="#E2E8F0",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(1.0)

    # -- Title --------------------------------------------------------------
    if title:
        ax.set_title(
            title,
            fontsize=font_size + 4,
            fontweight="bold",
            color="#2D3748",
            pad=20,
            fontfamily="sans-serif",
        )

    plt.tight_layout(pad=1.5)
    output_path = Path(output_path).resolve()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# File-based convenience wrapper
# ---------------------------------------------------------------------------

def render_neptune_graph_from_file(
    json_path: str | Path,
    output_path: str | Path = "graph.png",
    **kwargs,
) -> Path:
    """Load Neptune JSON from a file and render it as a graph PNG."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return render_neptune_graph(data, output_path=output_path, **kwargs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render an AWS Neptune OpenCypher JSON response as a graph PNG."
    )
    parser.add_argument("input", help="Path to the Neptune JSON response file.")
    parser.add_argument(
        "-o", "--output", default="graph.png",
        help="Output PNG file path (default: graph.png)."
    )
    parser.add_argument("--title", default=None, help="Title for the graph image.")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150).")
    parser.add_argument(
        "--no-edge-labels", action="store_true",
        help="Hide relationship type labels on edges."
    )
    parser.add_argument(
        "--show-properties", action="store_true",
        help="Display node properties as sub-labels."
    )

    args = parser.parse_args()

    result = render_neptune_graph_from_file(
        args.input,
        output_path=args.output,
        title=args.title,
        dpi=args.dpi,
        show_edge_labels=not args.no_edge_labels,
        show_properties=args.show_properties,
    )
    print(f"Graph saved to: {result}")
