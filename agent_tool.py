"""
Agent Tool Wrapper for Neptune Graph Visualizer
================================================
Drop-in tool function for an AI Agent system.

This module exposes a single `visualize_neptune_graph` function
that your agent can call as a tool. It accepts the raw Neptune
JSON (as a dict or JSON string) and returns the path to the
generated PNG image.

Integration example (LangChain-style):
    from agent_tool import visualize_neptune_graph

    tools = [
        Tool(
            name="visualize_graph",
            func=visualize_neptune_graph,
            description="Generates a PNG image of a graph from Neptune query results.",
        )
    ]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neptune_graph_visualizer import render_neptune_graph


def visualize_neptune_graph(
    neptune_response: dict | str,
    output_dir: str | Path = ".",
    alert_id: int | str = 0,
    title: str | None = None,
    show_properties: bool = False,
) -> str:
    """
    Agent-callable tool: render a Neptune query response as a PNG graph.

    Parameters
    ----------
    neptune_response : dict or str
        The Neptune OpenCypher JSON response. Can be a Python dict
        or a JSON string.
    output_dir : str or Path
        Directory where the PNG will be saved. Defaults to current dir.
    alert_id : int or str
        Alert identifier. The output file will be named
        ``{alert_id}_graph.png`` and the default title will be
        ``{alert_id} graph``.
    title : str, optional
        Optional title for the graph image. If None, defaults to
        ``{alert_id} graph``.
    show_properties : bool
        If True, display node properties as sub-labels.

    Returns
    -------
    str
        Absolute path to the generated PNG file.
    """
    # Parse string input if needed
    if isinstance(neptune_response, str):
        neptune_response = json.loads(neptune_response)

    # Generate output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{alert_id}_graph.png"
    output_path = output_dir / filename

    # Default title from alert_id
    if title is None:
        title = f"{alert_id} graph"

    # Render
    result_path = render_neptune_graph(
        neptune_response,
        output_path=output_path,
        title=title,
        show_properties=show_properties,
    )

    return str(result_path)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = Path(__file__).parent / "sample_data.json"
    if sample.exists():
        with open(sample, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = visualize_neptune_graph(
            neptune_response=data,
            output_dir="./output",
            alert_id=987789,
        )
        print(f"✓ Graph image saved to: {path}")
    else:
        print("No sample_data.json found. Provide Neptune JSON as input.")

