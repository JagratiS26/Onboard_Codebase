import argparse
import ast
import json
import sys
from pathlib import Path

import networkx as nx

SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    ".git", "__pycache__", ".tox", ".pytest_cache",
    "node_modules", "build", "dist", ".egg-info", ".mypy_cache",
}

NODE_COLORS = {"file": "#4CAF50", "class": "#2196F3", "function": "#FF9800"}
EDGE_COLORS = {"defines": "#9E9E9E", "imports": "#03A9F4", "calls": "#F44336"}


class RepoGraphBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.graph = nx.DiGraph()
        self.module_map = {}
        self.func_owner = {}

    def discover(self):
        py_files = []
        for p in self.repo_root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            py_files.append(p)

        for path in py_files:
            rel = path.relative_to(self.repo_root)
            parts = rel.with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            mod_name = ".".join(parts) if parts else rel.parts[0]

            self.module_map[mod_name] = str(path)
            self.graph.add_node(
                str(path),
                type="file",
                module=mod_name,
                loc=self._count_lines(path),
            )
        return py_files

    @staticmethod
    def _count_lines(path: Path) -> int:
        try:
            return sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore"))
        except OSError:
            return 0

    def build(self, py_files):
        for path in py_files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            self._register_symbols(tree, str(path))

        for path in py_files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            self._add_import_edges(tree, str(path))
            self._add_call_edges(tree, str(path))

        return self.graph

    def _register_symbols(self, tree, file_path):
        mod_name = self.graph.nodes[file_path]["module"]

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_func_node(mod_name, node.name, file_path, node)
            elif isinstance(node, ast.ClassDef):
                self.graph.add_node(
                    f"{mod_name}.{node.name}",
                    type="class",
                    file=file_path,
                    lineno=node.lineno,
                    docstring=(ast.get_docstring(node) or "")[:200],
                )
                self._add_typed_edge(file_path, f"{mod_name}.{node.name}", "defines")
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._add_func_node(
                            mod_name, f"{node.name}.{child.name}", file_path, child
                        )

    def _add_func_node(self, mod_name, qualname, file_path, node):
        fq_name = f"{mod_name}.{qualname}"
        self.func_owner[fq_name] = file_path
        self.graph.add_node(
            fq_name,
            type="function",
            file=file_path,
            lineno=node.lineno,
            docstring=(ast.get_docstring(node) or "")[:200],
        )
        self._add_typed_edge(file_path, fq_name, "defines")

    def _add_import_edges(self, tree, file_path):
        mod_name = self.graph.nodes[file_path]["module"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._link_module_hierarchy(file_path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                abs_module = self._resolve_relative(node, mod_name)
                if not abs_module:
                    continue
                self._link_module(file_path, abs_module)
                for alias in node.names:
                    target = f"{abs_module}.{alias.name}"
                    if target in self.func_owner:
                        self._add_typed_edge(file_path, target, "imports")

    def _resolve_relative(self, node: ast.ImportFrom, current_module: str):
        if node.level == 0:
            return node.module
        if not current_module:
            return None
        parts = current_module.split(".")
        level = node.level
        if level > len(parts):
            return None
        base = ".".join(parts[:-level]) if level < len(parts) else ""
        if node.module:
            return f"{base}.{node.module}" if base else node.module
        return base or None

    def _link_module_hierarchy(self, file_path, dotted_name):
        parts = dotted_name.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if self._link_module(file_path, prefix):
                break

    def _link_module(self, file_path, mod_name) -> bool:
        target = self.module_map.get(mod_name)
        if target and target != file_path:
            self._add_typed_edge(file_path, target, "imports")
            return True
        return False

    def _add_call_edges(self, tree, file_path):
        mod_name = self.graph.nodes[file_path]["module"]
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = f"{mod_name}.{node.name}"
                self._scan_calls_in(node, caller, mod_name)
            elif isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        caller = f"{mod_name}.{node.name}.{child.name}"
                        self._scan_calls_in(child, caller, mod_name)

    def _scan_calls_in(self, func_node, caller, mod_name):
        if caller not in self.func_owner:
            return
        for child in ast.walk(func_node):
            if isinstance(child, ast.Call):
                callee = self._resolve_call(child, mod_name)
                if callee and callee != caller:
                    self._add_typed_edge(caller, callee, "calls")

    def _resolve_call(self, call_node: ast.Call, caller_module: str):
        func = call_node.func

        if isinstance(func, ast.Name):
            return self._resolve_name(func.id, caller_module)
        if isinstance(func, ast.Attribute):
            chain = self._attr_chain(func)
            if not chain:
                return None
            if chain[0] == "self" and len(chain) == 2:
                method = chain[1]
                matches = [
                    f for f in self.func_owner
                    if f.startswith(f"{caller_module}.") and f.endswith(f".{method}")
                ]
                if len(matches) == 1:
                    return matches[0]
                return None
            if len(chain) == 2:
                first, second = chain
                candidate = f"{first}.{second}"
                if candidate in self.func_owner:
                    return candidate
                candidate = f"{caller_module}.{first}.{second}"
                if candidate in self.func_owner:
                    return candidate
                real_mod = self.module_map.get(first)
                if real_mod:
                    real_mod_name = self.graph.nodes[real_mod]["module"]
                    candidate = f"{real_mod_name}.{second}"
                    if candidate in self.func_owner:
                        return candidate

            if len(chain) >= 2:
                for i in range(len(chain) - 1, 0, -1):
                    mod_candidate = ".".join(chain[:i])
                    func_candidate = ".".join(chain[i:])
                    fq = f"{mod_candidate}.{func_candidate}"
                    if fq in self.func_owner:
                        return fq
                    real_mod = self.module_map.get(mod_candidate)
                    if real_mod:
                        real_mod_name = self.graph.nodes[real_mod]["module"]
                        fq = f"{real_mod_name}.{func_candidate}"
                        if fq in self.func_owner:
                            return fq
        return None

    def _resolve_name(self, name: str, caller_module: str):
        candidate = f"{caller_module}.{name}"
        if candidate in self.func_owner:
            return candidate
        matches = [f for f in self.func_owner if f.endswith(f".{name}")]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _attr_chain(node: ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return list(reversed(parts))
        return None

    def _add_typed_edge(self, u, v, edge_type):
        if self.graph.has_edge(u, v):
            existing = self.graph[u][v].get("type")
            types = existing if isinstance(existing, list) else [existing]
            if edge_type not in types:
                types.append(edge_type)
            self.graph[u][v]["type"] = types
        else:
            self.graph.add_edge(u, v, type=edge_type)

    def to_json(self):
        data = nx.node_link_data(self.graph, edges="edges")
        return json.dumps(data, indent=2)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _node_label(self, node, attrs):
        if attrs.get("type") == "file":
            return Path(node).name
        return node.split(".")[-1]

    def visualize_interactive(self, output_path="graph.html", physics=True):
        """
        Interactive, zoomable/draggable HTML graph via pyvis.
        Best option for exploring anything beyond a handful of nodes.
        Requires: pip install pyvis
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("pyvis is not installed. Run: pip install pyvis")
            return None

        net = Network(
            height="900px",
            width="100%",
            directed=True,
            notebook=False,
            bgcolor="#1e1e1e",
            font_color="white",
        )

        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            color = NODE_COLORS.get(node_type, "#9E9E9E")
            label = self._node_label(node, attrs)
            title = f"{node}\nType: {node_type}"
            if attrs.get("docstring"):
                title += f"\n{attrs['docstring']}"
            size = 25 if node_type == "file" else (18 if node_type == "class" else 12)
            net.add_node(node, label=label, color=color, title=title, size=size)

        for u, v, attrs in self.graph.edges(data=True):
            etype = attrs.get("type", "unknown")
            etypes = etype if isinstance(etype, list) else [etype]
            color = EDGE_COLORS.get(etypes[0], "#CCCCCC")
            net.add_edge(u, v, color=color, title=", ".join(etypes), arrows="to")

        net.set_options(f"""
        {{
          "physics": {{
            "enabled": {str(physics).lower()},
            "barnesHut": {{
              "gravitationalConstant": -8000,
              "springLength": 120,
              "springConstant": 0.04
            }},
            "minVelocity": 0.75
          }},
          "interaction": {{"hover": true, "tooltipDelay": 100}}
        }}
        """)

        net.write_html(output_path, notebook=False)
        print(f"Interactive graph written to {output_path} "
              f"(open it in a browser)")
        return output_path

    def visualize_static(self, output_path="graph.png", figsize=(20, 16)):
        """
        Static PNG fallback via matplotlib, no extra dependency beyond
        matplotlib itself. Gets cluttered fast on large repos.
        """
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.graph, k=0.6, seed=42, iterations=50)
        fig, ax = plt.subplots(figsize=figsize)

        for node_type, color in NODE_COLORS.items():
            nodes = [n for n, a in self.graph.nodes(data=True) if a.get("type") == node_type]
            nx.draw_networkx_nodes(
                self.graph, pos, nodelist=nodes, node_color=color,
                node_size=300 if node_type == "file" else 150,
                label=node_type, ax=ax,
            )

        for edge_type, color in EDGE_COLORS.items():
            edges = [
                (u, v) for u, v, a in self.graph.edges(data=True)
                if edge_type in (a["type"] if isinstance(a["type"], list) else [a["type"]])
            ]
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=edges, edge_color=color,
                alpha=0.5, arrows=True, arrowsize=8, ax=ax,
            )

        labels = {n: self._node_label(n, a) for n, a in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=6, ax=ax)

        ax.legend(scatterpoints=1)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Static graph written to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract a call/import graph from a Python repo.")
    parser.add_argument("repo_root", help="Path to the repository to analyze")
    parser.add_argument("output_json", help="Path to write the graph JSON to")
    parser.add_argument(
        "--visualize", choices=["html", "png", "both"], default=None,
        help="Also render a visualization: interactive HTML (pyvis), static PNG (matplotlib), or both",
    )
    parser.add_argument(
        "--no-physics", action="store_true",
        help="Disable physics simulation in the HTML view (useful for large graphs)",
    )
    args = parser.parse_args()

    builder = RepoGraphBuilder(args.repo_root)
    py_files = builder.discover()
    builder.build(py_files)

    with open(args.output_json, "w") as f:
        f.write(builder.to_json())

    print(f"Parsed {len(py_files)} files")
    print(f"Nodes: {builder.graph.number_of_nodes()}  Edges: {builder.graph.number_of_edges()}")
    print(f"Written to {args.output_json}")

    if args.visualize in ("html", "both"):
        out = Path(args.output_json).with_suffix(".html")
        builder.visualize_interactive(str(out), physics=not args.no_physics)

    if args.visualize in ("png", "both"):
        out = Path(args.output_json).with_suffix(".png")
        builder.visualize_static(str(out))


if __name__ == "__main__":
    main()