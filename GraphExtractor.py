import ast
import json
import os
import sys
from pathlib import Path

import networkx as nx

# Directories we never want to treat as source code
SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    ".git", "__pycache__", ".tox", ".pytest_cache",
    "node_modules", "build", "dist", ".egg-info", ".mypy_cache",
}


class RepoGraphBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.graph = nx.DiGraph()
        self.module_map = {}      # dotted module name -> file path (str)
        self.func_owner = {}      # "module.Class.method" or "module.func" -> file path

    # ---------- pass 1: discover files + build module name map ----------

    def discover(self):
        py_files = []
        for p in self.repo_root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            py_files.append(p)

        for path in py_files:
            rel = path.relative_to(self.repo_root)
            parts = rel.with_suffix("").parts
            # Fix __init__.py:  pkg/__init__  ->  pkg
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

    # ---------- pass 2: parse each file, add import + call edges ----------

    def build(self, py_files):
        # First pass: register all symbols (functions, methods, classes)
        for path in py_files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            self._register_symbols(tree, str(path))

        # Second pass: resolve imports and calls now that the symbol table is complete
        for path in py_files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            self._add_import_edges(tree, str(path))
            self._add_call_edges(tree, str(path))

        return self.graph

    # ---------- symbol registration ----------

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

    # ---------- import edges ----------

    def _add_import_edges(self, tree, file_path):
        mod_name = self.graph.nodes[file_path]["module"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # a.b.c -> link to a, a.b, a.b.c if they exist in repo
                    self._link_module_hierarchy(file_path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                abs_module = self._resolve_relative(node, mod_name)
                if not abs_module:
                    continue
                # Link to the module itself
                self._link_module(file_path, abs_module)
                # Try to link to specific imported names (functions/classes)
                for alias in node.names:
                    target = f"{abs_module}.{alias.name}"
                    if target in self.func_owner:
                        self._add_typed_edge(file_path, target, "imports")

    def _resolve_relative(self, node: ast.ImportFrom, current_module: str):
        """Convert relative import to absolute dotted name."""
        if node.level == 0:
            return node.module
        if not current_module:
            return None
        parts = current_module.split(".")
        level = node.level
        if level > len(parts):
            return None  # beyond package root
        base = ".".join(parts[:-level]) if level < len(parts) else ""
        if node.module:
            return f"{base}.{node.module}" if base else node.module
        return base or None

    def _link_module_hierarchy(self, file_path, dotted_name):
        """Link to the longest prefix that exists in our repo."""
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

    # ---------- call edges (func -> func) ----------

    def _add_call_edges(self, tree, file_path):
        mod_name = self.graph.nodes[file_path]["module"]
        # Walk class-aware so methods get their qualified caller id
        # (mod.Class.method), not just mod.method — ast.walk() alone loses
        # the class prefix and silently drops every method's call edges.
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
            return  # nested/unregistered func; skip for prototype
        for child in ast.walk(func_node):
            if isinstance(child, ast.Call):
                callee = self._resolve_call(child, mod_name)
                if callee and callee != caller:
                    self._add_typed_edge(caller, callee, "calls")

    def _resolve_call(self, call_node: ast.Call, caller_module: str):
        """
        Best-effort resolution of a Call node to a fully-qualified function name.
        Handles:
            foo()           -> same module
            self.bar()      -> same module (class method heuristic)
            module.baz()    -> cross-module
            ClassName.qux() -> same module class method
        """
        func = call_node.func

        # Simple name: foo()
        if isinstance(func, ast.Name):
            return self._resolve_name(func.id, caller_module)

        # Attribute chain: a.b.c()
        if isinstance(func, ast.Attribute):
            chain = self._attr_chain(func)
            if not chain:
                return None

            # self.method()  ->  look in same module for any Class.method
            if chain[0] == "self" and len(chain) == 2:
                method = chain[1]
                matches = [
                    f for f in self.func_owner
                    if f.startswith(f"{caller_module}.") and f.endswith(f".{method}")
                ]
                if len(matches) == 1:
                    return matches[0]
                return None

            # module.name()  or  ClassName.method()
            if len(chain) == 2:
                first, second = chain
                # Try as module.func
                candidate = f"{first}.{second}"
                if candidate in self.func_owner:
                    return candidate
                # Try as current_module.ClassName.method
                candidate = f"{caller_module}.{first}.{second}"
                if candidate in self.func_owner:
                    return candidate
                # Try resolving first as module alias
                real_mod = self.module_map.get(first)
                if real_mod:
                    real_mod_name = self.graph.nodes[real_mod]["module"]
                    candidate = f"{real_mod_name}.{second}"
                    if candidate in self.func_owner:
                        return candidate

            # Deeper chain: module.submodule.func()
            if len(chain) >= 2:
                # Try longest prefix as module
                for i in range(len(chain) - 1, 0, -1):
                    mod_candidate = ".".join(chain[:i])
                    func_candidate = ".".join(chain[i:])
                    # absolute
                    fq = f"{mod_candidate}.{func_candidate}"
                    if fq in self.func_owner:
                        return fq
                    # via module_map alias
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
        """Flatten an Attribute chain into a list of names."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return list(reversed(parts))
        return None  # too complex (subscript, call, etc.)

    # ---------- utilities ----------

    def _add_typed_edge(self, u, v, edge_type):
        if self.graph.has_edge(u, v):
            existing = self.graph[u][v].get("type")
            types = existing if isinstance(existing, list) else [existing]
            if edge_type not in types:
                types.append(edge_type)
            self.graph[u][v]["type"] = types
        else:
            self.graph.add_edge(u, v, type=edge_type)

    # ---------- export ----------

    def to_json(self):
        data = nx.node_link_data(self.graph, edges="edges")
        return json.dumps(data, indent=2)


def main():
    if len(sys.argv) != 3:
        print("Usage: python graph_extractor.py <repo_root> <output.json>")
        sys.exit(1)

    repo_root, out_path = sys.argv[1], sys.argv[2]
    builder = RepoGraphBuilder(repo_root)
    py_files = builder.discover()
    builder.build(py_files)

    with open(out_path, "w") as f:
        f.write(builder.to_json())

    print(f"Parsed {len(py_files)} files")
    print(f"Nodes: {builder.graph.number_of_nodes()}  Edges: {builder.graph.number_of_edges()}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()