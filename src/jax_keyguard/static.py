"""Static detection of JAX PRNG key reuse in Python source, without running it."""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

CODE = "KG001"

# jax.random functions that return a fresh key (or array of keys).
_PRODUCERS = frozenset({"key", "PRNGKey", "split", "fold_in", "clone", "wrap_key_data"})
# jax.random functions that take a key without consuming it.
_NON_CONSUMING = frozenset({"key", "PRNGKey", "fold_in", "clone", "key_data", "key_impl", "wrap_key_data"})

_TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _aliases(tree: ast.Module) -> dict[str, str]:
    """Map local names to the qualified names they import, e.g. {'jr': 'jax.random'}."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def _merge(*states):
    """Combine branch states; a key counts as consumed if any branch consumed it."""
    live = [s for s in states if s is not None]
    if not live:
        return None
    merged = {}
    for state in live:
        for name, use in state.items():
            if merged.get(name) is None:
                merged[name] = use
    return merged


class _Checker:
    def __init__(self, aliases, findings):
        self.aliases = aliases
        self.findings = findings

    def qualname(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(self.aliases.get(node.id, node.id))
        return ".".join(reversed(parts))

    def random_fn(self, call):
        """Name of the jax.random function called, or None."""
        qual = self.qualname(call.func)
        if qual and qual.startswith("jax.random.") and qual.count(".") == 2:
            return qual.rsplit(".", 1)[1]
        return None

    def is_producer(self, node):
        if isinstance(node, ast.Subscript):
            node = node.value
        return isinstance(node, ast.Call) and self.random_fn(node) in _PRODUCERS

    @staticmethod
    def key_arg(call):
        if call.args:
            return call.args[0]
        return next((kw.value for kw in call.keywords if kw.arg == "key"), None)

    # Scopes

    def check_scope(self, body, params=()):
        # A parameter is a key if the scope passes it to jax.random somewhere.
        passed = set()
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call) and self.random_fn(node):
                arg = self.key_arg(node)
                if isinstance(arg, ast.Name):
                    passed.add(arg.id)
        self.block(body, {p: None for p in params if p in passed})

    def nested_scope(self, node):
        if isinstance(node, ast.ClassDef):
            self.check_scope(node.body)
        else:
            a = node.args
            params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
            self.check_scope(node.body, params)

    # Statements

    def block(self, stmts, state):
        for stmt in stmts:
            if state is None:
                return None
            state = self.stmt(stmt, state)
        return state

    def stmt(self, node, state):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if node.value is not None:
                self.expr(node.value, state)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            fresh = isinstance(node, ast.Assign) and self.is_producer(node.value)
            self.bind(targets, state, fresh)
        elif isinstance(node, _TERMINATORS):
            for child in ast.iter_child_nodes(node):
                self.expr(child, state)
            return None
        elif isinstance(node, ast.If):
            self.expr(node.test, state)
            return _merge(self.block(node.body, dict(state)), self.block(node.orelse, dict(state)))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return self.loop(node, state)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.expr(item.context_expr, state)
            return self.block(node.body, state)
        elif isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            body = self.block(node.body, dict(state))
            entry = _merge(state, body)
            handlers = [self.block(h.body, dict(entry)) for h in node.handlers]
            orelse = self.block(node.orelse, body) if body is not None else None
            after = _merge(orelse if node.orelse else body, *handlers)
            return self.block(node.finalbody, after if after is not None else dict(entry))
        elif isinstance(node, ast.Match):
            self.expr(node.subject, state)
            return _merge(state, *(self.block(c.body, dict(state)) for c in node.cases))
        elif isinstance(node, _SCOPES):
            self.nested_scope(node)
            state.pop(node.name, None)
        elif isinstance(node, ast.Delete):
            self.bind(node.targets, state, fresh=False)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expr(child, state)
        return state

    def loop(self, node, state):
        # Run the body twice so a key consumed in one iteration and not
        # reassigned is reported when the next iteration consumes it again.
        head = node.test if isinstance(node, ast.While) else node.iter
        self.expr(head, state)
        fresh = not isinstance(node, ast.While) and self.is_producer(node.iter)
        entry, outs = state, []
        for _ in range(2):
            entry = dict(entry)
            if not isinstance(node, ast.While):
                self.bind([node.target], entry, fresh)
            out = self.block(node.body, entry)
            outs.append(out)
            entry = _merge(state, out) or dict(state)
        return self.block(node.orelse, _merge(state, *outs))

    def bind(self, targets, state, fresh):
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                self.bind(target.elts, state, fresh)
            elif isinstance(target, ast.Starred):
                self.bind([target.value], state, fresh)
            elif isinstance(target, ast.Name):
                if fresh:
                    state[target.id] = None
                else:
                    state.pop(target.id, None)

    # Expressions

    def expr(self, node, state):
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, _COMPREHENSIONS):
            self.expr(node.generators[0].iter, state)
            # Comprehension variables are local and shadow outer names.
            inner = dict(state)
            parts = []
            for i, g in enumerate(node.generators):
                if i:
                    parts.append(g.iter)
                self.bind([g.target], inner, self.is_producer(g.iter))
                parts += g.ifs
            parts += [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for _ in range(2):  # evaluated once per element
                for part in parts:
                    self.expr(part, inner)
            local = {n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
            for name in state:
                if name in inner and name not in local:
                    state[name] = inner[name]
            return
        if isinstance(node, ast.IfExp):
            self.expr(node.test, state)
            body, orelse = dict(state), dict(state)
            self.expr(node.body, body)
            self.expr(node.orelse, orelse)
            state.update(_merge(body, orelse))
            return
        for child in ast.iter_child_nodes(node):
            self.expr(child, state)
        if isinstance(node, ast.Call):
            self.call(node, state)

    def call(self, node, state):
        fn = self.random_fn(node)
        if fn is None or fn in _NON_CONSUMING:
            return
        arg = self.key_arg(node)
        if not isinstance(arg, ast.Name) or arg.id not in state:
            return
        here = (arg.lineno, arg.col_offset)
        first = state[arg.id]
        if first is None:
            state[arg.id] = (fn, here)
            return
        first_fn, first_pos = first
        where = "in a previous iteration" if first_pos == here else f"on line {first_pos[0]}"
        self.findings.add((
            *here,
            f"{CODE} PRNG key '{arg.id}' reused: already consumed by jax.random.{first_fn} {where}; split it first",
        ))


def check_source(source: str, filename: str = "<string>") -> list[tuple[int, int, str]]:
    """Return sorted (line, col, message) findings for the given source."""
    return check_tree(ast.parse(source, filename))


def check_tree(tree: ast.Module) -> list[tuple[int, int, str]]:
    aliases = _aliases(tree)
    if not any(q == "jax" or q.startswith("jax.") for q in aliases.values()):
        return []
    findings: set = set()
    _Checker(aliases, findings).check_scope(tree.body)
    return sorted(findings)


class Plugin:
    """flake8 plugin entry point."""

    name = "jax-keyguard"
    version = "0.2.0"

    def __init__(self, tree):
        self.tree = tree

    def run(self):
        for line, col, msg in check_tree(self.tree):
            yield line, col, msg, type(self)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Find JAX PRNG key reuse in Python files.")
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    files = [f for p in args.paths for f in (sorted(p.rglob("*.py")) if p.is_dir() else [p])]
    count = 0
    for path in files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for line, col, msg in check_source(source, str(path)):
            if "# noqa" not in lines[line - 1]:
                print(f"{path}:{line}:{col + 1}: {msg}")
                count += 1
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
