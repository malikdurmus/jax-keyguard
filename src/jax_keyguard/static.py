"""Static detection of JAX PRNG key reuse in Python source, without running it."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
import pathlib
import re
import sys
import tokenize

from jax_keyguard import __version__

CODE = "KG001"

# jax.random functions that return a fresh key (or array of keys).
_PRODUCERS = frozenset({"key", "PRNGKey", "split", "fold_in", "clone", "wrap_key_data"})
# jax.random functions that don't consume a key passed as their first argument.
_NON_CONSUMING = (_PRODUCERS - {"split"}) | {"key_data", "key_impl", "key_dtype", "Key"}
# Public jax.random functions, to resolve `from jax.random import *`.
_RANDOM_FUNCTIONS = _NON_CONSUMING | frozenset(
    "ball bernoulli beta binomial bits categorical cauchy chisquare choice dirichlet double_sided_maxwell "
    "exponential f gamma generalized_normal geometric gumbel laplace loggamma logistic lognormal maxwell "
    "multinomial multivariate_normal normal orthogonal pareto permutation poisson rademacher randint "
    "rayleigh split t triangular truncated_normal uniform wald weibull_min".split()
)
# Loops are re-analyzed until the state stops changing; this caps the passes.
_MAX_PASSES = 8

_TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Same syntax as flake8's inline noqa.
_NOQA = re.compile(r"# noqa(?::[\s]?(?P<codes>([A-Z]+[0-9]+(?:[,\s]+)?)+))?", re.IGNORECASE)

# A state maps each tracked variable to (group, use). Variables in the same group
# hold the same key (aliases); use is None while the key is unused, else
# (function name, (line, col)) of its first use.


def _merge(*states):
    """Combine branch states; a key counts as consumed if any branch consumed it."""
    live = [s for s in states if s is not None]
    if not live:
        return None
    merged = {}
    for state in live:
        for name, value in state.items():
            if name not in merged or (merged[name][1] is None and value[1] is not None):
                merged[name] = value
    return merged


def _consume(state, group, use):
    for name, (g, u) in list(state.items()):
        if g == group and u is None:
            state[name] = (g, use)


def _local_nodes(body):
    """Nodes in a scope's body, without descending into nested functions, classes or lambdas."""
    todo = list(body)
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (*_SCOPES, ast.Lambda)):
            todo.extend(ast.iter_child_nodes(node))


def _imports(nodes):
    """Map names bound by imports to what they import, e.g. {'jr': 'jax.random'}."""
    nodes = [n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))]
    aliases = {}
    for node in nodes:  # star imports first, so explicit imports win
        if isinstance(node, ast.ImportFrom) and node.module == "jax.random" and not node.level:
            if any(a.name == "*" for a in node.names):
                aliases.update({name: f"jax.random.{name}" for name in _RANDOM_FUNCTIONS})
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                aliases[a.asname or top] = a.name if a.asname else top
        elif node.module and not node.level:
            for a in node.names:
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def _pattern_names(pattern):
    for node in ast.walk(pattern):
        for name in (getattr(node, "name", None), getattr(node, "rest", None)):
            if isinstance(name, str):
                yield name


def _irrefutable(pattern):
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _irrefutable(pattern.pattern)
    if isinstance(pattern, ast.MatchOr):
        return any(_irrefutable(p) for p in pattern.patterns)
    return False


def _site(node):
    return (node.lineno, node.col_offset)


class _Checker:
    def __init__(self, findings):
        self.findings = findings
        self.aliases = {}
        self.loops = []  # per enclosing loop: (states at break, states at continue)
        self.exits = []  # per enclosing try/with: (kind, state) where control may leave it
        self.exit_log = []  # every recorded exit, so memoized results can replay them
        self.memo = {}
        self.scopes_done = set()

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
        if isinstance(node, ast.IfExp):
            return self.is_producer(node.body) and self.is_producer(node.orelse)
        return isinstance(node, ast.Call) and self.random_fn(node) in _PRODUCERS

    @staticmethod
    def key_arg(call):
        if call.args:
            return call.args[0]
        return next((kw.value for kw in call.keywords if kw.arg == "key"), None)

    def record(self, kind, state):
        entry = (kind, dict(state))
        self.exit_log.append(entry)
        for frame in self.exits:
            frame.append(entry)

    def memoized(self, node, state, compute):
        # Loops and comprehensions are analyzed several times per nesting level;
        # caching by input state keeps that from growing exponentially.
        key = (id(node), frozenset(state.items()), bool(self.exits))
        if key in self.memo:
            result, log = self.memo[key]
            for kind, s in log:
                self.record(kind, s)
        else:
            mark = len(self.exit_log)
            result = compute()
            self.memo[key] = (result, self.exit_log[mark:])
        return None if result is None else dict(result)

    # Scopes

    def check_scope(self, body, params=()):
        nodes = list(_local_nodes(body))
        outer = self.aliases, self.loops, self.exits
        self.aliases = {k: v for k, v in self.aliases.items() if k not in params}
        self.aliases.update(_imports(nodes))
        self.loops, self.exits = [], []
        # A parameter is a key if the scope passes it to jax.random somewhere.
        passed = set()
        for node in nodes:
            if isinstance(node, ast.Call) and self.random_fn(node):
                arg = self.key_arg(node)
                if isinstance(arg, ast.Name):
                    passed.add(arg.id)
        self.block(body, {p: (("param", p), None) for p in params if p in passed})
        self.aliases, self.loops, self.exits = outer

    def function_scope(self, node, args, body, state):
        # Defaults are evaluated in the enclosing scope, the body on its own.
        for default in args.defaults + [d for d in args.kw_defaults if d is not None]:
            self.expr(default, state)
        if id(node) in self.scopes_done:
            return
        self.scopes_done.add(id(node))
        params = [p.arg for p in args.posonlyargs + args.args + args.kwonlyargs]
        params += [p.arg for p in (args.vararg, args.kwarg) if p is not None]
        self.check_scope(body, params)

    def nested_scope(self, node, state):
        for decorator in node.decorator_list:
            self.expr(decorator, state)
        if isinstance(node, ast.ClassDef):
            for base in node.bases + [kw.value for kw in node.keywords]:
                self.expr(base, state)
            if id(node) not in self.scopes_done:
                self.scopes_done.add(id(node))
                self.check_scope(node.body)
        else:
            self.function_scope(node, node.args, node.body, state)

    # Statements

    def block(self, stmts, state):
        for stmt in stmts:
            if state is None:
                return None
            if self.exits:
                self.record("raise", state)  # an exception may leave before this statement finishes
            state = self.stmt(stmt, state)
        return state

    def stmt(self, node, state):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                self.expr(node.value, state)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.bind_value(target, node.value, state, _site(node))
        elif isinstance(node, ast.AugAssign):
            self.expr(node.value, state)
            self.drop([node.target], state)
        elif isinstance(node, _TERMINATORS):
            for child in ast.iter_child_nodes(node):
                self.expr(child, state)
            state = dict(state)
            if self.loops and isinstance(node, ast.Break):
                self.loops[-1][0].append(state)
            elif self.loops and isinstance(node, ast.Continue):
                self.loops[-1][1].append(state)
            if self.exits:
                self.record("raise" if isinstance(node, ast.Raise) else "exit", state)
            return None
        elif isinstance(node, ast.If):
            self.expr(node.test, state)
            return _merge(self.block(node.body, dict(state)), self.block(node.orelse, dict(state)))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return self.memoized(node, state, lambda: self.loop(node, state))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            return self.with_(node, state)
        elif isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            return self.try_(node, state)
        elif isinstance(node, ast.Match):
            return self.match(node, state)
        elif isinstance(node, _SCOPES):
            self.nested_scope(node, state)
            state.pop(node.name, None)
        elif isinstance(node, ast.Delete):
            self.drop(node.targets, state)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expr(child, state)
        return state

    def loop(self, node, state):
        # Re-run the body until the state at the loop head stops changing, so a key
        # consumed in one iteration and not reassigned is reported when a later
        # iteration consumes it again.
        is_while = isinstance(node, ast.While)
        if not is_while:
            self.expr(node.iter, state)
        endless = is_while and isinstance(node.test, ast.Constant) and bool(node.test.value)
        entry, exits, breaks, outs, links = state, [], [], [], []
        for _ in range(_MAX_PASSES):
            head = dict(entry)
            if is_while:
                self.expr(node.test, head)
            exits.append(head if is_while else entry)
            if not is_while:
                links = self.bind_iter(node.target, node.iter, head)
            frame = ([], [])
            self.loops.append(frame)
            out = self.block(node.body, dict(head))
            self.loops.pop()
            outs.append(out)
            breaks += frame[0]
            new = _merge(state, out, *frame[1]) or dict(state)
            if new == entry:
                break
            entry = new
        last = dict(entry)
        if is_while:
            self.expr(node.test, last)
        exits.append(last)
        result = _merge(self.block(node.orelse, None if endless else _merge(*exits)), *breaks)
        self.mark_arrays(links, outs, result)
        return result

    def mark_arrays(self, links, states, result):
        """Mark key arrays whose elements were consumed while iterating over them."""
        if result is None:
            return
        for name, group, array in links:
            use = next((s[name][1] for s in states if s and s.get(name, (None,))[0] == group and s[name][1]), None)
            if use and array in result and result[array][1] is None:
                _consume(result, result[array][0], use)

    def with_(self, node, state):
        for item in node.items:
            self.expr(item.context_expr, state)
            if item.optional_vars is not None:
                self.bind_value(item.optional_vars, item.context_expr, state, _site(item.optional_vars))
        frame = []
        self.exits.append(frame)
        body = self.block(node.body, state)
        self.exits.pop()
        if body is not None:
            return body
        # The body always leaves early, but a context manager such as
        # contextlib.suppress or pytest.raises may swallow the exception.
        return _merge(*(s for kind, s in frame if kind == "raise"))

    def try_(self, node, state):
        loop_frame = None
        if node.finalbody and self.loops:
            # break/continue inside the try run `finally` before leaving.
            loop_frame, self.loops[-1] = self.loops[-1], ([], [])
        body_exits = []
        self.exits.append(body_exits)
        body = self.block(node.body, dict(state))
        self.exits.pop()
        # An exception can leave the body before any statement, so handlers start
        # from everything the body may have consumed by then.
        entry = _merge(state, body, *(s for kind, s in body_exits if kind == "raise"))
        other_exits = []
        self.exits.append(other_exits)
        handlers = []
        for handler in node.handlers:
            s = dict(entry)
            if handler.name:
                s.pop(handler.name, None)
            handlers.append(self.block(handler.body, s))
        orelse = self.block(node.orelse, dict(body)) if body is not None else None
        self.exits.pop()
        after = _merge(orelse, *handlers)
        if not node.finalbody:
            return after
        early = [s for _, s in body_exits + other_exits]
        final = self.block(node.finalbody, _merge(after, *early) or dict(entry))
        if loop_frame is not None:
            jumps, self.loops[-1] = self.loops[-1], loop_frame
            for states, outer in zip(jumps, loop_frame):
                for s in states:
                    out = self.block(node.finalbody, dict(s))
                    if out is not None:
                        outer.append(out)
        return final if after is not None else None

    def match(self, node, state):
        # Cases are tried in order, so a guard's consumption carries to later cases.
        self.expr(node.subject, state)
        current, outs = state, []
        for case in node.cases:
            s = dict(current)
            for name in _pattern_names(case.pattern):
                s.pop(name, None)
            if case.guard is not None:
                self.expr(case.guard, s)
            outs.append(self.block(case.body, dict(s)))
            if case.guard is None and _irrefutable(case.pattern):
                current = None  # later cases and "no match" are unreachable
                break
            current = s
        return _merge(current, *outs)

    # Bindings

    def bind_value(self, target, value, state, group, source=None):
        """Bind `target = value`: a fresh key, an alias of a tracked key, or untracked.

        Names in `value` are looked up in `source` (default: `state`), so that
        `a, b = b, a` reads both names before assigning either.
        """
        if source is None:
            source = state
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and not any(isinstance(e, ast.Starred) for e in target.elts + value.elts)
        ):
            source = dict(source)
            for i, (t, v) in enumerate(zip(target.elts, value.elts)):
                self.bind_value(t, v, state, group + (i,), source)
        elif self.is_producer(value):
            self.fresh(target, state, group)
        elif isinstance(target, ast.Name) and isinstance(value, ast.Name) and value.id in source:
            state[target.id] = source[value.id]
        else:
            self.drop([target], state)

    def fresh(self, target, state, group):
        if isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                self.fresh(elt, state, group + (i,))
        elif isinstance(target, ast.Starred):
            self.fresh(target.value, state, group + ("*",))
        elif isinstance(target, ast.Name):
            state[target.id] = (group, None)
        else:
            self.expr(target, state)  # e.g. a subscript: evaluate the index

    def drop(self, targets, state):
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                self.drop(target.elts, state)
            elif isinstance(target, ast.Starred):
                self.drop([target.value], state)
            elif isinstance(target, ast.Name):
                state.pop(target.id, None)
            else:
                self.expr(target, state)

    def bind_iter(self, target, it, state):
        """Bind a loop target. Returns (name, group, array) links for key-array elements."""
        if self.is_producer(it):
            self.fresh(target, state, _site(target))
        elif isinstance(it, ast.Name) and it.id in state and isinstance(target, ast.Name):
            # Elements of a key array: fresh, unless an earlier loop consumed them.
            group = ("elem", *_site(target))
            state[target.id] = (group, state[it.id][1])
            return [(target.id, group, it.id)]
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and isinstance(target, ast.Tuple):
            elts, args = target.elts, it.args
            if it.func.id == "enumerate" and len(elts) == 2 and args:
                self.drop([elts[0]], state)
                return self.bind_iter(elts[1], args[0], state)
            if it.func.id == "zip" and len(elts) == len(args):
                return [link for t, a in zip(elts, args) for link in self.bind_iter(t, a, state)]
            self.drop([target], state)
        else:
            self.drop([target], state)
        return []

    # Expressions

    def expr(self, node, state):
        if isinstance(node, ast.Lambda):
            self.function_scope(node, node.args, [ast.Expr(node.body)], state)
            return
        if isinstance(node, _COMPREHENSIONS):
            self.expr(node.generators[0].iter, state)
            state.update(self.memoized(node, state, lambda: self.comprehension(node, dict(state))))
            return
        if isinstance(node, ast.NamedExpr):
            self.expr(node.value, state)
            self.bind_value(node.target, node.value, state, _site(node))
            return
        if isinstance(node, ast.IfExp):
            self.expr(node.test, state)
            body, orelse = dict(state), dict(state)
            self.expr(node.body, body)
            self.expr(node.orelse, orelse)
            state.update(_merge(body, orelse))
            return
        if isinstance(node, ast.BoolOp):
            # Operands after the first may be skipped (short-circuit).
            self.expr(node.values[0], state)
            outs, s = [dict(state)], state
            for value in node.values[1:]:
                s = dict(s)
                self.expr(value, s)
                outs.append(s)
            state.update(_merge(*outs))
            return
        for child in ast.iter_child_nodes(node):
            self.expr(child, state)
        if isinstance(node, ast.Call):
            self.call(node, state)

    def comprehension(self, node, inner):
        """Return the outer state after evaluating a comprehension."""
        outer = dict(inner)
        local = {n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
        # A walrus inside a comprehension binds in the enclosing scope.
        walrus = {n.target.id for n in ast.walk(node) if isinstance(n, ast.NamedExpr)} - local
        parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        states = []
        for _ in range(_MAX_PASSES):
            before = dict(inner)
            links = []
            for i, g in enumerate(node.generators):
                if i:
                    self.expr(g.iter, inner)
                links += self.bind_iter(g.target, g.iter, inner)
                for cond in g.ifs:
                    self.expr(cond, inner)
            for part in parts:
                self.expr(part, inner)
            states.append(dict(inner))
            if inner == before:
                break
        result = {name: (value if name in local else inner.get(name, value)) for name, value in outer.items()}
        result.update({name: inner[name] for name in walrus if name in inner})
        self.mark_arrays(links, states, result)
        return result

    def call(self, node, state):
        fn = self.random_fn(node)
        if fn is None or fn in _NON_CONSUMING:
            return
        arg = self.key_arg(node)
        if not isinstance(arg, ast.Name) or arg.id not in state:
            return
        here = _site(arg)
        group, first = state[arg.id]
        if first is None:
            _consume(state, group, (fn, here))
            return
        first_fn, first_pos = first
        where = "in a previous iteration" if first_pos == here else f"on line {first_pos[0]}"
        self.findings.add((
            *here,
            f"{CODE} PRNG key '{arg.id}' reused: already consumed by jax.random.{first_fn} {where}; split it first",
        ))


def check_source(source: str, filename: str = "<string>") -> list[tuple[int, int, str]]:
    """Return sorted (line, col, message) findings for the given source.

    Raises SyntaxError if the source doesn't parse, RecursionError if it is too
    deeply nested to analyze.
    """
    with _deep_recursion():  # Python 3.11's parser also counts against the limit
        tree = ast.parse(source, filename)
    return check_tree(tree, re.split(r"\r\n|\r|\n", source))


@contextlib.contextmanager
def _deep_recursion():
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10_000))
    try:
        yield
    finally:
        sys.setrecursionlimit(limit)


def check_tree(tree: ast.Module, lines=None) -> list[tuple[int, int, str]]:
    """Findings for a parsed module. Pass the source lines to get character columns."""
    imported = _imports(ast.walk(tree)).values()
    if not any(q == "jax" or q.startswith("jax.") for q in imported):
        return []
    findings: set = set()
    with _deep_recursion():
        _Checker(findings).check_scope(tree.body)
    if lines is not None:
        # ast columns are UTF-8 byte offsets.
        findings = {
            (line, len(lines[line - 1].encode()[:col].decode("utf-8", "ignore")), msg)
            for line, col, msg in findings
        }
    return sorted(findings)


class Plugin:
    """flake8 plugin entry point."""

    name = "jax-keyguard"
    version = __version__

    def __init__(self, tree, lines):
        self.tree = tree
        self.lines = [line.rstrip("\r\n") for line in lines]

    def run(self):
        try:
            findings = check_tree(self.tree, self.lines)
        except RecursionError:
            findings = [(1, 0, "KG000 too deeply nested to check for PRNG key reuse")]
        for line, col, msg in findings:
            yield line, col, msg, type(self)


def _noqa_lines(source):
    """Lines whose comment suppresses KG001, following flake8's rules."""
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            match = tok.type == tokenize.COMMENT and _NOQA.search(tok.string)
            if match:
                codes = match.group("codes")
                codes = [c for c in re.split(r"[,\s]+", codes) if c] if codes else []
                if not codes or CODE in codes or CODE.startswith(tuple(codes)):
                    lines.add(tok.start[0])
    except (tokenize.TokenError, SyntaxError):
        pass
    return lines


def _python_files(paths, errors):
    for path in paths:
        if path.is_dir():
            for root, dirs, names in os.walk(path, onerror=errors.append):
                dirs.sort()
                yield from (pathlib.Path(root, n) for n in sorted(names) if n.endswith(".py"))
        else:
            yield path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Find JAX PRNG key reuse in Python files.")
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    walk_errors = []
    files = {}
    for f in _python_files(args.paths, walk_errors):
        files.setdefault(f.resolve(), f)
    for e in walk_errors:
        print(f"{e.filename}: error: {e.strerror}", file=sys.stderr)
    found, errors = 0, len(walk_errors)
    for path in files.values():
        try:
            with tokenize.open(path) as f:  # honors coding cookies and BOMs
                source = f.read()
            findings = check_source(source, str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            print(f"{path}: error: {e}", file=sys.stderr)
            errors += 1
            continue
        except RecursionError:
            print(f"{path}: error: too deeply nested to check", file=sys.stderr)
            errors += 1
            continue
        noqa = _noqa_lines(source)
        for line, col, msg in findings:
            if line not in noqa:
                print(f"{path}:{line}:{col + 1}: {msg}")
                found += 1
    return 2 if errors else 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
