"""Validate source documentation, authorship, syntax, and core dependencies."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import ast
import io
import json
import sys
import tokenize
from importlib.util import resolve_name
from pathlib import Path


def python_files(root: Path):
    """Yield maintained Python files without inspecting caches or generated builds."""
    for directory in ("src", "benchmarks", "tests", "tools"):
        yield from sorted((root / directory).rglob("*.py"))


def core_dependencies(path: Path, tree: ast.Module) -> list[str]:
    """Reject dependencies from domain or application code to concrete adapters."""
    if path.parts[:2] != ("src", "exactus") or len(path.parts) < 4:
        return []
    layer = path.parts[2]
    if layer not in ("domain", "application"):
        return []
    package = ".".join(path.parts[1:-1])
    allowed = ("exactus.domain",)
    if layer == "application":
        allowed += ("exactus.application",)
    errors = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            modules = [module]
        for module in modules:
            if module.startswith("exactus"):
                if not any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in allowed
                ):
                    errors.append(f"{path}:{node.lineno}: {layer} depends on {module}")
            elif module.partition(".")[0] not in sys.stdlib_module_names:
                errors.append(
                    f"{path}:{node.lineno}: {layer} imports nonstandard module {module}"
                )
    return errors


def validate_file(path: Path, root: Path) -> tuple[dict, list[str]]:
    """Validate one module without executing code or importing optional runtimes."""
    relative = path.relative_to(root)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(relative), feature_version=(3, 10))
        compile(tree, str(relative), "exec")
    except (SyntaxError, ValueError) as exc:
        return {}, [f"{relative}: {exc}"]
    errors = core_dependencies(relative, tree)
    classes = functions = 0
    if not ast.get_docstring(tree):
        errors.append(f"{relative}: missing module docstring")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
        else:
            continue
        if not ast.get_docstring(node):
            errors.append(
                f"{relative}:{node.lineno}: missing docstring for {node.name}"
            )
    authors = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__author__"
            for target in node.targets
        )
    ]
    if (
        len(authors) != 1
        or not isinstance(authors[0], ast.Constant)
        or authors[0].value != "Olshansky Grigory"
    ):
        errors.append(f"{relative}: expected one Olshansky Grigory author declaration")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            errors.append(
                f"{relative}:{token.start[0]}: use a docstring instead of a comment"
            )
    return {
        "path": str(relative),
        "lines": len(source.splitlines()),
        "classes": classes,
        "functions": functions,
    }, errors


def main() -> int:
    """Report validation failures and large-module candidates without running models."""
    root = Path(__file__).resolve().parents[1]
    results, errors = [], []
    for path in python_files(root):
        result, failures = validate_file(path, root)
        if result:
            results.append(result)
        errors.extend(failures)
    print(
        json.dumps(
            {
                "files": len(results),
                "classes": sum(row["classes"] for row in results),
                "functions": sum(row["functions"] for row in results),
                "errors": errors,
                "large_module_candidates": sorted(
                    (row for row in results if row["lines"] > 200),
                    key=lambda row: row["lines"],
                    reverse=True,
                ),
            },
            indent=2,
        )
    )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
