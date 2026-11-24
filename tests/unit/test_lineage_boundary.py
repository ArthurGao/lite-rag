"""The ledger stays generic (lineage design §5.4): it imports nothing else of lite-rag, so it can
move to its own repository unchanged."""

import ast
from pathlib import Path

import lite_rag.lineage

PACKAGE = Path(lite_rag.lineage.__file__).parent


def _imported(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # a relative import can climb out of the package: none are allowed
            names.append((node.module or "") if node.level == 0 else f"<relative:{node.module}>")
    return names


def test_lineage_imports_no_other_lite_rag_module() -> None:
    sources = sorted(PACKAGE.rglob("*.py"))
    assert sources  # the walk found the package
    offending = [
        f"{path.relative_to(PACKAGE)}: {name}"
        for path in sources
        for name in _imported(ast.parse(path.read_text("utf-8")))
        if name.startswith("<relative")
        or (name.split(".")[0] == "lite_rag" and name.split(".")[:2] != ["lite_rag", "lineage"])
    ]
    assert offending == []
