"""故事工坊与流水线必须能被分开解释。

这是个**架构断言**，不是功能断言：`app/agent/` 里的任何一行都不许 import
`app/graph/` 里的东西，反过来也一样。理由不是洁癖——

  工作流的控制流全在代码里，agent 的控制流在模型里。一旦两边互相 import，
  「这段行为是谁决定的」就再也没有单一答案：读 agent 的人得先把 7 个节点
  读完，改一个节点的人可能悄悄改掉了 agent 的行为。

判断方式用 `ast` 而不是正则：`app/agent/__init__.py` 的 docstring 里**写着
这条规则本身**，用 `"app.graph" in source` 去查会把自己绊倒。这里只走
Import / ImportFrom 两种节点，散文里的提及一律不算。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"


def _modules_of(source: str, package: str) -> set[str]:
    """源码里真正 import 了哪些模块（相对导入按 package 补全成绝对路径）。"""
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                if node.module:
                    out.add(node.module)
                continue
            # 在 app/graph/nodes/ 里 `from ..bus import x` → app.graph.bus
            parts = package.split(".")
            base = parts[: len(parts) - node.level + 1]
            out.add(".".join([*base, node.module] if node.module else base))
    return out


def _package_of(path: Path) -> str:
    return ".".join(path.relative_to(APP.parent).with_suffix("").parts[:-1])


def _modules(path: Path) -> set[str]:
    return _modules_of(path.read_text(encoding="utf-8"), _package_of(path))


def _sources(package: str) -> list[Path]:
    files = sorted((APP / package).rglob("*.py"))
    assert files, f"没找到 app/{package} 下的任何源码——路径变了的话这条测试就成了空断言"
    return files


def _hits(package: str, forbidden: str) -> list[str]:
    """`package` 下所有源码里 import 了 `forbidden`（或它的子模块）的地方。"""
    bad: list[str] = []
    for path in _sources(package):
        for module in sorted(_modules(path)):
            if module == forbidden or module.startswith(f"{forbidden}."):
                bad.append(f"{path.name} → {module}")
    return bad


@pytest.mark.parametrize(
    "package,forbidden",
    [
        ("agent", "app.graph"),
        ("graph", "app.agent"),
        # 对话工坊是**第三条**控制流（多角色圆桌），与另外两条互相隔离的理由
        # 更硬：一旦 `app/scene/` 能 import `app/graph/` 或 `app/agent/`，
        # 「这句台词是谁决定的」就会有两套可能的答案。三对全查，缺一对就等于
        # 给那条路开了个后门——而 `_hits` 只会静静返回空列表。
        ("scene", "app.graph"),
        ("graph", "app.scene"),
        ("scene", "app.agent"),
        ("agent", "app.scene"),
    ],
)
def test_两边互不import(package: str, forbidden: str):
    assert _hits(package, forbidden) == []


def test_判据本身是有效的():
    """正向对照，两件事：

    1. **相对导入确实被解析成了绝对路径**（否则 `from ..bus import x`
       这种写法能从两条隔离断言底下走过去）；
    2. 同一个函数跑在 `app/graph/` 上**查得到东西**——说明它不是个永远
       返回空集的花架子。没有这一条，上面那条测试在函数写坏时会变成
       永远为真的空断言，而那种绿比红更危险。
    """
    # `package` 是**所在包**，与 Python 自己的相对导入规则一致：
    # 在 app/graph/nodes/ 里 `from ..bus import x` 就是 app.graph.bus
    assert _modules_of("from ..bus import x\n", "app.graph.nodes") == {"app.graph.bus"}
    assert _modules_of("from . import events\n", "app.agent") == {"app.agent"}
    assert _modules_of("import app.graph.bus\n", "app.agent") == {"app.graph.bus"}

    assert _hits("graph", "app.graph") != []
    assert _hits("agent", "app.agent") != []
    assert _hits("scene", "app.scene") != []
