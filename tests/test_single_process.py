"""单进程测试：演示脚本可导入、辅助函数正确、未初始化分布式时的报错路径。"""

import importlib.util
import pathlib
import sys

import pytest

DEMOS = pathlib.Path(__file__).resolve().parent.parent / "chapters" / "00-primitives" / "demos"


@pytest.mark.parametrize("script", sorted(p for p in DEMOS.glob("*.py") if not p.name.startswith("._")))
def test_demo_importable(script: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "main", None))


def test_sum_ranks_helper() -> None:
    spec = importlib.util.spec_from_file_location("demo_allreduce", DEMOS / "demo_allreduce.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.sum_ranks(1) == 1.0
    assert module.sum_ranks(4) == 10.0


def test_demo_requires_torchrun(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("demo_allreduce", DEMOS / "demo_allreduce.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["demo_allreduce"])
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(SystemExit, match="torchrun"):
        module.main()
