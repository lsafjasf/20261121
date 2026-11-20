"""零依赖测试运行器：收集 tests/ 下所有 test_* 函数并执行。

环境没有 pytest 时使用，行为接近 pytest 的函数式用例：
断言失败或抛异常即记为失败，最后以非零退出码反馈 CI。
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
USER_SITE = "/home/hj/.local/lib/python3.14/site-packages"
if os.path.isdir(USER_SITE):
    sys.path.insert(0, USER_SITE)
sys.path.insert(0, ROOT)


def main() -> int:
    tests_dir = os.path.join(ROOT, "tests")
    sys.path.insert(0, tests_dir)
    import conftest  # noqa: F401  负责路径注入

    failures = []
    passed = 0
    for filename in sorted(os.listdir(tests_dir)):
        if not filename.startswith("test_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        module = importlib.import_module(module_name)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            test = getattr(module, name)
            if not callable(test):
                continue
            try:
                test()
                passed += 1
                print(f"PASS {module_name}.{name}")
            except Exception:  # noqa: BLE001
                failures.append((f"{module_name}.{name}", traceback.format_exc()))
                print(f"FAIL {module_name}.{name}")

    print("\n" + "=" * 60)
    print(f"通过 {passed} 项，失败 {len(failures)} 项")
    for name, trace in failures:
        print("\n" + "-" * 60)
        print(name)
        print(trace)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
