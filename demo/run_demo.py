"""run_demo.py — thin shim for `python demo/run_demo.py`.

The demo itself lives in the package at ``loopward/demo.py`` so it can also be
exposed as the ``loopward-demo`` console script. This file just forwards to it,
preserving the classic ``python demo/run_demo.py [--demo-delay N]`` invocation
(default delay 0.0, so it stays fast for CI).
"""

from loopward.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
