"""Veeksha CLI — unified entrypoint with subcommands and GIL-free execution."""

import importlib
import os
import sys

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "benchmark": ("veeksha.cli.benchmarks", "Run full benchmark suite"),
    "microbench": ("veeksha.microbench.runner", "Run simplified microbenchmarks"),
    "config": ("veeksha.cli.config", "Configuration utilities"),
}


def _ensure_gil_disabled() -> None:
    """Re-exec with the GIL disabled on free-threaded Python (3.13t+).

    Some C-extension modules (e.g. tokenizers) force the GIL back on at
    import time.  Setting PYTHON_GIL=0 keeps it off for the entire process.
    This is a no-op on regular (non-free-threaded) builds.
    """
    if not hasattr(sys, "_is_gil_enabled") or not sys._is_gil_enabled():
        return
    if os.environ.get("PYTHON_GIL") == "0":
        return  # already requested; a module re-enabled it — nothing more we can do
    os.environ["PYTHON_GIL"] = "0"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


def _print_usage() -> None:
    print("usage: veeksha <command> [<args>]\n")
    print("commands:")
    for name, (_, desc) in SUBCOMMANDS.items():
        print(f"  {name:<15} {desc}")
    print(f"\nRun 'veeksha <command> --help' for command-specific options.")


def main() -> None:
    _ensure_gil_disabled()

    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    if sys.argv[1] in ("-h", "--help"):
        _print_usage()
        sys.exit(0)

    subcmd = sys.argv[1]
    if subcmd not in SUBCOMMANDS:
        print(f"Unknown command: {subcmd}\n")
        _print_usage()
        sys.exit(1)

    module_path, _ = SUBCOMMANDS[subcmd]
    # Rewrite argv so the sub-parser sees e.g. "veeksha microbench" as prog name
    sys.argv = [f"veeksha {subcmd}"] + sys.argv[2:]
    importlib.import_module(module_path).main()


if __name__ == "__main__":
    main()
