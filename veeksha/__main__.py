"""Veeksha CLI — unified entrypoint with subcommands and GIL-free execution."""

import importlib
import os
import sys

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "benchmark": ("veeksha.cli.benchmarks", "Run full benchmark suite"),
    "config": ("veeksha.cli.config", "Configuration utilities"),
}


def _ensure_gil_disabled() -> None:
    """Ensure the GIL is disabled on free-threaded Python (3.13t+).

    Veeksha requires GIL-free execution for concurrent dispatch/completion
    workers. On non-free-threaded builds this is a no-op. On free-threaded
    builds, re-execs with PYTHON_GIL=0 if needed, and fails hard if the
    GIL cannot be disabled.
    """
    if not hasattr(sys, "_is_gil_enabled"):
        return  # non-free-threaded build — GIL is always on, nothing to do
    if not sys._is_gil_enabled():
        return  # GIL already disabled
    if os.environ.get("PYTHON_GIL") == "0":
        # Already requested GIL=0 but a C-extension re-enabled it.
        print(
            "ERROR: Free-threaded Python detected but the GIL could not be "
            "disabled (a C-extension re-enabled it). Veeksha requires "
            "GIL-free execution. Run with: PYTHON_GIL=0 python -Xgil=0 ...",
            file=sys.stderr,
        )
        sys.exit(1)
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
