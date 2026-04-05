import sys
import sysconfig


def _require_free_threaded_python() -> None:
    """Exit with a clear error unless the interpreter is free-threaded."""
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        sys.exit(
            "veeksha requires free-threaded Python (GIL disabled).\n"
            "Run with a free-threaded interpreter (e.g. python3.14t)."
        )


def main() -> None:
    _require_free_threaded_python()
    from veeksha.cli.commands import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
