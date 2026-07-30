import sys
import sysconfig


def _is_define_invocation(argv: list[str]) -> bool:
    """True for ``veeksha define ...`` (generation-only, no server).

    Define only materializes the workload fingerprint; it does not need
    free-threaded concurrency. Authors can pin definitions on a normal
    interpreter when free-threaded wheels (e.g. tokenizers) are unavailable.
    """
    for arg in argv[1:]:
        if arg == "-m":
            continue
        return arg == "define"
    return False


def _require_free_threaded_python() -> None:
    """Exit with a clear error unless the interpreter is free-threaded.

    Exception: ``define`` is allowed on a GIL build so pins can be authored
    where free-threaded native wheels (tokenizers/transformers) fail.
    """
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    if _is_define_invocation(sys.argv):
        return
    sys.exit(
        "veeksha requires free-threaded Python (GIL disabled).\n"
        "Run with a free-threaded interpreter (e.g. python3.14t).\n"
        "Exception: ``veeksha define`` may run on a normal interpreter "
        "for pin authoring only."
    )


def main() -> None:
    _require_free_threaded_python()
    from veeksha.cli.commands import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
