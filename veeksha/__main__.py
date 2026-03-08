import sysconfig
import sys

if not sysconfig.get_config_var("Py_GIL_DISABLED"):
    sys.exit(
        "veeksha requires free-threaded Python (GIL disabled).\n"
        "Run with a free-threaded interpreter (e.g. python3.14t)."
    )

from veeksha.cli.commands import main  # noqa: E402

if __name__ == "__main__":
    main()
