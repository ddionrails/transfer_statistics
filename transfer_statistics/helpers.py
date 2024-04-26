import sys
import traceback


def multiprocessing_wrapper(arguments) -> None:
    """Make whole stacktrace available in parent process."""
    exec_function = arguments[0]
    arguments = arguments[1:]
    try:
        exec_function(arguments)
    except BaseException as error:
        raise RuntimeError(
            f"Arguments: {arguments}"
            + "".join(traceback.format_exception(*sys.exc_info()))
        ) from error
