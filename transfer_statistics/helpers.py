import sys
import traceback

from transfer_statistics.types import GeneralArguments, SingleInput, Variable


def multiprocessing_wrapper(arguments: SingleInput) -> None:
    """Make whole stacktrace available in parent process when error is raised."""
    function_to_execute = arguments[0]
    function_arguments: tuple[GeneralArguments, Variable] = (arguments[1], arguments[2])
    try:
        function_to_execute(function_arguments)
    except BaseException as error:
        raise RuntimeError(
            f"Arguments: {function_arguments}"
            + "".join(traceback.format_exception(*sys.exc_info()))
        ) from error


def order_columns_of_row(groups):
    return [
        "year",
        *groups,
        "n",
        "mean",
        "mean_lower_confidence",
        "mean_upper_confidence",
        "lower_quartile",
        "boxplot_median",
        "upper_quartile",
        "lower_whisker",
        "upper_whisker",
        "median",
        "median_lower_confidence",
        "median_upper_confidence",
    ]
