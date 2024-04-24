import multiprocessing
import sys
import traceback

from argparse import ArgumentParser
from os import mkdir
from itertools import repeat
from pathlib import Path
from shutil import rmtree
from sys import argv

from numpy import arange, array2string, isin, isnan, logical_and, sqrt, unique
from pandas import DataFrame, Series, read_stata

from transfer_statistics.handle_metadata import create_metadata_file

from transfer_statistics.calculate_metrics import (
    bootstrap_median,
    weighted_boxplot_sections,
    weighted_mean_and_confidence_interval,
)
from transfer_statistics.handle_files import (
    apply_value_labels,
    get_variable_combinations,
    read_value_label_metadata,
    read_variable_metadata,
)
from transfer_statistics.types import (
    GeneralArguments,
    Variable,
    VariableMetadata,
)


MINIMAL_GROUP_SIZE = 30
PROCESSES = 4
MEDIAN_BOOTSTRAP_PROCESSES = 8
Z_ALPHA = 1.96

MISSING_VALUES = arange(start=-1, stop=-9, step=-1)

EMPTY_NUMERICAL_RESULT = {
    "mean": None,
    "mean_lower_confidence": None,
    "mean_upper_confidence": None,
    "lower_quartile": None,
    "boxplot_median": None,
    "upper_quartile": None,
    "lower_whisker": None,
    "upper_whisker": None,
    "median": None,
    "median_lower_confidence": None,
    "median_upper_confidence": None,
    "n": None,
}


def _existing_path(path):
    output = Path(path).absolute()
    if output.exists():
        return output
    raise FileNotFoundError(f"Path {output} does not exist.")


def cli():
    """CLI entrypoint.

    Sets argument parser and manages program flow.
    """
    parser = ArgumentParser(
        prog="Transfer Statistics Pipeline",
        description="Calculate Transfer Statistics",
    )
    parser.add_argument("-d", "--dataset-path", type=_existing_path, required=True)
    parser.add_argument("-n", "--dataset-name", type=str, required=True)
    parser.add_argument("-m", "--metadata-path", type=_existing_path, required=True)
    parser.add_argument("-o", "--output-path", type=_existing_path, required=True)
    parser.add_argument("-w", "--weight-field-name", type=str, required=True)

    arguments = parser.parse_args(argv[1:])
    output_path: Path = arguments.output_path
    numerical_output_path = output_path.joinpath("numerical")
    categorical_output_path = output_path.joinpath("categorical")
    metadata = read_variable_metadata(
        arguments.metadata_path.joinpath("variables.csv"), arguments.dataset_name
    )
    value_labels, categorical_labels = read_value_label_metadata(
        arguments.metadata_path.joinpath("variable_categories.csv"), metadata
    )
    data = read_stata(
        arguments.dataset_path, convert_missing=False, convert_categoricals=False
    )

    data = data.replace(MISSING_VALUES, None)

    handle_categorical_statistics(
        data,
        metadata,
        value_labels,
        categorical_labels,
        categorical_output_path,
        arguments.weight_field_name,
    )
    handle_numerical_statistics(
        data, metadata, value_labels, numerical_output_path, arguments.weight_field_name
    )


def _create_variable_folders(variables, output_folder):
    for variable in variables:
        variable_file_target = output_folder.joinpath(variable["name"])
        if variable_file_target.exists():
            rmtree(variable_file_target)
        mkdir(variable_file_target)


def handle_numerical_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    value_labels,
    output_folder: Path,
    weight_name: str,
) -> None:
    _type = "numerical"

    if not output_folder.exists():
        mkdir(output_folder)
    variable_combinations = get_variable_combinations(metadata=metadata)

    _create_variable_folders(metadata[_type], output_folder)

    with multiprocessing.Pool(processes=PROCESSES) as pool:
        names = []
        _grouping_names = ["syear"]
        general_arguments: GeneralArguments = {
            "data": data,
            "grouping_names": _grouping_names,
            "weight_name": weight_name,
            "value_labels": value_labels,
            "output_folder": output_folder,
        }

        arguments = zip(
            repeat(create_metadata_file),
            repeat(general_arguments),
            metadata[_type],
        )
        pool.map(multiprocessing_wrapper, arguments)
        arguments = zip(
            repeat(_calculate_one_numerical_variable_in_parallel),
            repeat(general_arguments),
            metadata[_type],
        )
        pool.map(multiprocessing_wrapper, arguments)

    with multiprocessing.Pool(
        processes=MEDIAN_BOOTSTRAP_PROCESSES
    ) as median_bootstrap_pool:

        for group in variable_combinations:
            names = [variable["name"] for variable in group]
            _grouping_names = ["syear", *names]
            general_arguments = {
                "data": data,
                "grouping_names": _grouping_names,
                "weight_name": weight_name,
                "value_labels": value_labels,
                "output_folder": output_folder,
            }

            for variable in metadata[_type]:
                _calculate_one_numerical_variable_bootstrap_parallel(
                    general_arguments, variable, pool=median_bootstrap_pool
                )


def handle_categorical_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    value_labels,
    categorical_labels,
    output_folder: Path,
    weight_name: str,
) -> None:
    _type = "categorical"
    value_labels = {"group": value_labels, "categorical": categorical_labels}

    if not output_folder.exists():
        mkdir(output_folder)
    variable_combinations = get_variable_combinations(metadata=metadata)

    _create_variable_folders(metadata[_type], output_folder)

    with multiprocessing.Pool(processes=PROCESSES) as pool:
        names = []
        _grouping_names = ["syear"]
        general_arguments: GeneralArguments = {
            "data": data,
            "grouping_names": _grouping_names,
            "weight_name": weight_name,
            "value_labels": value_labels,
            "output_folder": output_folder,
        }

        arguments = zip(
            repeat(create_metadata_file),
            repeat(general_arguments),
            metadata[_type],
        )
        pool.map(multiprocessing_wrapper, arguments)
        arguments = zip(
            repeat(_calculate_one_categorical_variable_in_parallel),
            repeat(general_arguments),
            metadata[_type],
        )
        pool.map(multiprocessing_wrapper, arguments)

        for group in variable_combinations:
            names = [variable["name"] for variable in group]
            _grouping_names = ["syear", *names]
            general_arguments = {
                "data": data,
                "grouping_names": _grouping_names,
                "weight_name": weight_name,
                "value_labels": value_labels,
                "output_folder": output_folder,
            }
            arguments = zip(
                repeat(_calculate_one_categorical_variable_in_parallel),
                repeat(general_arguments),
                metadata[_type],
            )
            pool.map(multiprocessing_wrapper, arguments)


def multiprocessing_wrapper(arguments) -> None:
    """Make whole stacktrace available in parent process."""
    exec_function = arguments[0]
    arguments = arguments[1:]
    try:
        exec_function(arguments)
    except BaseException as error:
        raise RuntimeError(
            "".join(traceback.format_exception(*sys.exc_info()))
        ) from error


def _filter_year_from_group_names(group_names) -> list[str]:
    return group_names[1:]


def _calculate_one_categorical_variable_in_parallel(
    arguments: tuple[GeneralArguments, Variable]
) -> None:
    args = arguments[0]
    variable = arguments[1]
    data = args["data"]
    data_no_missing = data[~isin(data[variable["name"]], MISSING_VALUES)]
    data_slice = data_no_missing[[*args["grouping_names"], variable["name"]]]
    del data, data_no_missing

    aggregated_dataframe = (
        data_slice.groupby("syear").value_counts(normalize=True).reset_index()
    )
    population = args["data"]["syear"].value_counts().rename("n")
    aggregated_dataframe = aggregated_dataframe.merge(
        population, left_on="syear", right_on="syear"
    )
    aggregated_dataframe[["lower_confidence", "upper_confidence"]] = (
        aggregated_dataframe.apply(
            _calculate_population_confidence_interval, axis=1, args=("proportion", "n")
        )
    )

    # syear is at index 0 and does not need labeling
    columns_to_label = _filter_year_from_group_names(args["grouping_names"])

    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"]["group"], columns_to_label
    )
    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"]["categorical"], [variable["name"]]
    )
    _save_dataframe(aggregated_dataframe, args, variable)


def _calculate_population_confidence_interval(row, proportion_column, n_column):

    p = row[proportion_column]
    q = 1 - p
    n = row[n_column]
    stderr = Z_ALPHA * sqrt(p * q / n)
    return Series({"lower_confidence": p - stderr, "upper_confidence": p + stderr})


def _calculate_one_numerical_variable_in_parallel(
    arguments: tuple[Variable, GeneralArguments]
):
    args = arguments[0]
    variable = arguments[1]

    aggregated_dataframe = (
        args["data"][[*args["grouping_names"], variable["name"], args["weight_name"]]]
        .groupby(args["grouping_names"])
        .apply(
            _apply_numerical_aggregations,
            variable_name=variable["name"],
            weight_name=args["weight_name"],
        )
    )  # type: ignore

    columns_to_label = _filter_year_from_group_names(args["group_names"])
    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"], columns_to_label
    )
    _save_dataframe(aggregated_dataframe, args, variable)


def _calculate_one_numerical_variable_bootstrap_parallel(
    general_arguments: GeneralArguments, variable: Variable, pool
):

    aggregated_dataframe = (
        general_arguments["data"][
            [
                *general_arguments["grouping_names"],
                variable["name"],
                general_arguments["weight_name"],
            ]
        ]
        .groupby(general_arguments["grouping_names"])
        .apply(
            _apply_numerical_aggregations,
            variable_name=variable["name"],
            weight_name=general_arguments["weight_name"],
            pool=pool,
        )
    )  # type: ignore

    columns_to_label = _filter_year_from_group_names(general_arguments["group_names"])
    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, general_arguments["value_labels"], columns_to_label
    )
    _save_dataframe(aggregated_dataframe, general_arguments, variable)


def _save_dataframe(aggregated_dataframe, args, variable):

    labeled_columns = _filter_year_from_group_names(args["group_names"])

    group_file_name = "_".join(labeled_columns)
    if group_file_name:
        group_file_name = "_" + group_file_name

    file_name = (
        args["output_folder"]
        .joinpath(variable["name"])
        .joinpath(f"{variable['name']}_year{group_file_name}.csv")
    )

    aggregated_dataframe.rename(columns={"syear": "year"}, inplace=True)

    aggregated_dataframe.to_csv(file_name, index=False)


def _apply_numerical_aggregations(
    grouped_data_frame: DataFrame,
    variable_name: str,
    weight_name: str,
    pool: multiprocessing.Pool = None,
) -> Series:
    grouped_data_frame = grouped_data_frame.sort_values(by=variable_name)
    values = grouped_data_frame[variable_name].to_numpy()
    weights = grouped_data_frame[weight_name].to_numpy()

    no_missing_selector = logical_and(
        logical_and(~isin(values, MISSING_VALUES), ~isnan(values)),
        logical_and(~isin(weights, MISSING_VALUES), ~isnan(weights)),
    )

    weights = weights[no_missing_selector]
    values = values[no_missing_selector]

    if values.size == 0 or values.size < MINIMAL_GROUP_SIZE:
        return Series(EMPTY_NUMERICAL_RESULT.copy())

    try:
        output = weighted_mean_and_confidence_interval(values, weights)
        output = output | weighted_boxplot_sections(values, weights)
        output = output | bootstrap_median(values, weights, pool)
        output = output | {"n": values.size}
    except ValueError as error:
        values_to_print = array2string(unique(values), separator=", ")
        raise ValueError(
            f"Error with variable {variable_name} and values {values_to_print}"
        ) from error

    return Series(output, index=list(output.keys()))
