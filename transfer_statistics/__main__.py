import multiprocessing
from argparse import ArgumentParser
from os import mkdir
from pathlib import Path
from shutil import rmtree
from sys import argv

from numpy import arange, isin, isnan, logical_and, nan, sqrt
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
Z_ALPHA = 1.96

MISSING_VALUES = arange(start=-1, stop=-9, step=-1)


def _existing_path(path):
    output = Path(path).absolute()
    if output.exists():
        return output
    raise FileNotFoundError(f"Path {output} does not exist.")


def cli():
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
    value_labels = read_value_label_metadata(
        arguments.metadata_path.joinpath("variable_categories.csv"), metadata
    )
    data = read_stata(
        arguments.dataset_path, convert_missing=False, convert_categoricals=False
    )
    # data = data.rename(columns={"syear": "syear"})

    calculate_categorical_statistics(
        data, metadata, value_labels, categorical_output_path
    )
    calculate_numerical_statistics(
        data, metadata, value_labels, numerical_output_path, arguments.weight_field_name
    )


def _create_variable_folders(variables, output_folder):
    for variable in variables:
        variable_file_target = output_folder.joinpath(variable["name"])
        if variable_file_target.exists():
            rmtree(variable_file_target)
        mkdir(variable_file_target)


def calculate_numerical_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    value_labels,
    output_folder: Path,
    weight_name: str,
) -> None:
    calculate_statistics(
        data, metadata, value_labels, output_folder, weight_name, "numerical"
    )


def calculate_categorical_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    value_labels,
    output_folder: Path,
) -> None:
    calculate_statistics(data, metadata, value_labels, output_folder, "", "categorical")


def calculate_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    value_labels,
    output_folder: Path,
    weight_name: str,
    statistical_type: str,
) -> None:
    if not output_folder.exists():
        mkdir(output_folder)
    variable_combinations = get_variable_combinations(metadata=metadata)

    _create_variable_folders(metadata[statistical_type], output_folder)

    with multiprocessing.Pool(processes=PROCESSES) as pool:
        names = []
        _grouping_names = ["syear"]
        general_arguments: GeneralArguments = {
            "data": data,
            "names": names,
            "grouping_names": _grouping_names,
            "weight_name": weight_name,
            "value_labels": value_labels,
            "output_folder": output_folder,
        }
        arguments = [
            (variable, general_arguments) for variable in metadata[statistical_type]
        ]
        calculation_function = _calculate_one_numerical_variable_in_parallel
        if statistical_type == "categorical":
            calculation_function = _calculate_one_categorical_variable_in_parallel
        pool.map(create_metadata_file, arguments)
        pool.map(calculation_function, arguments)

        for group in variable_combinations:
            names = [variable["name"] for variable in group]
            _grouping_names = ["syear", *names]
            general_arguments = {
                "data": data,
                "names": names,
                "grouping_names": _grouping_names,
                "weight_name": weight_name,
                "value_labels": value_labels,
                "output_folder": output_folder,
            }
            arguments = [
                (variable, general_arguments) for variable in metadata[statistical_type]
            ]
            pool.map(calculation_function, arguments)


def calculate_one_variable(
    data, names, _grouping_names, variable, weight_name, value_labels, output_folder
):
    aggregated_dataframe = (
        data[[*_grouping_names, variable["name"], weight_name]]
        .groupby(_grouping_names)
        .apply(
            _apply_numerical_aggregations,
            variable_name=variable["name"],
            weight_name=weight_name,
        )
    )  # type: ignore
    aggregated_dataframe = apply_value_labels(aggregated_dataframe, value_labels, names)
    group_file_name = "_".join(names)
    file_name = output_folder.joinpath(variable["name"]).joinpath(
        f"{variable['name']}_year_{group_file_name}.csv"
    )
    aggregated_dataframe = aggregated_dataframe.rename(columns={"syear": "year"})
    aggregated_dataframe.to_csv(file_name, index=False)
    del aggregated_dataframe


def _calculate_one_categorical_variable_in_parallel(
    arguments: tuple[Variable, GeneralArguments]
):
    variable = arguments[0]
    args = arguments[1]
    data = args["data"][~isin(args["data"][variable["name"]], MISSING_VALUES)]
    data = data[[*args["grouping_names"], variable["name"]]]

    aggregated_dataframe = (
        data.groupby("syear").value_counts(normalize=True).reset_index()
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

    args["names"].insert(0, variable["name"])
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
    variable = arguments[0]
    args = arguments[1]

    aggregated_dataframe = (
        args["data"][[*args["grouping_names"], variable["name"], args["weight_name"]]]
        .groupby(args["grouping_names"])
        .apply(
            _apply_numerical_aggregations,
            variable_name=variable["name"],
            weight_name=args["weight_name"],
        )
    )  # type: ignore
    _save_dataframe(aggregated_dataframe, args, variable)


def _save_dataframe(aggregated_dataframe, args, variable):
    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"], args["names"]
    )
    # Fix for labeling of primary variables in categorical statistics
    # TODO: Think of a better way of handling this
    if variable["name"] in args["names"]:
        args["names"].pop(args["names"].index(variable["name"]))

    group_file_name = "_".join(args["names"])
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
    grouped_data_frame: DataFrame, variable_name: str, weight_name: str
) -> Series:
    grouped_data_frame = grouped_data_frame.sort_values(by=variable_name)
    values = grouped_data_frame[variable_name].to_numpy()
    weights = grouped_data_frame[weight_name].to_numpy()
    # TODO: What to do with Missings?
    no_missing_selector = logical_and(~isin(values, MISSING_VALUES), ~isnan(values))

    weights = weights[no_missing_selector]
    values = values[no_missing_selector]

    if values.size == 0 or values.size < MINIMAL_GROUP_SIZE:
        return None

    output = weighted_mean_and_confidence_interval(values, weights)
    output = output | weighted_boxplot_sections(values, weights)
    output = output | bootstrap_median(values, weights)
    output = output | {"n": values.size}
    return Series(output, index=list(output.keys()))
