import multiprocessing
from argparse import ArgumentParser
from csv import DictWriter
from itertools import repeat
from os import mkdir
from pathlib import Path
from shutil import rmtree
from sys import argv
from typing import Literal

from numpy import (
    NaN,
    arange,
    argsort,
    array2string,
    isin,
    isnan,
    logical_and,
    unique,
)
from pandas import DataFrame, Series, read_stata

from transfer_statistics.calculate_metrics import (
    calculate_population_confidence_interval,
    calculate_weighted_percentage,
    weighted_boxplot_sections,
    weighted_mean_and_confidence_interval,
)
from transfer_statistics.handle_files import (
    apply_value_labels,
    apply_value_labels_to_list_of_dict,
    get_variable_combinations,
    read_value_label_metadata,
    read_variable_metadata,
    write_group_variables_metadata_file,
)
from transfer_statistics.handle_metadata import (
    create_categorical_variable_metadata_file,
    create_numerical_variable_metadata_file,
)
from transfer_statistics.helpers import multiprocessing_wrapper, order_columns_of_row
from transfer_statistics.types import (
    GeneralArguments,
    MultiProcessingInput,
    ResultRow,
    ValueLabels,
    Variable,
    VariableMetadata,
)

MINIMAL_GROUP_SIZE = 30
PROCESSES = 4
MEDIAN_BOOTSTRAP_PROCESSES = 20

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
    value_labels, categorical_main_variables_labels = read_value_label_metadata(
        arguments.metadata_path.joinpath("variable_categories.csv"), metadata
    )
    write_group_variables_metadata_file(path=output_path, value_labels=value_labels)
    data = read_stata(
        arguments.dataset_path, convert_missing=False, convert_categoricals=False
    )

    all_value_labels: dict[str, ValueLabels] = {
        "group": value_labels,
        "categorical": categorical_main_variables_labels,
    }

    handle_statistics(
        data=data,
        metadata=metadata,
        all_value_labels=all_value_labels,
        output_folder=categorical_output_path,
        weight_name=arguments.weight_field_name,
        _type="categorical",
    )
    handle_statistics(
        data=data,
        metadata=metadata,
        all_value_labels=all_value_labels,
        output_folder=numerical_output_path,
        weight_name=arguments.weight_field_name,
        _type="numerical",
    )


def _create_variable_folders(variables, output_folder):

    if output_folder.exists():
        rmtree(output_folder)
    mkdir(output_folder)

    for variable in variables:
        variable_file_target = output_folder.joinpath(variable["name"])
        if variable_file_target.exists():
            rmtree(variable_file_target)
        mkdir(variable_file_target)


def handle_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    all_value_labels,
    output_folder: Path,
    weight_name: str,
    _type: Literal["categorical"] | Literal["numerical"],
) -> None:

    variable_combinations = get_variable_combinations(metadata=metadata)

    _create_variable_folders(metadata[_type], output_folder)

    names = []
    _group_by_column_names = ["syear"]
    general_arguments: GeneralArguments = {
        "data": data,
        "grouping_names": _group_by_column_names,
        "weight_name": weight_name,
        "value_labels": all_value_labels,
        "output_folder": output_folder,
    }

    if _type == "categorical":
        metadata_function = create_categorical_variable_metadata_file
        no_grouping_function = _calculate_one_categorical_variable_in_parallel
    elif _type == "numerical":
        metadata_function = create_numerical_variable_metadata_file
        no_grouping_function = _calculate_one_numerical_variable_in_parallel

    metadata_file_arguments: MultiProcessingInput = zip(
        repeat(metadata_function),
        repeat(general_arguments),
        metadata[_type],
    )
    no_grouping_variables_arguments: MultiProcessingInput = zip(
        repeat(no_grouping_function),
        repeat(general_arguments),
        metadata[_type],
    )

    with multiprocessing.Pool(processes=PROCESSES) as pool:
        pool.map(multiprocessing_wrapper, metadata_file_arguments)
        del metadata_file_arguments
        pool.map(multiprocessing_wrapper, no_grouping_variables_arguments)
        del no_grouping_variables_arguments

    for group in variable_combinations:
        names = [variable["name"] for variable in group]
        _group_by_column_names = ["syear", *names]
        general_arguments = {
            "data": data,
            "grouping_names": _group_by_column_names,
            "weight_name": weight_name,
            "value_labels": all_value_labels,
            "output_folder": output_folder,
        }

        if _type == "categorical":
            with multiprocessing.Pool(processes=PROCESSES) as pool:
                arguments: MultiProcessingInput = zip(
                    repeat(_calculate_one_categorical_variable_in_parallel),
                    repeat(general_arguments),
                    metadata[_type],
                )
                pool.map(multiprocessing_wrapper, arguments)
        if _type == "numerical":
            for variable in metadata[_type]:
                _parallelize_by_group_numerical(general_arguments, variable)


def _remove_year_from_group_names(group_names) -> list[str]:
    if group_names[0] not in ("syear", "year"):
        raise ValueError("Year column not in correct position.")
    return group_names[1:]


def _calculate_one_categorical_variable_in_parallel(
    arguments: tuple[GeneralArguments, Variable]
) -> None:
    args = arguments[0]
    variable = arguments[1]
    data = args["data"]
    data_no_missing = data[~isin(data[variable["name"]], MISSING_VALUES)]
    data_slice = data_no_missing[[*args["grouping_names"], variable["name"]]]
    weight_data_slice = data_no_missing[
        [*args["grouping_names"], variable["name"], args["weight_name"]]
    ]
    del data, data_no_missing

    grouped_data = data_slice.groupby(
        [*args["grouping_names"], variable["name"]], observed=True
    )
    filtered_data = grouped_data.filter(lambda size: len(size) > MINIMAL_GROUP_SIZE)
    grouped_data = filtered_data.groupby(
        [*args["grouping_names"], variable["name"]], observed=True
    )

    small_n = grouped_data.value_counts().rename("n").reset_index()
    aggregated_dataframe = calculate_weighted_percentage(
        weight_data_slice,
        args["grouping_names"],
        variable["name"],
        args["weight_name"],
    )

    population = (
        filtered_data[args["grouping_names"]]
        .groupby(args["grouping_names"], observed=True)
        .value_counts()
        .rename("group_size")
    )
    aggregated_dataframe = aggregated_dataframe.merge(
        population, left_on=args["grouping_names"], right_on=args["grouping_names"]
    )
    aggregated_dataframe = aggregated_dataframe.merge(
        small_n,
        left_on=[*args["grouping_names"], variable["name"]],
        right_on=[*args["grouping_names"], variable["name"]],
    )

    try:
        aggregated_dataframe[
            ["proportion_lower_confidence", "proportion_upper_confidence"]
        ] = aggregated_dataframe.apply(
            calculate_population_confidence_interval,
            axis=1,
            args=("proportion", "group_size"),
        )
    except ValueError as error:
        if aggregated_dataframe.empty:
            return None
        raise error

    columns_to_label = _remove_year_from_group_names(args["grouping_names"])

    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"]["group"], columns_to_label
    )
    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"]["categorical"], [variable["name"]]
    )

    aggregated_dataframe = aggregated_dataframe[
        [
            *args["grouping_names"],
            variable["name"],
            "proportion",
            "n",
            "group_size",
            "proportion_lower_confidence",
            "proportion_upper_confidence",
        ]
    ]

    _save_dataframe(aggregated_dataframe, args, variable)
    return None


def _calculate_one_numerical_variable_in_parallel(
    arguments: tuple[GeneralArguments, Variable]
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

    columns_to_label = _remove_year_from_group_names(args["grouping_names"])
    aggregated_dataframe = apply_value_labels(
        aggregated_dataframe, args["value_labels"], columns_to_label
    )
    _save_dataframe(aggregated_dataframe, args, variable)


def _parallelize_by_group_numerical(
    general_arguments: GeneralArguments, variable: Variable
):

    dataframe = general_arguments["data"]
    filtered_dataframe = dataframe[
        [
            *general_arguments["grouping_names"],
            variable["name"],
            general_arguments["weight_name"],
        ]
    ]
    dataframe_groups = filtered_dataframe.groupby(general_arguments["grouping_names"])

    arguments = []
    for grouped_by, group in dataframe_groups:
        grouped_columns: ResultRow = dict(
            zip(general_arguments["grouping_names"], grouped_by)
        )
        arguments.append(
            (
                grouped_columns,
                group[variable["name"]].to_numpy(),
                group[general_arguments["weight_name"]].to_numpy(),
            )
        )

    with multiprocessing.Pool(processes=MEDIAN_BOOTSTRAP_PROCESSES) as pool:
        rows = pool.map(_caclulate_numerical_aggregations_in_parallel, arguments)

        columns_to_label = _remove_year_from_group_names(
            general_arguments["grouping_names"]
        )
        rows = apply_value_labels_to_list_of_dict(
            rows, general_arguments["value_labels"], columns_to_label
        )
        _save_list_of_dicts(rows, general_arguments, variable)


def _save_dataframe(aggregated_dataframe, args, variable):

    labeled_columns = _remove_year_from_group_names(args["grouping_names"])

    group_file_name = "_".join(labeled_columns)
    if group_file_name:
        group_file_name = "_" + group_file_name

    file_name = (
        args["output_folder"]
        .joinpath(variable["name"])
        .joinpath(f"{variable['name']}_year{group_file_name}.csv")
    )

    aggregated_dataframe.rename(columns={"syear": "year"}, inplace=True)

    aggregated_dataframe.to_csv(file_name, index=False, float_format="%.5f")


def _save_list_of_dicts(
    rows: list[dict[str, str | float]], args: GeneralArguments, variable: Variable
):
    """Save numeric calculation output to CSV file

    Numerical calculations return a dict for each row/grouping.
    The keys are the names of the columns and the values are the calculated values or
    the original values from the grouping process,
    e.g. "1990" for one row in the year column
    or "Yes" for one row in a  "Yes"/"No" grouping column/variable.
    """

    labeled_columns = _remove_year_from_group_names(args["grouping_names"])

    group_file_name = "_".join(labeled_columns)
    if group_file_name:
        group_file_name = "_" + group_file_name

    file_name = (
        args["output_folder"]
        .joinpath(variable["name"])
        .joinpath(f"{variable['name']}_year{group_file_name}.csv")
    )

    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = f"{value:.5f}"

    with open(file_name, "w", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=order_columns_of_row(labeled_columns))
        writer.writeheader()
        writer.writerows(rows)


def _apply_numerical_aggregations(
    grouped_data_frame: DataFrame,
    variable_name: str,
    weight_name: str,
    pool: multiprocessing.Pool = None,
) -> Series:
    values = grouped_data_frame[variable_name].to_numpy()
    weights = grouped_data_frame[weight_name].to_numpy()

    # Boxplot need the values sorted
    sorter = argsort(values)
    values = values[sorter]
    weights = weights[sorter]

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
        output = output | {"n": values.size}
    except ValueError as error:
        values_to_print = array2string(unique(values), separator=", ")
        raise ValueError(
            f"Error with variable {variable_name} and values {values_to_print}"
        ) from error

    return Series(output, index=list(output.keys()))


def _caclulate_numerical_aggregations_in_parallel(
    arguments,
) -> Series:
    name_mapping = arguments[0]

    values = arguments[1]
    weights = arguments[2]

    # Boxplot need the values sorted
    sorter = argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    no_missing_selector = logical_and(
        logical_and(~isin(values, MISSING_VALUES), ~isnan(values)),
        logical_and(~isin(weights, MISSING_VALUES), ~isnan(weights)),
    )

    weights = weights[no_missing_selector]
    values = values[no_missing_selector]

    if values.size == 0 or values.size < MINIMAL_GROUP_SIZE:
        return name_mapping | EMPTY_NUMERICAL_RESULT.copy()

    try:
        output = name_mapping
        output = output | weighted_mean_and_confidence_interval(values, weights)
        output = output | weighted_boxplot_sections(values, weights)
        output = output | {"n": values.size}
    except ValueError as error:
        values_to_print = array2string(unique(values), separator=", ")
        raise ValueError(
            f"Error with group {name_mapping} and values {values_to_print}"
        ) from error

    return output
