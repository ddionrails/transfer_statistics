from os import mkdir
from pathlib import Path
from shutil import rmtree
from sys import argv

from argparse import ArgumentParser
from pandas import DataFrame, Series, read_stata
from numpy import isnan, nan


from transfer_statistics.types import VariableMetadata
from transfer_statistics.handle_files import (
    apply_value_labels,
    get_variable_combinations,
    read_value_label_metadata,
    read_variable_metadata,
)
from transfer_statistics.calculate_metrics import (
    weighted_mean_and_confidence_interval,
    weighted_boxplot_sections,
    bootstrap_median,
)


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
    metadata = read_variable_metadata(arguments.metadata_path.joinpath("variables.csv"))
    value_labels = read_value_label_metadata(
        arguments.metadata_path.joinpath("variable_categories.csv"), metadata
    )
    data = read_stata(
        arguments.dataset_path, convert_missing=False, convert_categoricals=False
    )

    calculate_numerical_statistics(
        data, metadata, value_labels, numerical_output_path, arguments.weight_field_name
    )


def calculate_numerical_statistics(
    data: DataFrame,
    metadata: VariableMetadata,
    value_labels,
    output_folder: Path,
    weight_name: str,
) -> None:
    if not output_folder.exists():
        mkdir(output_folder)
    variable_combinations = get_variable_combinations(metadata=metadata)

    for variable in metadata["numerical"]:
        variable_file_target = output_folder.joinpath(variable["name"])
        if variable_file_target.exists():
            rmtree(variable_file_target)
        mkdir(variable_file_target)

    for group in variable_combinations:
        names = [variable["name"] for variable in group]
        _grouping_names = ["syear", *names]
        for variable in metadata["numerical"]:
            aggregated_dataframe = (
                data[[*_grouping_names, variable["name"], weight_name]]
                .groupby(_grouping_names)
                .apply(
                    _apply_numerical_aggregations,
                    variable_name=variable["name"],
                    weight_name=weight_name,
                )
            )  # type: ignore
            aggregated_dataframe = apply_value_labels(
                aggregated_dataframe, value_labels, names
            )
            group_file_name = "_".join(names)
            file_name = output_folder.joinpath(variable["name"]).joinpath(
                f"{variable['name']}_year_{group_file_name}.csv"
            )
            aggregated_dataframe.to_csv(file_name, index=False)


def _apply_numerical_aggregations(
    grouped_data_frame: DataFrame, variable_name: str, weight_name: str
) -> Series:
    grouped_data_frame = grouped_data_frame.sort_values(by=variable_name)
    values = grouped_data_frame[variable_name].to_numpy()
    weights = grouped_data_frame[weight_name].to_numpy()
    # TODO: What to do with Missings?
    weights = weights[~isnan(values)]
    values = values[~isnan(values)]
    if values.size == 0:
        return None
        return Series(EMPTY_RESULT.copy(), index=list(EMPTY_RESULT.keys()))

    output = weighted_mean_and_confidence_interval(values, weights)
    output = output | weighted_boxplot_sections(values, weights)
    output = output | bootstrap_median(values, weights)
    return Series(output, index=list(output.keys()))


EMPTY_RESULT = {
    "median": nan,
    "median_lower_confidence": nan,
    "median_upper_confidence": nan,
    "lower_quartile": nan,
    "boxplot_median": nan,
    "upper_quartile": nan,
    "lower_whisker": nan,
    "upper_whisker": nan,
    "mean" "mean_lower_confidence": nan,
    "mean_upper_confidence": nan,
}
