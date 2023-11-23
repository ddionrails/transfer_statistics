from os import mkdir
from pathlib import Path
from shutil import rmtree
from sys import argv

from argparse import ArgumentParser
from pandas import DataFrame, Series, read_stata
from numpy import isnan


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

    arguments = parser.parse_args(argv)
    output_path = arguments.output_path
    numeric_output_path = output_path.join("numeric")
    metadata = read_variable_metadata(arguments.metadata_path.join("variables.csv"))
    value_labels = read_value_label_metadata(
        arguments.metadata_path.join("variable_categories.csv"), metadata
    )
    data = read_stata(
        arguments.dataset.path, convert_missing=True, convert_categoricals=False
    )

    calculate_numeric_statistics(data, metadata, value_labels, numeric_output_path)


def calculate_numeric_statistics(
    data: DataFrame, metadata: VariableMetadata, value_labels, output_folder: Path
) -> None:
    variable_combinations = get_variable_combinations(metadata=metadata)

    for variable_name in metadata["numeric"]:
        variable_file_target = output_folder.joinpath(variable_name)
        if variable_file_target.exists():
            rmtree(variable_file_target)
        mkdir(variable_file_target)

    for group in variable_combinations:
        _grouping_names = ["syear", *group]
        grouped_dataframe = data.groupby(by=_grouping_names)
        for variable_name in metadata["numeric"]:
            aggregated_dataframe = grouped_dataframe[
                [*_grouping_names, variable_name]
            ].apply(
                _apply_numeric_aggregations, variable_name=variable_name
            )  # type: ignore
            aggregated_dataframe = apply_value_labels(
                aggregated_dataframe, value_labels, group
            )
            group_file_name = "_".join(group)
            file_name = output_folder.joinpath(variable_name).joinpath(
                f"{variable_name}_year_{group_file_name}.csv"
            )
            aggregated_dataframe.to_csv(file_name, index=False)


def _apply_numeric_aggregations(grouped_data_frame: DataFrame, variable_name) -> Series:
    grouped_data_frame = grouped_data_frame.sort_values(by=variable_name)
    values = grouped_data_frame[variable_name].to_numpy()
    weights = grouped_data_frame["weights"].to_numpy()
    # TODO: What to do with Missings?
    weights = weights[~isnan(values)]
    values = values[~isnan(values)]
    output = weighted_mean_and_confidence_interval(values, weights)
    output = output | weighted_boxplot_sections(values, weights)
    output = output | bootstrap_median(values, weights)
    return Series(output, index=list(output.keys()))
