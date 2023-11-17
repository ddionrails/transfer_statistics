from pathlib import Path

from pandas import DataFrame, Series
from numpy import isnan


from transfer_statistics.types import VariableMetadata
from transfer_statistics.handle_files import get_variable_combinations
from transfer_statistics.calculate_metrics import (
    weighted_mean_and_confidence_interval,
    weighted_boxplot_sections,
    bootstrap_median,
)


def calculate_numeric_statistics(
    data: DataFrame, metadata: VariableMetadata, output_folder: Path
) -> None:
    variable_combinations = get_variable_combinations(metadata=metadata)

    for group in variable_combinations:
        _grouping_names = ["syear", *group]
        grouped_dataframe = data.groupby(by=_grouping_names)
        for variable_name in metadata["numeric"]:
            aggregated_dataframe = grouped_dataframe[
                [*_grouping_names, variable_name]
            ].apply(
                _apply_numeric_aggregations, variable_name=variable_name
            )  # type: ignore


def _apply_numeric_aggregations(
    grouped_data_frame: DataFrame, variable_name
) -> Series[float64]:
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
