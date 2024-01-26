from csv import DictReader
from itertools import combinations
from pathlib import Path
from pandas import DataFrame

from transfer_statistics.types import (
    GroupingVariable,
    ValueLabels,
    Variable,
    VariableMetadata,
)


def apply_value_labels(
    dataframe: DataFrame, value_labels: ValueLabels, grouping_variables: tuple[str, ...]
) -> DataFrame:
    dataframe = dataframe.reset_index(drop=False)
    for variable in grouping_variables:
        dataframe[variable] = dataframe[variable].replace(
            value_labels[variable]["values"], value_labels[variable]["value_labels"]
        )
    return dataframe


def read_variable_metadata(metadata_file: Path, dataset_name: str) -> VariableMetadata:
    metadata: VariableMetadata = VariableMetadata(categorical=[], numerical=[], group=[])
    with open(metadata_file, "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for line in reader:
            if line["dataset"] != dataset_name:
                continue
            if line["type"]:
                metadata[line["type"]].append(
                    Variable(
                        dataset=line["dataset"],
                        name=line["variable"],
                        label=line["label"],
                        label_de=line["label_de"],
                    )
                )
    return metadata


def read_value_label_metadata(
    value_label_file: Path, variable_metadata: VariableMetadata
) -> ValueLabels:
    output: ValueLabels = {}
    grouping_variables: dict[tuple[str, str], Variable] = {}
    _id = ()

    for variable in variable_metadata["group"]:
        grouping_variables[variable["name"]] = variable

    with open(value_label_file, "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for line in reader:
            _id = line["variable"]
            if _id not in grouping_variables:
                continue
            if _id not in output:
                output[_id] = GroupingVariable(
                    variable=line["variable"],
                    label=grouping_variables[_id]["label_de"],
                    values=[int(line["value"])],
                    value_labels=[line["label_de"]],
                )
                continue
            output[_id]["values"].append(int(line["value"]))
            output[_id]["value_labels"].append(line["label_de"])
    return output


def get_variable_combinations(metadata: VariableMetadata):
    group: list[Variable] = []
    group_combinations: list[tuple[Variable] | tuple[Variable, Variable]] = []
    for variable in metadata["group"]:
        if variable["name"] != "syear":
            group.append(variable)
            group_combinations.append((variable,))

    group_combinations.extend(list(combinations(group, 2)))
    return group_combinations
