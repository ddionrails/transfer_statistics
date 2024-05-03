from csv import DictReader
from itertools import combinations
from pathlib import Path
from typing import Literal, Union

from pandas import DataFrame

from transfer_statistics.types import (LabeledVariable, ValueLabels, Variable,
                                       VariableMetadata)


def apply_value_labels(
    dataframe: DataFrame, value_labels: ValueLabels, grouping_variables: list[str]
) -> DataFrame:
    dataframe = dataframe.reset_index(drop=False)
    if "index" in dataframe.columns:
        dataframe = dataframe.drop("index", axis=1)
    for variable in grouping_variables:
        dataframe[variable] = dataframe[variable].replace(
            value_labels[variable]["values"], value_labels[variable]["value_labels"]
        )
    return dataframe


def apply_value_labels_to_list_of_dict(
    rows: DataFrame, value_labels: ValueLabels, grouping_variables: list[str]
) -> DataFrame:
    mapping = {}
    for variable in grouping_variables:
        mapping[variable] = dict(
            zip(value_labels[variable]["values"], value_labels[variable]["value_labels"])
        )
    for row in rows:
        row["year"] = row["syear"]
        del row["syear"]
        for variable in grouping_variables:
            row[variable] = mapping[variable][int(row[variable])]

    return rows


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
) -> tuple[ValueLabels, ValueLabels]:
    labeled_variables: dict[
        Union[Literal["group"], Literal["categorical"]], ValueLabels
    ] = {"group": {}, "categorical": {}}
    grouping_variables: dict[tuple[str, str], Variable] = {}
    categorical_variables: dict[tuple[str, str], Variable] = {}
    _id = ()

    for variable in variable_metadata["group"]:
        grouping_variables[variable["name"]] = variable

    for variable in variable_metadata["categorical"]:
        categorical_variables[variable["name"]] = variable

    with open(value_label_file, "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for line in reader:
            _id = line["variable"]
            _type: Union[Literal["group"], Literal["categorical"]] = "group"
            variables = grouping_variables
            if _id not in grouping_variables and _id not in categorical_variables:
                continue
            if _id in categorical_variables:
                _type = "categorical"
                variables = categorical_variables
            if _id not in labeled_variables[_type]:
                labeled_variables[_type][_id] = LabeledVariable(
                    variable=line["variable"],
                    label=variables[_id]["label"],
                    label_de=variables[_id]["label_de"],
                    values=[],
                    value_labels=[],
                )
            labeled_variables[_type][_id]["values"].append(int(line["value"]))
            labeled_variables[_type][_id]["value_labels"].append(line["label_de"])
    return labeled_variables["group"], labeled_variables["categorical"]


def get_variable_combinations(metadata: VariableMetadata):
    group: list[Variable] = []
    group_combinations: list[tuple[Variable] | tuple[Variable, Variable]] = []
    for variable in metadata["group"]:
        if variable["name"] != "syear":
            group.append(variable)
            group_combinations.append((variable,))

    group_combinations.extend(list(combinations(group, 2)))
    return group_combinations
