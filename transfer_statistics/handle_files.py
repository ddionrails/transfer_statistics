from csv import DictReader
from itertools import combinations
from pathlib import Path

from transfer_statistics.types import VariableMetadata, Variable, GroupingVariable


def read_variable_metadata(metadata_file: Path) -> VariableMetadata:
    metadata: VariableMetadata = VariableMetadata(categorical=[], numeric=[], group=[])
    with open(metadata_file, "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for line in reader:
            metadata[line["statistical_type"]].append(
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
) -> dict[tuple[str, str], GroupingVariable]:
    output: dict[tuple[str, str], GroupingVariable] = {}
    grouping_variables: dict[tuple[str, str], Variable] = {}
    _id = ()

    for variable in variable_metadata["group"]:
        grouping_variables[(variable["dataset"], variable["name"])] = variable

    with open(value_label_file, "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for line in reader:
            _id = (line["dataset"], line["variable"])
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
    group_combinations: list[tuple[Variable] | tuple[Variable, Variable]] = [
        (variable,) for variable in metadata["group"]
    ]
    group_combinations.extend(list(combinations(metadata["group"], 2)))
    return group_combinations
