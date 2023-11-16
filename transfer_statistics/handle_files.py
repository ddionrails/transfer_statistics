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
) -> list[GroupingVariable]:
    output: list[GroupingVariable] = []
    current_variable = GroupingVariable(
        variable="", label="", value_labels=[], values=[]
    )
    with open(value_label_file, "r", encoding="utf-8") as file:
        reader = DictReader(file)
        for line in reader:
            if line["variable"] != current_variable["variable"]:
                output.append(current_variable)
                current_variable = GroupingVariable(
                    variable=line["variable"],
                    label="", # TODO: Change var_metadata be able to get that
                    values=[line["value"]],
                    value_labels=[line["label_de"]],
                )
                continue
            current_variable["values"].append(int(line["value"]))
            current_variable["value_labels"].append(line["label_de"])
        output.append(current_variable)
    output.pop(0)
    return output


def get_variable_combinations(metadata: VariableMetadata):
    group_combinations: list[tuple[Variable] | tuple[Variable, Variable]] = [
        (variable,) for variable in metadata["group"]
    ]
    group_combinations.extend(list(combinations(metadata["group"], 2)))
    return group_combinations
