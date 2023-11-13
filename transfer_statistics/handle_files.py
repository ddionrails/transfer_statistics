from csv import DictReader
from itertools import combinations
from pathlib import Path

from transfer_statistics.types import VariableMetadata, Variable


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


def get_variable_combinations(metadata: VariableMetadata):
    group_combinations: list[tuple[Variable] | tuple[Variable, Variable]] = [
        (variable,) for variable in metadata["group"]
    ]
    group_combinations.extend(list(combinations(metadata["group"], 2)))
    return group_combinations
