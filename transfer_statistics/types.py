from pathlib import Path
from typing import TypedDict

from pandas import DataFrame


class Variable(TypedDict):
    dataset: str
    name: str
    label: str
    label_de: str


class VariableMetadata(TypedDict):
    categorical: list[Variable]
    numerical: list[Variable]
    group: list[Variable]


class LabeledVariable(TypedDict):
    variable: str
    label: str
    label_de: str
    value_labels: list[str]
    value_labels_de: list[str]
    values: list[int]


type VariableName = str
type VariableID = VariableName
type ValueLabels = dict[VariableID, LabeledVariable]


class GeneralArguments(TypedDict):
    data: DataFrame
    grouping_names: list[str]
    weight_name: str
    value_labels: ValueLabels
    output_folder: Path


class MetadataFile(TypedDict):
    dataset: str
    title: str
    label: str
    label_de: str
    variable: str
    dimensions: list[LabeledVariable]
    groups: list[LabeledVariable]
    start_year: int
    end_year: int
