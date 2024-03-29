from pathlib import Path
from typing import TypedDict

from pandas import DataFrame

YEAR_COLUMN = "year"


class Variable(TypedDict):
    dataset: str
    name: str
    label: str
    label_de: str


class VariableMetadata(TypedDict):
    categorical: list[Variable]
    numerical: list[Variable]
    group: list[Variable]


class GroupingVariable(TypedDict):
    variable: str
    label: str
    label_de: str
    value_labels: list[str]
    values: list[int]


type VariableName = str
type VariableID = VariableName
type ValueLabels = dict[VariableID, GroupingVariable]


class GeneralArguments(TypedDict):
    data: DataFrame
    names: list[str]
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
    dimensions: list[GroupingVariable]
    groups: list[GroupingVariable]
    start_year: int
    end_year: int
