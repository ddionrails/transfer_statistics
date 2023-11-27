from typing import TypedDict


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
    value_labels: list[str]
    values: list[int]


type VariableName = str
type VariableID = VariableName
type ValueLabels = dict[VariableID, GroupingVariable]
