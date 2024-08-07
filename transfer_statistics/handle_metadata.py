from json import dump

from pandas import DataFrame, to_numeric

from transfer_statistics.types import (GeneralArguments, MetadataFile,
                                       ValueLabels, Variable)


def _get_start_and_end_year(data: DataFrame):
    years = to_numeric(data[data.notnull()]["syear"])
    return int(years.min()), int(years.max())


def create_numerical_variable_metadata_file(arguments: tuple[GeneralArguments, Variable]):
    data = arguments[0]["data"]
    variable = arguments[1]
    output_folder = arguments[0]["output_folder"].joinpath(variable["name"])
    output_file = output_folder.joinpath("meta.json")
    value_labels: ValueLabels = arguments[0]["value_labels"]
    grouping_variables_names = list(arguments[0]["value_labels"].get("group", {}).keys())

    # TODO: Refactor; fix typing issues and untangle the value_label handling
    if "group" in value_labels:
        value_labels = value_labels["group"]
    groups = list(value_labels.values())
    dimensions = []
    for group in groups:
        dimensions.append(
            {
                "variable": group["variable"],
                "label": group["label"],
                "label_de": group["label_de"],
                "values": group["values"],
                "labels": group["value_labels"],
                "labels_de": group["value_labels_de"],
            }
        )
    start_year, end_year = _get_start_and_end_year(data[["syear", variable["name"]]])
    metadata: MetadataFile = {
        "dataset": variable["dataset"],
        "title": variable["label"],
        "label": variable["label"],
        "label_de": variable["label_de"],
        "variable": variable["name"],
        "groups": grouping_variables_names,
        "start_year": start_year,
        "end_year": end_year,
    }
    with open(output_file, "w", encoding="utf-8") as file:
        dump(metadata, fp=file, ensure_ascii=False)


def create_categorical_variable_metadata_file(
    arguments: tuple[GeneralArguments, Variable]
):
    data = arguments[0]["data"]
    variable = arguments[1]

    output_folder = arguments[0]["output_folder"].joinpath(variable["name"])
    output_file = output_folder.joinpath("meta.json")

    value_labels_container = arguments[0]["value_labels"].get("categorical", {})
    # TODO: Consolidate value_label handling
    grouping_variables_names = list(arguments[0]["value_labels"].get("group", {}).keys())

    values = value_labels_container[variable["name"]].get("values", [])
    value_labels = value_labels_container[variable["name"]].get("value_labels", [])
    value_labels_de = value_labels_container[variable["name"]].get("value_labels_de", [])

    start_year, end_year = _get_start_and_end_year(data[["syear", variable["name"]]])
    metadata = {
        "dataset": variable["dataset"],
        "title": variable["label"],
        "label": variable["label"],
        "label_de": variable["label_de"],
        "variable": variable["name"],
        "values": values,
        "value_labels": value_labels,
        "value_labels_de": value_labels_de,
        "groups": grouping_variables_names,
        "start_year": start_year,
        "end_year": end_year,
    }
    with open(output_file, "w", encoding="utf-8") as file:
        dump(metadata, fp=file, ensure_ascii=False)
