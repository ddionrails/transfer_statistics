from json import dump

from pandas import DataFrame, to_numeric

from transfer_statistics.types import (
    GeneralArguments,
    MetadataFile,
    Variable,
)


def _get_start_and_end_year(data: DataFrame):
    years = to_numeric(data[data.notnull()]["syear"])
    return int(years.min()), int(years.max())


def create_variable_metadata(
    general_arguments, data, variable, variable_type="numerical"
):

    output_folder = general_arguments["output_folder"].joinpath(variable["name"])
    output_file = output_folder.joinpath("meta.json")

    grouping_variables_names = list(
        general_arguments["value_labels"].get("group", {}).keys()
    )

    start_year, end_year = _get_start_and_end_year(data[["syear", variable["name"]]])
    if variable_type == "categorical":
        value_labels_container = general_arguments["value_labels"].get("categorical", {})
        this_variable_value_labels = value_labels_container[variable["name"]]

        metadata: MetadataFile = {
            "dataset": variable["dataset"],
            "title": variable["label"],
            "label": variable["label"],
            "label_de": variable["label_de"],
            "variable": variable["name"],
            "values": this_variable_value_labels.get("values", []),
            "value_labels": this_variable_value_labels.get("value_labels", []),
            "value_labels_de": this_variable_value_labels.get("value_labels_de", []),
            "groups": grouping_variables_names,
            "start_year": start_year,
            "end_year": end_year,
        }
    else:
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

    return metadata, output_file


def create_numerical_variable_metadata_file(arguments: tuple[GeneralArguments, Variable]):

    metadata, output_file = create_variable_metadata(
        general_arguments=arguments[0], data=arguments[0]["data"], variable=arguments[1]
    )

    with open(output_file, "w", encoding="utf-8") as file:
        dump(metadata, fp=file, ensure_ascii=False)


def create_categorical_variable_metadata_file(
    arguments: tuple[GeneralArguments, Variable]
):
    metadata, output_file = create_variable_metadata(
        general_arguments=arguments[0],
        data=arguments[0]["data"],
        variable=arguments[1],
        variable_type="categorical",
    )

    with open(output_file, "w", encoding="utf-8") as file:
        dump(metadata, fp=file, ensure_ascii=False)
