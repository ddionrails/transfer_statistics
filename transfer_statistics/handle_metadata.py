from json import dump

from pandas import DataFrame, to_numeric


from transfer_statistics.types import (
    GeneralArguments,
    Variable,
    ValueLabels,
    MetadataFile,
)


def _get_start_and_end_year(data: DataFrame):
    years = to_numeric(data[data.notnull()]["syear"])
    return int(years.min()), int(years.max())


def create_metadata_file(arguments: tuple[Variable, GeneralArguments]):
    data = arguments[1]["data"]
    variable = arguments[0]
    output_folder = arguments[1]["output_folder"].joinpath(variable["name"])
    output_file = output_folder.joinpath("meta.json")
    value_labels: ValueLabels = arguments[1]["value_labels"]
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
            }
        )
    start_year, end_year = _get_start_and_end_year(data[["syear", variable["name"]]])
    metadata: MetadataFile = {
        "dataset": variable["dataset"],
        "title": variable["label"],
        "label": variable["label"],
        "label_de": variable["label_de"],
        "variable": variable["name"],
        "dimensions": dimensions,
        "groups": groups,
        "start_year": start_year,
        "end_year": end_year,
    }
    with open(output_file, "w", encoding="utf-8") as file:
        dump(metadata, fp=file, ensure_ascii=False)
