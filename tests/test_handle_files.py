from pathlib import Path
from unittest import TestCase

from pandas import DataFrame

from transfer_statistics.handle_files import (
    apply_value_labels,
    get_variable_combinations,
    read_value_label_metadata,
    read_variable_metadata,
)
from transfer_statistics.types import LabeledVariable, VariableMetadata

EXPECTED_GROUPS = {
    "age_gr": LabeledVariable(
        variable="age_gr",
        label="",
        label_de="Altersgruppe",
        values=[1, 2, 3, 4],
        value_labels=["17-29", "30-45", "46-65", "66"],
        value_labels_de=["17-29", "30-45", "46-65", "66"],
    ),
    "bildungsniveau": LabeledVariable(
        variable="bildungsniveau",
        label="",
        label_de="Bildungsniveau",
        values=[1, 2, 3, 4],
        value_labels=[
            "No School Degree Yet",
            "Secondary School Degree",
            "Intermediate School Degree",
            "Technical School Degree",
        ],
        value_labels_de=[
            "(noch) kein Abschluss",
            "Hauptschule",
            "Realschule",
            "(Fach-)Abitur",
        ],
    ),
}


EXPECTED_METADATA: VariableMetadata = {
    "categorical": [
        {
            "dataset": "p_statistics",
            "name": "e11102",
            "label": "Employment Status of Individual",
            "label_de": "Beschäftigungsstatus",
        },
        {
            "dataset": "p_statistics",
            "name": "e11103",
            "label": "Employment Level of Individual",
            "label_de": "Beschäftigungslevel",
        },
        {
            "dataset": "p_statistics",
            "name": "erwst",
            "label": "Employment Status",
            "label_de": "Erwerbsstatus",
        },
    ],
    "numerical": [
        {
            "dataset": "p_statistics",
            "name": "agre",
            "label": "agreeableness",
            "label_de": "Verträglichkeit",
        },
        {
            "dataset": "p_statistics",
            "name": "bmi",
            "label": "Body Mass Index (BMI)",
            "label_de": "Body Mass Index (BMI)",
        },
        {
            "dataset": "p_statistics",
            "name": "conc",
            "label": "Conscientiousness",
            "label_de": "Gewissenhaftigkeit ",
        },
        {
            "dataset": "p_statistics",
            "name": "d11109",
            "label": "Number of Years of Education",
            "label_de": "Jahre in Bildung",
        },
        {
            "dataset": "p_statistics",
            "name": "extr",
            "label": "Extraversion",
            "label_de": "Extraversion",
        },
    ],
    "group": [
        {
            "dataset": "p_statistics",
            "name": "age_gr",
            "label": "",
            "label_de": "Altersgruppe",
        },
        {
            "dataset": "p_statistics",
            "name": "bildungsniveau",
            "label": "",
            "label_de": "Bildungsniveau",
        },
        {
            "dataset": "p_statistics",
            "name": "bula_h",
            "label": "",
            "label_de": "Bundesland",
        },
        {
            "dataset": "p_statistics",
            "name": "corigin",
            "label": "Country of birth",
            "label_de": "Geburtsland",
        },
    ],
}

expected_combinations = [
    (
        {
            "dataset": "p_statistics",
            "name": "age_gr",
            "label": "",
            "label_de": "Altersgruppe",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "bildungsniveau",
            "label": "",
            "label_de": "Bildungsniveau",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "bula_h",
            "label": "",
            "label_de": "Bundesland",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "corigin",
            "label": "Country of birth",
            "label_de": "Geburtsland",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "age_gr",
            "label": "",
            "label_de": "Altersgruppe",
        },
        {
            "dataset": "p_statistics",
            "name": "bildungsniveau",
            "label": "",
            "label_de": "Bildungsniveau",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "age_gr",
            "label": "",
            "label_de": "Altersgruppe",
        },
        {
            "dataset": "p_statistics",
            "name": "bula_h",
            "label": "",
            "label_de": "Bundesland",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "age_gr",
            "label": "",
            "label_de": "Altersgruppe",
        },
        {
            "dataset": "p_statistics",
            "name": "corigin",
            "label": "Country of birth",
            "label_de": "Geburtsland",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "bildungsniveau",
            "label": "",
            "label_de": "Bildungsniveau",
        },
        {
            "dataset": "p_statistics",
            "name": "bula_h",
            "label": "",
            "label_de": "Bundesland",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "bildungsniveau",
            "label": "",
            "label_de": "Bildungsniveau",
        },
        {
            "dataset": "p_statistics",
            "name": "corigin",
            "label": "Country of birth",
            "label_de": "Geburtsland",
        },
    ),
    (
        {
            "dataset": "p_statistics",
            "name": "bula_h",
            "label": "",
            "label_de": "Bundesland",
        },
        {
            "dataset": "p_statistics",
            "name": "corigin",
            "label": "Country of birth",
            "label_de": "Geburtsland",
        },
    ),
]


class TestHandleFiles(TestCase):
    variables_csv: Path
    variable_labels_csv: Path

    def setUp(self) -> None:
        self.variables_csv = Path("./tests/testdata/variables.csv").absolute()
        self.variable_labels_csv = Path(
            "./tests/testdata/variable_categories.csv"
        ).absolute()
        return super().setUp()

    def test_read_variables_metadata(self):
        result = read_variable_metadata(self.variables_csv, "p_statistics")
        self.assertDictEqual(EXPECTED_METADATA, result)

    def test_read_value_label_metadata(self):
        metadata = read_variable_metadata(self.variables_csv, "p_statistics")
        result, _ = read_value_label_metadata(self.variable_labels_csv, metadata)
        self.assertDictEqual(EXPECTED_GROUPS, result)

    def test_get_variable_combinations(self):
        result = get_variable_combinations(EXPECTED_METADATA)

        self.assertEqual(expected_combinations, result)

    def test_apply_value_labels(self):
        data = {
            "year": [1999, 1999, 1999],
            "age_gr": [1, 2, 3],
            "mean": [2, 3, 4],
        }
        expected_data = {
            "year": [1999, 1999, 1999],
            "age_gr": ["17-29", "30-45", "46-65"],
            "mean": [2, 3, 4],
        }
        dataframe_input = DataFrame(data)
        expected_dataframe = DataFrame(expected_data)
        grouping_input = ("age_gr",)
        result = apply_value_labels(dataframe_input, EXPECTED_GROUPS, grouping_input)
        self.assertTrue(expected_dataframe.equals(result))
        data = {
            "year": [1999, 1999, 1999],
            "age_gr": [1, 2, 3],
            "bildungsniveau": [1, 1, 1],
            "mean": [2, 3, 4],
        }
        expected_data = {
            "year": [1999, 1999, 1999],
            "age_gr": ["17-29", "30-45", "46-65"],
            "bildungsniveau": [
                "(noch) kein Abschluss",
                "(noch) kein Abschluss",
                "(noch) kein Abschluss",
            ],
            "mean": [2, 3, 4],
        }
        dataframe_input = DataFrame(data)
        expected_dataframe = DataFrame(expected_data)
        grouping_input = ("age_gr", "bildungsniveau")
        result = apply_value_labels(dataframe_input, EXPECTED_GROUPS, grouping_input)
        self.assertTrue(expected_dataframe.equals(result))
