from pathlib import Path
from unittest import TestCase

from transfer_statistics.handle_files import read_variable_metadata

from transfer_statistics.types import VariableMetadata


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
    "numeric": [
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
            "label_de": "Bildungsniveu",
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


class TestHandleFiles(TestCase):
    def test_read_variables_metadata(self):
        variables_csv = Path("./testdata/variables.csv").absolute()
        result = read_variable_metadata(variables_csv)
        self.assertDictEqual(EXPECTED_METADATA, result)
