from pathlib import Path

from pandas import DataFrame


from transfer_statistics.types import VariableMetadata
from transfer_statistics.handle_files import get_variable_combinations


def calculate_numeric_statistics(
    data: DataFrame, metadata: VariableMetadata, output_folder: Path
) -> None:
    ...
