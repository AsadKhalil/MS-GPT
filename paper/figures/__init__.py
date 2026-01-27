"""Paper figures and tables generation."""

from .generate_tables import (
    generate_dataset_stats_table,
    generate_retrieval_results_table,
    generate_faithfulness_table,
    generate_all_tables,
)

__all__ = [
    "generate_dataset_stats_table",
    "generate_retrieval_results_table",
    "generate_faithfulness_table",
    "generate_all_tables",
]
