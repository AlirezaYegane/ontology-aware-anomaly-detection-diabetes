"""
Filter Tracker Module

Utility for tracking how the dataset is filtered at each preprocessing step.
It records row counts before/after each operation, tracks final class balance,
and can export human-readable reports (console, JSON, Markdown).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


StepInfo = Dict[str, Union[str, int, float]]


class FilterTracker:
    """
    Track data filtering steps throughout the preprocessing pipeline.

    Features
    --------
    - Per-step tracking of row counts (before / after / removed / % removed)
    - Class balance summary for the final target
    - Console table summary
    - JSON and Markdown report export
    """

    def __init__(self) -> None:
        """Initialize an empty filter tracker."""
        self.steps: List[StepInfo] = []
        self.class_balance: Optional[Dict[str, Union[int, float]]] = None

    # --------------------------------------------------------------------- #
    # Tracking methods
    # --------------------------------------------------------------------- #
    def track_step(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        step_name: str,
        description: str = "",
        verbose: bool = True,
    ) -> None:
        """
        Record a single filtering step.

        Parameters
        ----------
        df_before :
            DataFrame before the filtering operation.
        df_after :
            DataFrame after the filtering operation.
        step_name :
            Short identifier for the step (e.g. ``"drop_missing"``).
        description :
            Human-readable description used in reports.
            If empty, ``step_name`` is reused.
        verbose :
            If True, print a one-line summary to the console.
        """
        n_before = len(df_before)
        n_after = len(df_after)
        n_removed = n_before - n_after
        pct_removed = (n_removed / n_before * 100.0) if n_before > 0 else 0.0

        step_info: StepInfo = {
            "step_name": step_name,
            "description": description or step_name,
            "n_before": n_before,
            "n_after": n_after,
            "n_removed": n_removed,
            "pct_removed": round(pct_removed, 2),
        }
        self.steps.append(step_info)

        if verbose:
            print(
                f"  [{step_name}] {n_before:,} → {n_after:,} rows "
                f"(-{n_removed:,}, -{pct_removed:.1f}%)"
            )

    def track_class_balance(self, y: pd.Series, verbose: bool = True) -> None:
        """
        Track the class balance of the final target variable.

        Parameters
        ----------
        y :
            Binary target series (0 / 1).
        verbose :
            If True, print a short class balance summary.
        """
        total = len(y)
        n_positive = int(y.sum())
        n_negative = total - n_positive
        positive_ratio = (n_positive / total * 100.0) if total > 0 else 0.0

        self.class_balance = {
            "total_samples": total,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "positive_ratio": round(positive_ratio, 2),
        }

        if verbose:
            print(
                f"\n  Class balance: {n_positive:,} positive ({positive_ratio:.1f}%) / "
                f"{n_negative:,} negative ({100.0 - positive_ratio:.1f}%)"
            )

    # --------------------------------------------------------------------- #
    # Summary accessors
    # --------------------------------------------------------------------- #
    def get_summary_dict(self) -> Dict[str, object]:
        """
        Return a dictionary with all filtering steps and class balance.

        Returns
        -------
        dict
            {
                "filtering_steps": [...],
                "class_balance": {...} or None
            }
        """
        return {
            "filtering_steps": self.steps,
            "class_balance": self.class_balance,
        }

    def print_summary_table(self) -> None:
        """Print a formatted table of all filtering steps to the console."""
        print("\n" + "=" * 80)
        print("  DATA FILTERING SUMMARY")
        print("=" * 80 + "\n")

        # Header
        print(f"{'Step':<40} {'Before':<10} {'After':<10} {'Removed':<10} {'%':<8}")
        print("-" * 80)

        # Rows
        for step in self.steps:
            print(
                f"{step['description']:<40} "
                f"{step['n_before']:<10,} "
                f"{step['n_after']:<10,} "
                f"{step['n_removed']:<10,} "
                f"{step['pct_removed']:<8.1f}%"
            )

        print("\n" + "=" * 80)
        print("  CLASS BALANCE SUMMARY")
        print("=" * 80 + "\n")

        if self.class_balance:
            cb = self.class_balance
            total = cb["total_samples"]
            n_pos = cb["n_positive"]
            n_neg = cb["n_negative"]
            pos_ratio = cb["positive_ratio"]
            print(f"Total samples:     {total:,}")
            print(f"Positive (y=1):    {n_pos:,} ({pos_ratio:.1f}%)")
            print(f"Negative (y=0):    {n_neg:,} ({100.0 - pos_ratio:.1f}%)")
        else:
            print("Class balance not tracked yet.")

        print("\n" + "=" * 80 + "\n")

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def save_summary_json(self, filepath: Union[str, Path]) -> None:
        """
        Save the summary to a JSON file.

        Parameters
        ----------
        filepath :
            Destination path for the JSON file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.get_summary_dict(), f, indent=2)

        print(f"Saved JSON summary: {path}")

    def save_summary_markdown(self, filepath: Union[str, Path]) -> None:
        """
        Save the summary to a Markdown file.

        Parameters
        ----------
        filepath :
            Destination path for the Markdown file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "# Data Filtering Summary Report",
            "",
            "## Filtering Steps",
            "",
            "| Step | Before | After | Removed | % Removed |",
            "|------|-------:|------:|--------:|----------:|",
        ]

        for step in self.steps:
            lines.append(
                f"| {step['description']} | "
                f"{step['n_before']:,} | "
                f"{step['n_after']:,} | "
                f"{step['n_removed']:,} | "
                f"{step['pct_removed']:.1f}% |"
            )

        lines.append("")
        lines.append("## Class Balance")
        lines.append("")

        if self.class_balance:
            cb = self.class_balance
            lines.extend(
                [
                    f"- **Total samples**: {cb['total_samples']:,}",
                    f"- **Positive class (y=1)**: {cb['n_positive']:,} "
                    f"({cb['positive_ratio']:.1f}%)",
                    f"- **Negative class (y=0)**: {cb['n_negative']:,} "
                    f"({100.0 - cb['positive_ratio']:.1f}%)",
                ]
            )
        else:
            lines.append("Class balance not tracked yet.")

        lines.append("")
        lines.append("---")
        lines.append("*Generated automatically by FilterTracker*")
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved Markdown summary: {path}")

    def save_summary(
        self,
        output_dir: Union[str, Path],
        basename: str = "data_filtering_summary",
    ) -> None:
        """
        Save JSON + Markdown summaries in the given directory.

        Parameters
        ----------
        output_dir :
            Target directory for the report files.
        basename :
            Base filename (without extension).
        """
        output_path = Path(output_dir)
        self.save_summary_json(output_path / f"{basename}.json")
        self.save_summary_markdown(output_path / f"{basename}.md")
