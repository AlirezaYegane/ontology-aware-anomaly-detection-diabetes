"""
Filter Tracker Module - Tracks data filtering operations and generates reports.

This module provides instrumentation for the preprocessing pipeline to track
how the dataset is filtered at each step, from raw data to final feature matrix.
"""

import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path
import json


class FilterTracker:
    """
    Tracks data filtering steps throughout the preprocessing pipeline.
    
    Provides:
    - Per-step tracking of row counts (before/after/removed/%)
    - Summary generation for console and file output
    - Class balance tracking for final dataset
    """
    
    def __init__(self):
        """Initialize the filter tracker."""
        self.steps: List[Dict[str, Union[str, int, float]]] = []
        self.class_balance: Optional[Dict[str, Union[int, float]]] = None
    
    def track_step(self, 
                   df_before: pd.DataFrame, 
                   df_after: pd.DataFrame, 
                   step_name: str, 
                   description: str = "",
                   verbose: bool = True) -> None:
        """
        Track a filtering step by recording before/after counts.
        
        Args:
            df_before: DataFrame before the filtering operation
            df_after: DataFrame after the filtering operation
            step_name: Short name for the step (e.g., "drop_missing")
            description: Human-readable description of the step
            verbose: If True, print a summary line to console
        """
        n_before = len(df_before)
        n_after = len(df_after)
        n_removed = n_before - n_after
        pct_removed = (n_removed / n_before * 100) if n_before > 0 else 0.0
        
        step_info = {
            'step_name': step_name,
            'description': description or step_name,
            'n_before': n_before,
            'n_after': n_after,
            'n_removed': n_removed,
            'pct_removed': round(pct_removed, 2)
        }
        
        self.steps.append(step_info)
        
        if verbose:
            print(f"  [{step_name}] {n_before:,} → {n_after:,} rows "
                  f"(-{n_removed:,}, -{pct_removed:.1f}%)")
    
    def track_class_balance(self, y: pd.Series, verbose: bool = True) -> None:
        """
        Track the class balance of the final target variable.
        
        Args:
            y: Target variable (binary: 0 or 1)
            verbose: If True, print class balance summary
        """
        total = len(y)
        n_positive = int(y.sum())
        n_negative = total - n_positive
        positive_ratio = (n_positive / total * 100) if total > 0 else 0.0
        
        self.class_balance = {
            'total_samples': total,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'positive_ratio': round(positive_ratio, 2)
        }
        
        if verbose:
            print(f"\n  Class balance: {n_positive:,} positive ({positive_ratio:.1f}%) / "
                  f"{n_negative:,} negative ({100-positive_ratio:.1f}%)")
    
    def get_summary_dict(self) -> Dict:
        """
        Get the complete summary as a dictionary.
        
        Returns:
            Dictionary containing all filtering steps and class balance
        """
        return {
            'filtering_steps': self.steps,
            'class_balance': self.class_balance
        }
    
    def print_summary_table(self) -> None:
        """Print a formatted table of all filtering steps to console."""
        print("\n" + "="*80)
        print("  DATA FILTERING SUMMARY")
        print("="*80 + "\n")
        
        # Header
        print(f"{'Step':<40} {'Before':<10} {'After':<10} {'Removed':<10} {'%':<8}")
        print("-" * 80)
        
        # Rows
        for step in self.steps:
            print(f"{step['description']:<40} "
                  f"{step['n_before']:<10,} "
                  f"{step['n_after']:<10,} "
                  f"{step['n_removed']:<10,} "
                  f"{step['pct_removed']:<8.1f}%")
        
        print("\n" + "="*80)
        print("  CLASS BALANCE SUMMARY")
        print("="*80 + "\n")
        
        if self.class_balance:
            cb = self.class_balance
            print(f"Total samples:     {cb['total_samples']:,}")
            print(f"Positive (y=1):    {cb['n_positive']:,} ({cb['positive_ratio']:.1f}%)")
            print(f"Negative (y=0):    {cb['n_negative']:,} ({100 - cb['positive_ratio']:.1f}%)")
        else:
            print("Class balance not tracked yet.")
        
        print("\n" + "="*80 + "\n")
    
    def save_summary_json(self, filepath: Union[str, Path]) -> None:
        """
        Save the summary to a JSON file.
        
        Args:
            filepath: Path where JSON file should be saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.get_summary_dict(), f, indent=2)
        
        print(f"✅ Saved JSON summary: {filepath}")
    
    def save_summary_markdown(self, filepath: Union[str, Path]) -> None:
        """
        Save the summary to a Markdown file.
        
        Args:
            filepath: Path where Markdown file should be saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "# Data Filtering Summary Report\n",
            "## Filtering Steps\n",
            "| Step | Before | After | Removed | % Removed |",
            "|------|-------:|------:|--------:|----------:|"
        ]
        
        for step in self.steps:
            lines.append(
                f"| {step['description']} | "
                f"{step['n_before']:,} | "
                f"{step['n_after']:,} | "
                f"{step['n_removed']:,} | "
                f"{step['pct_removed']:.1f}% |"
            )
        
        lines.extend([
            "\n## Class Balance\n"
        ])
        
        if self.class_balance:
            cb = self.class_balance
            lines.extend([
                f"- **Total samples**: {cb['total_samples']:,}",
                f"- **Positive class (y=1)**: {cb['n_positive']:,} ({cb['positive_ratio']:.1f}%)",
                f"- **Negative class (y=0)**: {cb['n_negative']:,} ({100 - cb['positive_ratio']:.1f}%)"
            ])
        else:
            lines.append("Class balance not tracked yet.")
        
        lines.append(f"\n---\n*Generated automatically by FilterTracker*\n")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Saved Markdown summary: {filepath}")
    
    def save_summary(self, 
                     output_dir: Union[str, Path], 
                     basename: str = "data_filtering_summary") -> None:
        """
        Save summaries in both JSON and Markdown formats.
        
        Args:
            output_dir: Directory where summaries should be saved
            basename: Base name for the files (without extension)
        """
        output_dir = Path(output_dir)
        self.save_summary_json(output_dir / f"{basename}.json")
        self.save_summary_markdown(output_dir / f"{basename}.md")
