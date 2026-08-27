"""
main.py
-------
Convenience entry point for the Personal Loan Portfolio Analysis project.

Run the full analysis pipeline with:

    python main.py

Or import and run programmatically:

    from main import main
    main()
"""

import sys
from pathlib import Path

# Make src/ importable when running main.py directly.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from loan_analysis import main as run_analysis


def main() -> None:
    """Execute the full personal-loan analysis pipeline."""
    run_analysis()


if __name__ == "__main__":
    main()
