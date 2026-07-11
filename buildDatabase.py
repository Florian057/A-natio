"""Manager script — the analog of moneyball/buildDatabase.ipynb, but a plain
.py instead of a notebook (both data sources here are frozen/static mirrors,
so there's no "which seasons changed" incremental logic to own; a notebook
would just add friction for running this end-to-end in one shot).

Run with `python buildDatabase.py` to (re)populate data/. Safe to re-run —
each build() skips tables that already exist on disk; pass force=True to
refetch everything.
"""

from __future__ import annotations

import scrapeFootballData as football
import scrapeInternationalResults as intl


def main(force: bool = False) -> None:
    print("=== football (club side, Big-5 leagues) ===")
    football.build(force=force)
    print("\n=== international results (validation side) ===")
    intl.build(force=force)


if __name__ == "__main__":
    main()
