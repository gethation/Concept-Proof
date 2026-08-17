"""Rebuild every report from the current data, in one command.

    python scripts/make_reports.py            # all three
    python scripts/make_reports.py --only ccf_umc

The point of this script is that the report set has one entry point. Before it
existed, half the reports were produced by throwaway scratchpad scripts, so a
figure could not be corrected without rewriting its generator from scratch --
which is exactly how the set drifted into several data vintages at once.

Refresh the data first. Every command below reads and writes the canonical path
for its series, so re-running is safe and idempotent:

    python scripts/ingest/taifex_1m.py --product CCF
    python scripts/ingest/taifex_1m.py --product QFF
    python scripts/ingest/tv_umc.py
    python scripts/ingest/tv_ccf_umc.py
    python scripts/ingest/ccxt_ohlcv.py --feed binance_tsmusdtp
    python scripts/ingest/ccxt_ohlcv.py --feed bitopro_usdttwd
    python scripts/features/spread.py --pair ccf_umc --interval 1m --weekend-policy none
    python scripts/features/spread.py --pair qff_tsm --interval 1m

These commands used to carry explicit --out paths naming a dated capture
(``..._0812.csv``). That is what broke the refresh: the download step wrote the
undated default while the spread step read the dated file, so running the
documented sequence updated a file nothing else read, and the reports kept
building on a stale vintage. There are no dated filenames left to get wrong.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import comparison as generate_comparison_report  # noqa: E402
from report import pair as generate_pair_report  # noqa: E402

REPORTS = ["ccf_umc", "qff_tsm", "comparison"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the report set.")
    ap.add_argument(
        "--only", choices=REPORTS, action="append",
        help="Build just these (repeatable). Default: all three.",
    )
    args = ap.parse_args(argv)
    wanted = args.only or REPORTS

    for name in wanted:
        started = time.monotonic()
        print(f"\n=== {name} " + "=" * (56 - len(name)))
        if name == "comparison":
            generate_comparison_report.main([])
        else:
            generate_pair_report.build(generate_pair_report.PAIRS[name])
        print(f"    {time.monotonic() - started:.1f}s")

    print("\nAll requested reports rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
