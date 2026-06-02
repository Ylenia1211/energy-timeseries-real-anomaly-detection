from __future__ import annotations

from pathlib import Path

from tsad_feature.cli import main

# Preferisci la CLI da terminale:
# tsad-feature real-run --csv data/raw/ED14_20240426.csv --meter-id ARCH_FM --value-col TotW ...

if __name__ == "__main__":
    raise SystemExit(
        "Esegui da terminale: tsad-feature real-run --csv data/raw/ED14_20240426.csv "
        "--meter-id ARCH_FM --value-col TotW --resample 15min --window-size 96 "
        "--outdir outputs/ED14_ARCH_FM_TotW"
    )
