"""Template for exporting measurements from MongoDB.

Do not commit credentials. Configure the connection through environment variables.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from pymongo import MongoClient


def export_building(building: str, output_dir: str = "data/raw") -> Path:
    uri = os.environ["MONGODB_URI"]
    database = os.environ.get("MONGODB_DATABASE", "sensor_db")
    collection_name = os.environ.get("MONGODB_COLLECTION", "ElectricityProd_v2")
    client = MongoClient(uri)
    docs = list(client[database][collection_name].find({"building": building}))
    if not docs:
        raise ValueError(f"No documents found for building={building!r}")
    df = pd.DataFrame(docs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{building}_{datetime.now():%Y%m%d}.csv"
    df.to_csv(path, index=False)
    return path
