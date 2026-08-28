# -*- coding: utf-8 -*-
"""English labels for cooling shelter (CS) attributes from Excel/CSV source."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from integrated.build_cs_english import (
    ADDRESSES,
    NAMES,
    REST_SPACE,
    translate_notes,
    translate_schedule,
)

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "ジオコーディング" / "cooling_shelters.csv"


@lru_cache(maxsize=1)
def _load() -> dict[int, dict[str, str]]:
    catalog: dict[int, dict[str, str]] = {}
    if not CSV_PATH.is_file():
        return catalog
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["cs_id"])
            catalog[cid] = {
                "cs_name": NAMES[cid],
                "address": ADDRESSES[cid],
                "open_time": translate_schedule(row.get("open_time") or ""),
                "rest_space": REST_SPACE.get(cid, (row.get("rest_space") or "").strip()),
                "notes": translate_notes(row.get("notes") or ""),
            }
    return catalog


def cs_english_props(cs_id: int) -> dict[str, str] | None:
    return _load().get(int(cs_id))


def localize_cs_properties(props: dict[str, Any], lang: str = "ja") -> dict[str, Any]:
    out = dict(props)
    if (lang or "ja").lower() != "en":
        return out
    cid = props.get("cs_id")
    if cid is None:
        return out
    en = cs_english_props(int(cid))
    if not en:
        return out
    for key in ("cs_name", "address", "open_time", "rest_space", "notes"):
        val = en.get(key)
        if val is not None and str(val).strip():
            out[key] = val
    out["lang"] = "en"
    return out


def localize_geojson_feature(feature: dict[str, Any], lang: str = "ja") -> dict[str, Any]:
    feat = dict(feature)
    props = feat.get("properties") or {}
    feat["properties"] = localize_cs_properties(props, lang)
    return feat


def localize_visit_block(visit: dict[str, Any] | None, lang: str = "ja") -> dict[str, Any] | None:
    if not visit:
        return visit
    out = dict(visit)
    if (lang or "ja").lower() != "en":
        return out
    cid = visit.get("cs_id")
    if cid is None:
        return out
    en = cs_english_props(int(cid))
    if not en:
        return out
    if en.get("cs_name"):
        out["cs_name"] = en["cs_name"]
    if en.get("address"):
        out["cs_address"] = en["address"]
    for key in ("open_time", "rest_space", "notes"):
        if en.get(key):
            out[key] = en[key]
    return out
