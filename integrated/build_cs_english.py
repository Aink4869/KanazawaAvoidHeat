# -*- coding: utf-8 -*-
"""Build cooling_shelters_en.json from cooling_shelters.csv (Excel source)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "ジオコーディング" / "cooling_shelters.csv"
OUT_PATH = Path(__file__).resolve().parent / "cooling_shelters_en.json"

# Facility names (cs_id -> English name)
NAMES: dict[int, str] = {
    1: "Kanazawa City Hall — Main Building",
    2: "Kanazawa City Hall — Second Building",
    3: "Ekimae-Nishi Welfare & Health Center",
    4: "Izumino Welfare & Health Center",
    5: "Motomachi Welfare & Health Center",
    6: "Kanazawa Health Plaza Otemachi East Wing",
    7: "Kanazawa Umimirai Library",
    8: "Izumino Library",
    9: "Johoku Children's Hall",
    10: "Kanazawa Student Town Civic Exchange Hall",
    11: "Omicho Exchange Plaza",
    12: "Kanazawa Pool",
    13: "Narutodai Civic Gymnasium",
    14: "Ishikawa Prefectural Government (19F Observatory Lobby)",
    15: "Ishikawa Prefectural Library",
    16: "Nagadobe Youth Exchange Center",
    17: "West District Civic Rest House",
    18: "Shojusou",
    19: "Kakujuen",
    20: "Kanazawa Mirai Town Creation Hall",
    21: "Monozukuri Hall",
    22: "Agricultural Center",
    23: "Central Community Center Hikoso Branch",
    24: "Education Plaza Tomikichi",
    25: "Tomuro Recycle Plaza",
    26: "Kanazawa Workers' Plaza",
    27: "Senjukan",
    28: "Kanazawa Central Tourist Information Center",
    29: "Kanazawa Port Cruise Terminal",
    30: "Ishikawa Children's Exchange Center",
    31: "Kanazawa Castle Park — Tsurunomaru Rest Hall",
    32: "Kanazawa Castle Park — Rest House Gyokusen-an",
    33: "Kanazawa Castle Park — Ishikawa Gate Entrance Rest Area",
    34: "Ishikawa Eco House",
    35: "Kanazawa Welfare Equipment Information Plaza",
    36: "Nishi Chaya Tourist Parking",
    37: "21st Century Museum of Contemporary Art, Kanazawa",
    38: "Welcia Kanazawa Gaku Shinbo Store",
    39: "Welcia Kanazawa Shin-hombo Store",
    40: "Welcia Pharmacy Kanazawa Shin-hombo",
    41: "Welcia Kanazawa Naoe Store",
    42: "Welcia Pharmacy Kanazawa Naoe",
    43: "Welcia Kanazawa Kibiyano Store",
    44: "Welcia Pharmacy Kanazawa Kibiyano",
    45: "Welcia Pharmacy Kanazawa Miiike",
    46: "Welcia Pharmacy Kanazawa Shijuman",
    47: "Welcia Pharmacy Kanazawa Izumicho",
    48: "Welcia Pharmacy Kanazawa Okuwa",
    49: "Welcia Pharmacy Kanazawa Suemachi",
    50: "Welcia Pharmacy Kanazawa Arimatsu",
    51: "AEON Morinosato Store",
    52: "AEON Town Kanazawa Shikino",
    53: "Meiji Yasuda Village Kanazawa",
    54: "Arukafe",
    55: "Kanazawa Forus",
    56: "Class Place Korinbo (Museum Shop & Café KAMU kanazawa)",
    57: "Orange Heart Laundry Ake-myomachi",
}

ADDRESSES: dict[int, str] = {
    1: "1-1-1 Hirozaka, Kanazawa, Ishikawa",
    2: "1-1-1 Kakinokibatake, Kanazawa, Ishikawa",
    3: "3-4-25 Nishinen, Kanazawa, Ishikawa",
    4: "1-2-22 Izumigaoka, Kanazawa, Ishikawa",
    5: "1-12-12 Motomachi, Kanazawa, Ishikawa",
    6: "3-21 Otemachi, Kanazawa, Ishikawa",
    7: "Terachumachi I-1, Kanazawa, Ishikawa",
    8: "4-22-22 Izuminocho, Kanazawa, Ishikawa",
    9: "8-11 Kosakamachi-Nishi, Kanazawa, Ishikawa",
    10: "2-5-17 Katamachi, Kanazawa, Ishikawa",
    11: "88 Aosugimachi, Kanazawa, Ishikawa",
    12: "Ha-55 Isobemachi, Kanazawa, Ishikawa",
    13: "285 Narutodai, Kanazawa, Ishikawa",
    14: "1-1 Kuratsuki, Kanazawa, Ishikawa",
    15: "2-43-1 Kodatsuno, Kanazawa, Ishikawa",
    16: "3-3-3 Nagamachi, Kanazawa, Ishikawa",
    17: "Ha-256 Torikemachi, Kanazawa, Ishikawa",
    18: "3-3-33 Kanaokita 3-chome, Kanazawa, Ishikawa",
    19: "Nu-1 Nukatani-cho, Kanazawa, Ishikawa",
    20: "3-11-1 Nomachi, Kanazawa, Ishikawa",
    21: "4-80-1 Awagasakimachi, Kanazawa, Ishikawa",
    22: "Higashi1471 Shimyasuhamacho, Kanazawa, Ishikawa",
    23: "1-15-5 Hikoso 1-chome, Kanazawa, Ishikawa",
    24: "3-10-1 Tomikichi 3-chome, Kanazawa, Ishikawa",
    25: "Ha-604 Tomuro-Shimbo, Kanazawa, Ishikawa",
    26: "3-2-20 Kita-Yasue, Kanazawa, Ishikawa",
    27: "2-1 Higashinagaecho-be, Kanazawa, Ishikawa",
    28: "4-1 Minamimachi, Kanazawa, Ishikawa",
    29: "Ri-65 Muryojimachi, Kanazawa, Ishikawa",
    30: "11-8 Hojima-cho, Kanazawa, Ishikawa",
    31: "1-1 Marunouchi, Kanazawa, Ishikawa",
    32: "1-1 Marunouchi, Kanazawa, Ishikawa",
    33: "1-1 Marunouchi, Kanazawa, Ishikawa",
    34: "2-1 Kuratsuki 2-chome, Kanazawa, Ishikawa",
    35: "1-10-1 Honmachi 1-chome, Kanazawa, Ishikawa",
    36: "2-25-1 Nomachi 2-chome, Kanazawa, Ishikawa",
    37: "1-2-1 Hirozaka, Kanazawa, Ishikawa",
    38: "3-188 Gaku Shinbo 3-chome, Kanazawa, Ishikawa",
    39: "3-152-1 Shin-hombo 3-chome, Kanazawa, Ishikawa",
    40: "3-152-1 Shin-hombo 3-chome, Kanazawa, Ishikawa",
    41: "Minami1-44 Naoe, Kanazawa, Ishikawa",
    42: "Minami1-44 Naoe, Kanazawa, Ishikawa",
    43: "4-223 Kibiyano 4-chome, Kanazawa, Ishikawa",
    44: "4-223 Kibiyano 4-chome, Kanazawa, Ishikawa",
    45: "312 Miiike Sakae-cho, Kanazawa, Ishikawa",
    46: "4-228 Shijuman 4-chome, Kanazawa, Ishikawa",
    47: "I-222 Izumicho, Kanazawa, Ishikawa",
    48: "2-47 Okuwa 2-chome, Kanazawa, Ishikawa",
    49: "16-12-1 Suemachi, Kanazawa, Ishikawa",
    50: "4-13-13 Arimatsu 4-chome, Kanazawa, Ishikawa",
    51: "1-70 Morinosato, Kanazawa, Ishikawa",
    52: "1-55 Toita-Nishi, Kanazawa, Ishikawa",
    53: "4-47 Minamimachi, Kanazawa, Ishikawa",
    54: "1-20-20 Zosen, Kanazawa, Ishikawa",
    55: "3-1 Horikawa-Shinmachi, Kanazawa, Ishikawa",
    56: "2-1-1 Korinbo, Kanazawa, Ishikawa",
    57: "1-240 Akemyomachi, Kanazawa, Ishikawa",
}

REST_SPACE: dict[int, str] = {
    24: "Building 1, 1st floor space",
    27: "1st floor lobby",
    38: "Wel Cafe",
    39: "Wel Cafe",
    40: "Pharmacy waiting area",
    41: "Wel Cafe",
    42: "Pharmacy waiting area",
    43: "Wel Cafe",
    44: "Pharmacy waiting area",
    45: "Pharmacy waiting area",
    46: "Pharmacy waiting area",
    47: "Pharmacy waiting area",
    48: "Pharmacy waiting area",
    49: "Pharmacy waiting area",
    50: "Pharmacy waiting area",
    51: "1st floor community plaza",
    52: "MaxValu wing shared corridor",
    53: "Multipurpose space",
    54: "Community space",
    55: "Seating areas on each floor",
    56: "Museum Shop & Café open space",
    57: "Inside the laundry",
}


def translate_schedule(ja: str) -> str:
    if not ja or not ja.strip():
        return ""
    s = ja.strip()
    reps = [
        ("期間中無休", "Open daily during the program period"),
        ("月～金", "Mon–Fri"),
        ("月～土", "Mon–Sat"),
        ("月～日", "Mon–Sun"),
        ("火～日", "Tue–Sun"),
        ("月火木～日", "Mon, Tue, Thu–Sun"),
        ("月火木土", "Mon, Tue, Thu, Sat"),
        ("月火木金", "Mon, Tue, Thu, Fri"),
        ("月水木金", "Mon, Wed, Thu, Fri"),
        ("月水木金土日", "Mon, Wed, Thu, Fri, Sat, Sun"),
        ("月火木金土日", "Mon, Tue, Thu, Fri, Sat, Sun"),
        ("9:00", "9:00"),
    ]
    for a, b in reps:
        s = s.replace(a, b)
    s = s.replace("／", " / ")
    s = s.replace("～", "–")
    s = re.sub(r"(\d+)時", r"\1:00", s)
    return s


def translate_notes(ja: str) -> str:
    if not ja or not ja.strip():
        return ""
    s = ja.strip()
    reps = [
        ("祝日休館", "Closed on public holidays"),
        ("月・祝休館", "Closed Mon and public holidays"),
        ("水曜休館", "Closed Wed"),
        ("土日祝休館", "Closed Sat, Sun and public holidays"),
        ("特定日休館", "Closed on designated days"),
        ("月曜祝日は開館・翌日休館", "Open if Mon is a holiday; closed the next day"),
        ("月曜祝日は開館・直後の平日休館", "Open if Mon is a holiday; closed the following weekday"),
        ("火曜祝日は開館", "Open if Tue is a holiday"),
        ("水曜祝日は開館・直後の平日休館", "Open if Wed is a holiday; closed the following weekday"),
        ("7/19～8/31は9:30開館", "Opens 9:30 from Jul 19–Aug 31"),
        ("利用者含む", "Includes library users in capacity count"),
        ("見学者優先", "Priority for visitors on tours"),
        ("飲食不可", "No food or drinks"),
        ("食事不可", "No meals"),
        ("館内飲食不可（水分補給のみ可）", "No food in building (water only)"),
        ("館内飲食不可（交流ゾーンのみ水分補給可）", "No food in galleries (water allowed in exchange zone only)"),
        ("11:30～13:30食事可", "Meals allowed 11:30–13:30"),
        ("イベント開催日除く", "Except event days"),
        ("駐車場なし", "No parking"),
        ("金沢ニューグランドビル1階", "Kanazawa New Grand Bldg, 1st floor"),
        (
            "一般開放がない日(7/5・6・14・26・27・29、8/18・25、9/1・8・16・22・29)除く",
            "Except days without public access (Jul 5–6, 14, 26–27, 29; Aug 18, 25; Sep 1, 8, 16, 22, 29)",
        ),
        ("3人掛けベンチ12台", "12 three-seat benches"),
    ]
    for a, b in reps:
        s = s.replace(a, b)
    return s


def main() -> None:
    out: dict[str, dict[str, str]] = {}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["cs_id"])
            out[str(cid)] = {
                "cs_name": NAMES[cid],
                "address": ADDRESSES[cid],
                "open_time": translate_schedule(row.get("open_time") or ""),
                "rest_space": REST_SPACE.get(cid, (row.get("rest_space") or "").strip()),
                "notes": translate_notes(row.get("notes") or ""),
            }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()
