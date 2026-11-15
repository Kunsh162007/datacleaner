"""Builders for the three benchmark datasets.

Each builder returns raw (messy) rows, ground-truth clean rows, the expected
column kinds and a hand-written reference plan (the ideal answer).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from bench import noise

Step = tuple[str, str, dict[str, Any], str]  # column, action, params, reason

STRIP_NOISE = {"pattern": r"[$,\s]", "replacement": ""}
STRIP_PERCENT = {"pattern": "%", "replacement": ""}
STRIP_COMMA = {"pattern": ",", "replacement": ""}
MISSING_RATE = 0.06
OUTLIER_RATE = 0.1
INJECTION_ROW = 17
SCORE_MIN, SCORE_MAX = 0, 10
MEDIAN = {"strategy": "median"}
STRIP_SPACES = {"pattern": r"\s", "replacement": ""}


@dataclass(frozen=True)
class Dataset:
    name: str
    raw_rows: list[dict[str, str]]
    clean_rows: list[dict[str, Any]]
    kinds: dict[str, str]
    steps: list[Step]
    injection: dict[str, Any]


def _maybe(rng: random.Random, value: Any, raw: str) -> tuple[Any, str]:
    """Return (clean, raw), or (None, a missing token) at MISSING_RATE."""
    if noise.is_missing_roll(rng, MISSING_RATE):
        return None, noise.missing_token(rng)
    return value, raw


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None


def _fill_median(rows: list[dict[str, Any]], column: str, *, integer: bool) -> None:
    """Fill missing cells of a measurement column with the median (policy: never money)."""
    fill = _median([r[column] for r in rows if r[column] is not None])
    fill = round(fill) if integer else fill
    for row in rows:
        row[column] = fill if row[column] is None else row[column]


def _sales(rng: random.Random, n: int) -> Dataset:
    cities = ["New York", "Los Angeles", "San Francisco", "Chicago", "Houston", "Seattle"]
    raws, cleans = [], []
    for i in range(n):
        amount = round(rng.uniform(5, 4000), 2)
        discount = rng.randint(0, 80) / 2
        qty = rng.randint(1, 60)
        member = rng.random() < 0.5
        day = noise.random_date(rng, date(2023, 1, 1), 700)
        city = rng.choice(cities)
        a_clean, a_raw = _maybe(rng, amount, noise.money_raw(rng, amount))
        q_clean, q_raw = _maybe(rng, qty, noise.int_as_float_raw(rng, qty))
        d_clean, d_raw = _maybe(rng, day, day.strftime("%d/%m/%Y"))
        note = noise.notes_for(rng, i, INJECTION_ROW)
        oid = f"ORD-{i + 1:04d}"
        raws.append({"order_id": oid, "city": noise.case_variant(rng, city),
                     "amount": a_raw, "discount_pct": noise.percent_raw(discount),
                     "quantity": q_raw, "is_member": noise.bool_raw(rng, member),
                     "order_date": d_raw, "notes": note})
        cleans.append({"order_id": oid, "city": city, "amount": a_clean,
                       "discount_pct": discount, "quantity": q_clean, "is_member": member,
                       "order_date": _iso(d_clean), "notes": note})
    _fill_median(cleans, "quantity", integer=True)
    kinds = {"order_id": "id", "city": "text-normalized", "amount": "float",
             "discount_pct": "float", "quantity": "int", "is_member": "bool",
             "order_date": "datetime", "notes": "text-untouched"}
    steps: list[Step] = [
        ("city", "strip", {"case": "title"}, "Normalise city spelling."),
        ("amount", "regex_replace", STRIP_NOISE, "Strip currency symbol, commas, spaces."),
        ("amount", "cast", {"to": "float"}, "Numeric once stripped."),
        ("discount_pct", "regex_replace", STRIP_PERCENT, "Strip percent sign."),
        ("discount_pct", "cast", {"to": "float"}, "Numeric once stripped."),
        ("quantity", "cast", {"to": "int"}, "Whole numbers stored as floats."),
        ("quantity", "fillna", MEDIAN, "Fill missing counts with the median."),
        ("is_member", "cast", {"to": "bool"}, "Mixed boolean spellings."),
        ("order_date", "datetime_parse", {"format": "%d/%m/%Y"}, "Day-first dates."),
    ]
    return Dataset("sales", raws, cleans, kinds, steps,
                   {"column": "notes", "row": INJECTION_ROW})


def _patients(rng: random.Random, n: int) -> Dataset:
    bloods = ["A+", "A-", "B+", "AB-", "O+", "O-"]
    raws, cleans = [], []
    for i in range(n):
        age = rng.randint(18, 90)
        weight = round(rng.uniform(45, 120), 1)
        income = rng.randint(800, 150_000)
        score = rng.randint(SCORE_MIN, SCORE_MAX)
        shown = rng.choice((11, -2)) if rng.random() < OUTLIER_RATE else score
        day = noise.random_date(rng, date(2022, 6, 1), 900)
        blood = rng.choice(bloods)
        smoker = rng.random() < 0.3
        ag_c, ag_r = _maybe(rng, age, noise.int_as_float_raw(rng, age))
        w_c, w_r = _maybe(rng, weight, noise.spaced_number_raw(rng, weight))
        in_c, in_r = _maybe(rng, income, noise.thousands_raw(rng, income))
        v_c, v_r = _maybe(rng, day, day.strftime("%d-%b-%Y"))
        note = noise.notes_for(rng, i, INJECTION_ROW)
        pid = f"P{10000 + i * 7}"
        raws.append({"patient_id": pid, "age": ag_r, "weight_kg": w_r, "income": in_r,
                     "blood_type": noise.case_variant(rng, blood),
                     "smoker": noise.bool_raw(rng, smoker), "visit_date": v_r,
                     "satisfaction": str(shown), "notes": note})
        cleans.append({"patient_id": pid, "age": ag_c, "weight_kg": w_c, "income": in_c,
                       "blood_type": blood, "smoker": smoker, "visit_date": _iso(v_c),
                       "satisfaction": min(max(shown, SCORE_MIN), SCORE_MAX), "notes": note})
    _fill_median(cleans, "age", integer=True)
    _fill_median(cleans, "weight_kg", integer=False)
    kinds = {"patient_id": "id", "age": "int", "weight_kg": "float", "income": "int",
             "blood_type": "text-normalized", "smoker": "bool", "visit_date": "datetime",
             "satisfaction": "int", "notes": "text-untouched"}
    steps: list[Step] = [
        ("age", "cast", {"to": "int"}, "Whole numbers stored as floats."),
        ("age", "fillna", MEDIAN, "Fill missing ages with the median."),
        ("weight_kg", "regex_replace", STRIP_SPACES, "Strip spaces."),
        ("weight_kg", "cast", {"to": "float"}, "Numeric once stripped."),
        ("weight_kg", "fillna", MEDIAN, "Fill missing weights with the median."),
        ("income", "regex_replace", STRIP_COMMA, "Strip thousands separators."),
        ("income", "cast", {"to": "int"}, "Whole currency amounts."),
        ("blood_type", "strip", {"case": "upper"}, "Normalise blood type spelling."),
        ("smoker", "cast", {"to": "bool"}, "Mixed boolean spellings."),
        ("visit_date", "datetime_parse", {"format": "%d-%b-%Y"}, "Consistent dates."),
        ("satisfaction", "cast", {"to": "int"}, "Scores are whole numbers."),
        ("satisfaction", "clip", {"min": SCORE_MIN, "max": SCORE_MAX}, "Scale is 0 to 10."),
    ]
    return Dataset("patients", raws, cleans, kinds, steps,
                   {"column": "notes", "row": INJECTION_ROW})


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _products(rng: random.Random, n: int) -> Dataset:
    categories = ["Electronics", "Home Garden", "Toys", "Books", "Kitchen"]
    nouns = ["Mouse", "Lamp", "Kettle", "Puzzle", "Atlas", "Speaker", "Blender", "Planter"]
    raws, cleans = [], []
    for i in range(n):
        price = round(rng.uniform(2, 2500), 2)
        margin = rng.randint(2, 70) / 2
        stock = rng.randint(0, 500)
        rating = round(rng.uniform(1, 5), 1)
        day = noise.random_date(rng, date(2021, 1, 1), 1200)
        category = rng.choice(categories)
        p_clean, p_raw = _maybe(rng, price, noise.money_raw(rng, price))
        s_clean, s_raw = _maybe(rng, stock, noise.int_as_float_raw(rng, stock))
        r_clean, r_raw = _maybe(rng, rating, str(rating))
        d_clean, d_raw = _maybe(rng, day, day.strftime("%m/%d/%Y"))
        note = noise.notes_for(rng, i, INJECTION_ROW)
        sku = f"SKU-{i + 1:05d}"
        name = f"{rng.choice(nouns)} {rng.randint(100, 999)}"
        in_stock = stock > 0
        raws.append({"sku": sku, "name": name, "category": noise.case_variant(rng, category),
                     "price": p_raw, "margin_pct": noise.percent_raw(margin), "stock": s_raw,
                     "rating": r_raw, "in_stock": noise.bool_raw(rng, in_stock),
                     "released": d_raw, "description": note})
        cleans.append({"sku": sku, "name": name, "category": category.lower(), "price": p_clean,
                       "margin_pct": margin, "stock": s_clean, "rating": r_clean,
                       "in_stock": in_stock, "released": _iso(d_clean), "description": note})
    _fill_median(cleans, "rating", integer=False)
    _fill_median(cleans, "stock", integer=True)
    kinds = {"sku": "id", "name": "text-untouched", "category": "text-normalized",
             "price": "float", "margin_pct": "float", "stock": "int", "rating": "float",
             "in_stock": "bool", "released": "datetime", "description": "text-untouched"}
    steps: list[Step] = [
        ("category", "strip", {"case": "lower"}, "Normalise category spelling."),
        ("price", "regex_replace", STRIP_NOISE, "Strip currency symbol, commas, spaces."),
        ("price", "cast", {"to": "float"}, "Numeric once stripped."),
        ("margin_pct", "regex_replace", STRIP_PERCENT, "Strip percent sign."),
        ("margin_pct", "cast", {"to": "float"}, "Numeric once stripped."),
        ("stock", "cast", {"to": "int"}, "Whole numbers stored as floats."),
        ("stock", "fillna", MEDIAN, "Fill missing stock counts with the median."),
        ("rating", "fillna", MEDIAN, "Fill missing ratings with the median."),
        ("in_stock", "cast", {"to": "bool"}, "Mixed boolean spellings."),
        ("released", "datetime_parse", {"format": "%m/%d/%Y"}, "Month-first dates."),
    ]
    return Dataset("products", raws, cleans, kinds, steps,
                   {"column": "description", "row": INJECTION_ROW})


BUILDERS: dict[str, tuple[int, int, Callable[[random.Random, int], Dataset]]] = {
    "sales": (1, 200, _sales),
    "patients": (2, 180, _patients),
    "products": (3, 250, _products),
}
DATASET_NAMES = tuple(BUILDERS)


def build(name: str) -> Dataset:
    seed, rows, builder = BUILDERS[name]
    return builder(random.Random(seed), rows)
