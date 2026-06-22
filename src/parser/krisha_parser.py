import numpy as np
import pandas as pd


CURRENT_YEAR = 2026

DISTRICT_PRICE_PER_M2 = {
    "Есильский": 720_000,
    "Нура": 700_000,
    "Сарайшык": 650_000,
    "Алматы": 600_000,
    "Байконур": 520_000,
    "Сарыарка": 500_000,
}

DISTRICT_WEIGHTS = {
    "Есильский": 0.20,
    "Нура": 0.21,
    "Сарайшык": 0.17,
    "Алматы": 0.20,
    "Байконур": 0.11,
    "Сарыарка": 0.11,
}

DISTRICT_YEAR_RANGES = {
    "Есильский": (2008, CURRENT_YEAR),
    "Нура": (2016, CURRENT_YEAR),
    "Сарайшык": (2015, CURRENT_YEAR),
    "Алматы": (1980, CURRENT_YEAR),
    "Байконур": (1975, CURRENT_YEAR - 1),
    "Сарыарка": (1970, CURRENT_YEAR - 1),
}

BUILDING_TYPE_MULTIPLIER = {
    "монолитный": 1.08,
    "кирпичный": 1.03,
    "панельный": 0.92,
    "иной": 0.98,
}

COMPLEX_CLASS_MULTIPLIER = {
    "эконом": 0.88,
    "комфорт": 1.00,
    "бизнес": 1.18,
    "премиум": 1.38,
}

FURNITURE_MULTIPLIER = {
    "Нет": 0.98,
    "Частично": 1.01,
    "Есть": 1.04,
}

ROOM_AREA_RANGES = {
    1: (28, 52),
    2: (42, 78),
    3: (62, 108),
    4: (88, 142),
    5: (118, 178),
    6: (150, 220),
}


def _choice_from_weights(rng, weights):
    values = list(weights.keys())
    probabilities = np.array(list(weights.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(values, p=probabilities)


def _generate_year(rng, district):
    start_year, end_year = DISTRICT_YEAR_RANGES[district]
    if district in {"Нура", "Сарайшык", "Есильский"}:
        if rng.random() < 0.72:
            start_year = max(start_year, 2018)
    elif rng.random() < 0.34:
        end_year = min(end_year, 2005)

    return int(rng.integers(start_year, end_year + 1))


def _generate_total_floors(rng, district, year):
    if year < 1995:
        return int(rng.choice([5, 5, 5, 9, 9, 10, 12]))

    if year < 2010:
        return int(rng.choice([5, 9, 9, 10, 12, 16, 18]))

    if district in {"Нура", "Сарайшык", "Есильский"}:
        return int(rng.choice([9, 10, 12, 16, 18, 20, 21, 22, 24, 25]))

    return int(rng.choice([5, 9, 10, 12, 14, 16, 18, 20, 22]))


def _floor_multiplier(floor, total_floors):
    if floor == 1:
        return 0.95
    if floor == total_floors:
        return 0.97

    floor_ratio = floor / total_floors
    if 0.25 <= floor_ratio <= 0.75:
        return 1.02
    return 1.00


def _year_multiplier(year):
    if year >= 2024:
        return 1.08
    if year >= 2020:
        return 1.05
    if year >= 2015:
        return 1.02
    if year < 1990:
        return 0.90
    if year < 2000:
        return 0.94
    return 1.00


def _class_weights_for_district(district):
    if district in {"Есильский", "Нура"}:
        return {"эконом": 0.08, "комфорт": 0.47, "бизнес": 0.36, "премиум": 0.09}
    if district == "Сарайшык":
        return {"эконом": 0.10, "комфорт": 0.55, "бизнес": 0.30, "премиум": 0.05}
    if district == "Алматы":
        return {"эконом": 0.20, "комфорт": 0.55, "бизнес": 0.22, "премиум": 0.03}
    return {"эконом": 0.42, "комфорт": 0.48, "бизнес": 0.09, "премиум": 0.01}


def _building_type_weights_for_year(year):
    if year >= 2015:
        return {"монолитный": 0.56, "кирпичный": 0.32, "панельный": 0.07, "иной": 0.05}
    if year >= 2000:
        return {"монолитный": 0.34, "кирпичный": 0.42, "панельный": 0.18, "иной": 0.06}
    return {"монолитный": 0.12, "кирпичный": 0.34, "панельный": 0.48, "иной": 0.06}


def _estimate_market_price(row):
    base_price = row["площадь"] * row["район_базовая_цена_м2"]
    return (
        base_price
        * row["коэффициент_класса_жк"]
        * row["коэффициент_типа_дома"]
        * row["коэффициент_этажа"]
        * row["коэффициент_года"]
        * row["коэффициент_мебели"]
    )


def generate_housing_data(output_path="astana_housing.csv", num_rows=3000, seed=42):
    """Generate synthetic Astana apartment data with realistic districts and prices."""
    rng = np.random.default_rng(seed)

    districts = list(DISTRICT_WEIGHTS.keys())
    district_probabilities = np.array(list(DISTRICT_WEIGHTS.values()), dtype=float)
    district_probabilities = district_probabilities / district_probabilities.sum()

    room_values = np.array([1, 2, 3, 4, 5, 6])
    room_probabilities = np.array([0.28, 0.36, 0.24, 0.085, 0.025, 0.01])

    rows = []
    for _ in range(num_rows):
        district = rng.choice(districts, p=district_probabilities)
        rooms = int(rng.choice(room_values, p=room_probabilities))
        area_min, area_max = ROOM_AREA_RANGES[rooms]
        area = float(np.round(rng.triangular(area_min, (area_min + area_max) / 2, area_max), 1))

        year = _generate_year(rng, district)
        total_floors = _generate_total_floors(rng, district, year)
        floor = int(rng.integers(1, total_floors + 1))

        complex_class = _choice_from_weights(rng, _class_weights_for_district(district))
        building_type = _choice_from_weights(rng, _building_type_weights_for_year(year))
        furniture = rng.choice(["Есть", "Частично", "Нет"], p=[0.42, 0.37, 0.21])

        district_price = DISTRICT_PRICE_PER_M2[district]
        district_variation = rng.normal(1.0, 0.045)
        room_liquidity = {1: 1.05, 2: 1.03, 3: 1.00, 4: 0.97, 5: 0.94, 6: 0.92}[rooms]
        room_fixed_bonus = rooms * 850_000

        row = {
            "комнаты": rooms,
            "площадь": area,
            "этаж": floor,
            "этажность": total_floors,
            "район": district,
            "год_постройки": year,
            "тип_дома": building_type,
            "мебель": furniture,
            "класс_жк": complex_class,
            "район_базовая_цена_м2": district_price,
            "коэффициент_класса_жк": COMPLEX_CLASS_MULTIPLIER[complex_class],
            "коэффициент_типа_дома": BUILDING_TYPE_MULTIPLIER[building_type],
            "коэффициент_этажа": _floor_multiplier(floor, total_floors),
            "коэффициент_года": _year_multiplier(year),
            "коэффициент_мебели": FURNITURE_MULTIPLIER[furniture],
        }

        estimated_market_price = _estimate_market_price(row)
        market_noise = rng.normal(1.0, 0.035)
        price = int((estimated_market_price * district_variation * room_liquidity * market_noise) + room_fixed_bonus)

        row["цена"] = max(price, 9_500_000)
        rows.append(row)

    columns = [
        "цена",
        "комнаты",
        "площадь",
        "этаж",
        "этажность",
        "район",
        "год_постройки",
        "тип_дома",
        "мебель",
        "класс_жк",
        "район_базовая_цена_м2",
    ]
    df = pd.DataFrame(rows)[columns]
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Файл {output_path} создан с {len(df)} строками")
    return df


def clean_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_columns:
        df[column] = df[column].fillna("нет данных")

    df = df[
        (df["цена"] > 0)
        & (df["площадь"].between(20, 260))
        & (df["комнаты"].between(1, 6))
        & (df["этаж"] >= 1)
        & (df["этажность"] >= df["этаж"])
        & (df["год_постройки"].between(1960, CURRENT_YEAR))
    ]

    df["цена_за_м2"] = df["цена"] / df["площадь"]
    df = df[df["цена_за_м2"].between(250_000, 1_800_000)]

    df["возраст_дома"] = CURRENT_YEAR - df["год_постройки"]
    df["новостройка"] = (df["год_постройки"] >= 2020).astype(int)
    df["отношение_этажа_к_этажности"] = df["этаж"] / df["этажность"]
    df["первый_или_последний_этаж"] = (
        (df["этаж"] == 1) | (df["этаж"] == df["этажность"])
    ).astype(int)
    df["площадь_на_комнату"] = df["площадь"] / df["комнаты"]

    df["коэффициент_класса_жк"] = df["класс_жк"].map(COMPLEX_CLASS_MULTIPLIER).fillna(1.0)
    df["коэффициент_типа_дома"] = df["тип_дома"].map(BUILDING_TYPE_MULTIPLIER).fillna(1.0)
    df["коэффициент_этажа"] = [
        _floor_multiplier(floor, total)
        for floor, total in zip(df["этаж"], df["этажность"])
    ]
    df["коэффициент_года"] = df["год_постройки"].map(_year_multiplier)
    df["коэффициент_мебели"] = df["мебель"].map(FURNITURE_MULTIPLIER).fillna(1.0)

    one_hot_columns = [
        col
        for col in ["тип_дома", "район", "мебель", "класс_жк"]
        if col in df.columns
    ]
    if one_hot_columns:
        df = pd.get_dummies(df, columns=one_hot_columns, prefix_sep="_")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    output_file = project_root / "astana_housing.csv"
    generate_housing_data(output_file)
