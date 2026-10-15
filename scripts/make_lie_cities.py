"""Build the learn-tier "True or false?" dataset for the lie-detection track.

Statements follow Marks & Tegmark's "cities" dataset ("The city of Paris is in
France."). Ground truth comes from construction: every city below is paired with the
one country it is in, and a false statement pairs the city with another country from
the list. The list sticks to very well-known cities whose country is not disputed, has
not changed in living memory, and is not commonly confused. Cities that share their
name with a famous city in another country (Valencia, Cordoba, Santiago, Hyderabad,
Perth, Lagos, Alexandria, Cartagena, Granada, Salvador, Christchurch) are left out.

Each city gives one true and one false statement. Cities are split between train and
test, so no city appears in both. Within a split, the false countries are a shuffle of
the true countries, so every country shows up as often in false statements as in true
ones and the country name alone says nothing about the label.

Writes:
  data/public/lie/cities/train.csv          id,statement,label   (label 1 = true)
  data/public/lie/cities/test.csv           id,statement         (no labels)
  data/private/lie/cities_test_labels.json  {id: 0|1}            (gitignored)

Run from the repo root:  python scripts/make_lie_cities.py
The output is deterministic given the seed in data/private/lie/cities_seed.txt (made on
first run). The seed is private so the public script cannot rebuild the answer key.
"""

from __future__ import annotations

import csv
import random

from make_lie_common import PRIVATE, PUBLIC, secret_seed, write_labels

TEST_FRACTION = 0.5

# Country name as it reads after "is in".
CITIES: dict[str, list[str]] = {
    "France": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Bordeaux",
               "Strasbourg", "Lille", "Nantes"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart",
                "Dresden", "Leipzig", "Bremen", "Nuremberg"],
    "Italy": ["Rome", "Milan", "Naples", "Turin", "Florence", "Venice", "Bologna",
              "Genoa", "Palermo"],
    "Spain": ["Madrid", "Barcelona", "Seville", "Bilbao", "Malaga", "Zaragoza"],
    "Portugal": ["Lisbon", "Porto", "Braga", "Coimbra"],
    "the Netherlands": ["Amsterdam", "Rotterdam", "Utrecht", "Eindhoven"],
    "Belgium": ["Brussels", "Antwerp", "Bruges", "Ghent"],
    "Switzerland": ["Zurich", "Geneva", "Basel", "Lausanne"],
    "Austria": ["Vienna", "Salzburg", "Innsbruck", "Graz"],
    "Poland": ["Warsaw", "Krakow", "Gdansk", "Wroclaw", "Poznan"],
    "Sweden": ["Stockholm", "Gothenburg", "Malmo", "Uppsala"],
    "Norway": ["Oslo", "Bergen", "Trondheim"],
    "Denmark": ["Copenhagen", "Aarhus", "Odense"],
    "Finland": ["Helsinki", "Tampere", "Turku"],
    "Greece": ["Athens", "Thessaloniki", "Patras"],
    "Hungary": ["Budapest", "Debrecen", "Szeged"],
    "Ireland": ["Dublin", "Cork", "Galway", "Limerick"],
    "the United Kingdom": ["London", "Manchester", "Liverpool", "Edinburgh", "Glasgow",
                           "Leeds", "Bristol", "Cardiff"],
    "Russia": ["Moscow", "Novosibirsk", "Yekaterinburg", "Vladivostok", "Kazan"],
    "Turkey": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa"],
    "Egypt": ["Cairo", "Giza", "Luxor", "Aswan"],
    "Morocco": ["Casablanca", "Marrakesh", "Rabat", "Fez", "Tangier"],
    "Nigeria": ["Abuja", "Kano", "Ibadan"],
    "Kenya": ["Nairobi", "Mombasa", "Kisumu"],
    "South Africa": ["Johannesburg", "Cape Town", "Durban", "Pretoria"],
    "Iran": ["Tehran", "Isfahan", "Shiraz", "Tabriz", "Mashhad"],
    "Saudi Arabia": ["Riyadh", "Jeddah", "Mecca", "Medina"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Jaipur",
              "Ahmedabad", "Pune", "Lucknow"],
    "Pakistan": ["Karachi", "Lahore", "Islamabad", "Peshawar", "Rawalpindi", "Quetta"],
    "China": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Wuhan",
              "Nanjing", "Hangzhou", "Chongqing", "Tianjin"],
    "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya", "Sapporo", "Hiroshima",
              "Fukuoka", "Kobe"],
    "South Korea": ["Seoul", "Busan", "Incheon", "Daegu"],
    "Thailand": ["Bangkok", "Chiang Mai", "Phuket", "Pattaya"],
    "Vietnam": ["Hanoi", "Ho Chi Minh City", "Da Nang", "Haiphong"],
    "Indonesia": ["Jakarta", "Surabaya", "Bandung", "Medan"],
    "the Philippines": ["Manila", "Cebu", "Davao"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Canberra", "Hobart"],
    "New Zealand": ["Auckland", "Wellington"],
    "Canada": ["Toronto", "Montreal", "Vancouver", "Ottawa", "Calgary", "Edmonton",
               "Winnipeg", "Quebec City"],
    "the United States": ["Chicago", "Houston", "Los Angeles", "San Francisco",
                          "Seattle", "Boston", "Philadelphia", "Miami", "Atlanta",
                          "Denver", "Detroit", "New Orleans", "Las Vegas"],
    "Mexico": ["Mexico City", "Guadalajara", "Monterrey", "Tijuana", "Cancun", "Puebla"],
    "Brazil": ["Rio de Janeiro", "Sao Paulo", "Brasilia", "Recife", "Fortaleza",
               "Manaus", "Belo Horizonte", "Porto Alegre", "Curitiba"],
    "Argentina": ["Buenos Aires", "Rosario", "Mendoza"],
    "Chile": ["Valparaiso", "Antofagasta"],
    "Peru": ["Lima", "Cusco", "Arequipa"],
    "Colombia": ["Bogota", "Medellin", "Cali", "Barranquilla"],
}

OUT = PUBLIC / "cities"


def statement(city: str, country: str) -> str:
    return f"The city of {city} is in {country}."


def split_cities(rng: random.Random) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split each country's cities between train and test, so both splits cover every
    country with more than one city."""
    train, test = [], []
    for country, cities in CITIES.items():
        cities = sorted(cities)
        rng.shuffle(cities)
        n_test = round(len(cities) * TEST_FRACTION)
        if len(cities) > 1:
            n_test = min(max(n_test, 1), len(cities) - 1)
        test += [(c, country) for c in cities[:n_test]]
        train += [(c, country) for c in cities[n_test:]]
    return train, test


def false_countries(pairs: list[tuple[str, str]], rng: random.Random) -> list[str]:
    """A shuffle of the true countries with no city keeping its own country. Retrying
    the shuffle is fast here because no country holds more than a few percent of
    cities."""
    countries = [c for _, c in pairs]
    for _ in range(10000):
        perm = countries[:]
        rng.shuffle(perm)
        if all(p != c for p, c in zip(perm, countries)):
            return perm
    raise RuntimeError("could not derange the countries")


def build_split(pairs: list[tuple[str, str]], prefix: str,
                rng: random.Random) -> list[dict]:
    rows = []
    for (city, country), wrong in zip(pairs, false_countries(pairs, rng)):
        rows.append({"city": city, "statement": statement(city, country), "label": 1})
        rows.append({"city": city, "statement": statement(city, wrong), "label": 0})
    rng.shuffle(rows)
    for n, row in enumerate(rows, 1):
        row["id"] = f"{prefix}-{n:04d}"
    return rows


def check_cities() -> None:
    seen: dict[str, str] = {}
    for country, cities in CITIES.items():
        for city in cities:
            if city in seen:
                raise ValueError(f"{city} listed under {seen[city]} and {country}")
            seen[city] = country


def main() -> None:
    check_cities()
    rng = random.Random(secret_seed("cities"))
    train_pairs, test_pairs = split_cities(rng)
    train = build_split(train_pairs, "cities-train", rng)
    test = build_split(test_pairs, "cities-test", rng)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "statement", "label"])
        for r in train:
            w.writerow([r["id"], r["statement"], r["label"]])
    with open(OUT / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "statement"])
        for r in test:
            w.writerow([r["id"], r["statement"]])
    labels = {r["id"]: r["label"] for r in test}
    write_labels(PRIVATE / "cities_test_labels.json", labels)

    print(f"cities: {len(train_pairs)} train cities, {len(test_pairs)} test cities")
    print(f"train: {len(train)} statements ({sum(r['label'] for r in train)} true)")
    print(f"test:  {len(test)} statements ({sum(labels.values())} true)")


if __name__ == "__main__":
    main()
