"""Retention Lens: Flask backend for the Churn_Modelling dataset.

This version uses only Python's standard library for the analysis (csv, statistics),
so it needs nothing but Flask. No pandas or numpy required.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import csv
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import correlation, mean

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"

app = Flask(__name__, static_folder="static", static_url_path="")

# --------------------------------------------------------------------------
# Load the data (done once, when the server starts)
# --------------------------------------------------------------------------
INT_COLUMNS = ["RowNumber", "CustomerId", "CreditScore", "Age", "Tenure",
               "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]
FLOAT_COLUMNS = ["Balance", "EstimatedSalary"]


def load_rows():
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for col in INT_COLUMNS:
                row[col] = int(raw[col])
            for col in FLOAT_COLUMNS:
                row[col] = float(raw[col])
            rows.append(row)
    return rows


ROWS = load_rows()
TOTAL = len(ROWS)
OVERALL_RATE = sum(r["Exited"] for r in ROWS) / TOTAL * 100


# --------------------------------------------------------------------------
# Grouping helpers
# --------------------------------------------------------------------------
def age_group(r):
    a = r["Age"]
    return "18–29" if a <= 29 else "30–39" if a <= 39 else "40–49" if a <= 49 else "50–59" if a <= 59 else "60+"


def credit_band(r):
    s = r["CreditScore"]
    if s <= 579:
        return "Poor (<580)"
    if s <= 669:
        return "Fair (580–669)"
    if s <= 739:
        return "Good (670–739)"
    if s <= 799:
        return "Very good (740–799)"
    return "Exceptional (800+)"


def balance_band(r):
    b = r["Balance"]
    if b <= 0:
        return "No balance"
    if b <= 50_000:
        return "1–50k"
    if b <= 100_000:
        return "50–100k"
    if b <= 150_000:
        return "100–150k"
    if b <= 200_000:
        return "150–200k"
    return "200k+"


AGE_ORDER = ["18–29", "30–39", "40–49", "50–59", "60+"]
CREDIT_ORDER = ["Poor (<580)", "Fair (580–669)", "Good (670–739)",
                "Very good (740–799)", "Exceptional (800+)"]
BALANCE_ORDER = ["No balance", "1–50k", "50–100k", "100–150k", "150–200k", "200k+"]

# dimension key -> (label shown on the page, function giving the group, fixed order or None)
DIMENSIONS = {
    "geography": ("Country", lambda r: r["Geography"], None),
    "gender": ("Gender", lambda r: r["Gender"], None),
    "age": ("Age group", age_group, AGE_ORDER),
    "products": ("Products held", lambda r: r["NumOfProducts"], None),
    "active": ("Membership", lambda r: "Active" if r["IsActiveMember"] else "Inactive", None),
    "card": ("Credit card", lambda r: "Has card" if r["HasCrCard"] else "No card", None),
    "credit": ("Credit score", credit_band, CREDIT_ORDER),
    "tenure": ("Years with bank", lambda r: r["Tenure"], None),
    "balance": ("Balance", balance_band, BALANCE_ORDER),
}

# Histogram settings: key -> (column, bin edges, bin labels). Bins are (low, high].
DISTRIBUTIONS = {
    "age": (
        "Age",
        list(range(14, 95, 5)),
        [f"{a + 1}–{a + 5}" for a in range(14, 94, 5)],
    ),
    "credit": (
        "CreditScore",
        [349] + list(range(399, 800, 50)) + [850],
        ["350–399", "400–449", "450–499", "500–549", "550–599", "600–649",
         "650–699", "700–749", "750–799", "800–850"],
    ),
    "balance": (
        "Balance",
        [-1, 0] + list(range(25_000, 250_001, 25_000)) + [260_000],
        ["0", "1–25k", "25–50k", "50–75k", "75–100k", "100–125k", "125–150k",
         "150–175k", "175–200k", "200–225k", "225–250k", "250k+"],
    ),
}


def round1(x):
    return round(x + 0.0, 1)


def group_stats(keyfunc, order=None):
    """Customers, leavers and churn rate for each group."""
    counts = defaultdict(lambda: [0, 0])  # group -> [customers, churned]
    for r in ROWS:
        c = counts[keyfunc(r)]
        c[0] += 1
        c[1] += r["Exited"]
    keys = [k for k in order if k in counts] if order else sorted(counts)
    return [
        {
            "label": str(k),
            "customers": counts[k][0],
            "churned": counts[k][1],
            "rate": round1(counts[k][1] / counts[k][0] * 100),
        }
        for k in keys
    ]


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/overview")
def overview():
    metrics = [
        ("Age", "Average age", "decimal"),
        ("CreditScore", "Average credit score", "whole"),
        ("Balance", "Average balance", "money"),
        ("EstimatedSalary", "Average salary", "money"),
        ("Tenure", "Years with the bank", "decimal"),
        ("NumOfProducts", "Products held", "decimal"),
    ]
    stayed = [r for r in ROWS if r["Exited"] == 0]
    left = [r for r in ROWS if r["Exited"] == 1]
    profile = [
        {
            "label": label,
            "kind": kind,
            "stayed": round(mean(r[col] for r in stayed), 2),
            "left": round(mean(r[col] for r in left), 2),
        }
        for col, label, kind in metrics
    ]
    return jsonify(
        total=TOTAL,
        churned=len(left),
        rate=round1(OVERALL_RATE),
        avg_age=round1(mean(r["Age"] for r in ROWS)),
        avg_credit=round(mean(r["CreditScore"] for r in ROWS)),
        avg_balance=round(mean(r["Balance"] for r in ROWS)),
        active_share=round1(mean(r["IsActiveMember"] for r in ROWS) * 100),
        profile=profile,
    )


@app.get("/api/churn-by")
def churn_by():
    key = request.args.get("dim", "geography")
    if key not in DIMENSIONS:
        return jsonify(error="Unknown dimension"), 400
    label, keyfunc, order = DIMENSIONS[key]
    return jsonify(
        dimension=label,
        overall=round1(OVERALL_RATE),
        groups=group_stats(keyfunc, order),
    )


@app.get("/api/hotspots")
def hotspots():
    """The single highest-churn group in each dimension (groups of 50+ customers)."""
    found = []
    for label, keyfunc, order in DIMENSIONS.values():
        groups = [g for g in group_stats(keyfunc, order) if g["customers"] >= 50]
        if not groups:
            continue
        top = max(groups, key=lambda g: g["rate"])
        found.append({
            "dimension": label,
            "group": top["label"],
            "rate": top["rate"],
            "customers": top["customers"],
            "lift": round1(top["rate"] / OVERALL_RATE),
        })
    found.sort(key=lambda g: g["rate"], reverse=True)
    return jsonify(found[:6])


@app.get("/api/distribution")
def distribution():
    key = request.args.get("field", "age")
    if key not in DISTRIBUTIONS:
        return jsonify(error="Unknown field"), 400
    column, edges, labels = DISTRIBUTIONS[key]
    stayed = [0] * len(labels)
    left = [0] * len(labels)
    for r in ROWS:
        i = bisect_left(edges, r[column]) - 1  # which (low, high] bin the value falls in
        if 0 <= i < len(labels):
            (left if r["Exited"] else stayed)[i] += 1
    return jsonify(labels=labels, stayed=stayed, left=left)


@app.get("/api/correlation")
def correlation_matrix():
    columns = {
        "CreditScore": "Credit score",
        "Age": "Age",
        "Tenure": "Tenure",
        "Balance": "Balance",
        "NumOfProducts": "Products",
        "HasCrCard": "Has card",
        "IsActiveMember": "Active",
        "EstimatedSalary": "Salary",
        "Exited": "Left bank",
    }
    series = {col: [r[col] for r in ROWS] for col in columns}
    names = list(columns)
    matrix = [
        [1.0 if a == b else round(correlation(series[a], series[b]) + 0.0, 2) for b in names]
        for a in names
    ]
    return jsonify(labels=list(columns.values()), matrix=matrix)


@app.get("/api/segment")
def segment():
    """Churn rate for any combination of filters."""
    args = request.args
    subset = ROWS

    if args.get("geography") in ("France", "Germany", "Spain"):
        subset = [r for r in subset if r["Geography"] == args["geography"]]
    if args.get("gender") in ("Female", "Male"):
        subset = [r for r in subset if r["Gender"] == args["gender"]]
    age_min = args.get("age_min", type=int)
    age_max = args.get("age_max", type=int)
    if age_min is not None:
        subset = [r for r in subset if r["Age"] >= age_min]
    if age_max is not None:
        subset = [r for r in subset if r["Age"] <= age_max]
    products = args.get("products", type=int)
    if products is not None:
        subset = [r for r in subset if r["NumOfProducts"] == products]
    active = args.get("active", type=int)
    if active in (0, 1):
        subset = [r for r in subset if r["IsActiveMember"] == active]
    card = args.get("card", type=int)
    if card in (0, 1):
        subset = [r for r in subset if r["HasCrCard"] == card]

    if not subset:
        return jsonify(customers=0, churned=0, rate=0, overall=round1(OVERALL_RATE),
                       avg_balance=0, avg_credit=0)
    churned = sum(r["Exited"] for r in subset)
    return jsonify(
        customers=len(subset),
        churned=churned,
        rate=round1(churned / len(subset) * 100),
        overall=round1(OVERALL_RATE),
        avg_balance=round(mean(r["Balance"] for r in subset)),
        avg_credit=round(mean(r["CreditScore"] for r in subset)),
    )


TABLE_COLUMNS = ["CustomerId", "Surname", "Geography", "Gender", "Age", "Tenure",
                 "CreditScore", "Balance", "NumOfProducts", "IsActiveMember", "Exited"]


@app.get("/api/customers")
def customers():
    args = request.args
    per_page = 12
    page = max(args.get("page", 1, type=int), 1)
    query = args.get("q", "").strip().lower()
    status = args.get("status", "")
    sort = args.get("sort", "CustomerId")
    if sort not in TABLE_COLUMNS:
        sort = "CustomerId"
    descending = args.get("order", "asc") == "desc"

    view = ROWS
    if query:
        view = [r for r in view if query in r["Surname"].lower()]
    if status == "left":
        view = [r for r in view if r["Exited"] == 1]
    elif status == "stayed":
        view = [r for r in view if r["Exited"] == 0]
    view = sorted(view, key=lambda r: r[sort], reverse=descending)

    total = len(view)
    pages = max(-(-total // per_page), 1)
    page = min(page, pages)
    chunk = view[(page - 1) * per_page: page * per_page]
    return jsonify(
        rows=[{col: r[col] for col in TABLE_COLUMNS} for r in chunk],
        total=total,
        page=page,
        pages=pages,
        per_page=per_page,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
