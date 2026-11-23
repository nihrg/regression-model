"""
Used-car sale price: fit on past lot sales, then score auction buy/pass rules.

The 240 sold cars are the only rows used to fit or compare models.
The 40 auction lots are held out of fitting and are used only to score decisions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Paths sit next to this script so a run does not depend on the working directory.
DATA_DIR = Path(__file__).resolve().parent
SALES_PATH = DATA_DIR / "used_car_sales.csv"
AUCTION_PATH = DATA_DIR / "auction_lots.csv"
PLOT_PATH = DATA_DIR / "test_fit_comparison.png"

FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
TARGET = "Sale_Price"
PREP_COST = 1500
P_VALUE_CUTOFF = 0.05
RANDOM_STATE = 42

# Set to "full" or "reduced" to override the training adjusted-R^2 rule below.
# None keeps the smaller model only when dropping weak features does not
# lower training adjusted R^2. The test set is never used to make this choice.
V1_OVERRIDE = None


def encode_accident(series):
    """Map the history-report flag to 1 (accident) or 0 (no accident)."""
    cleaned = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(cleaned.unique()) - {"yes", "no"})
    if unknown:
        raise ValueError(f"Unexpected Accident values: {unknown}")
    return cleaned.map({"yes": 1, "no": 0}).astype(int)


def load_sales(path):
    """Load the 240 lot sales and encode Accident before any split."""
    sales = pd.read_csv(path)
    if len(sales) != 240:
        raise ValueError(f"Expected 240 sold cars, found {len(sales)}.")
    sales = sales.copy()
    sales["Accident"] = encode_accident(sales["Accident"])
    return sales


def design_matrix(frame, features):
    """Build the OLS exog matrix: intercept first, then the same feature order."""
    if not features:
        return pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index)
    return sm.add_constant(frame[features], has_constant="add")


def decide_buy(predicted_price, auction_price, prep_cost=1500):
    """Buy when the predicted lot price is above the auction price plus prep cost.

    Getting a car onto the lot costs about $1,500. A tie is a pass: the car
    is bought only when predicted price strictly clears that all-in cost.
    """
    if auction_price + prep_cost < predicted_price:
        return "BUY"
    return "PASS"


def apply_rule(predicted_prices, auction_prices):
    """Apply decide_buy to each lot. Predictions are already fixed before this."""
    return [
        decide_buy(predicted, auction)
        for predicted, auction in zip(predicted_prices, auction_prices)
    ]


def training_fit_metrics(y_true, y_pred):
    """R^2 and dollar RMSE against sale prices the model was not fit on."""
    return r2_score(y_true, y_pred), root_mean_squared_error(y_true, y_pred)


def choose_v1(full_model, reduced_model, dropped):
    """Pick the model to hand the dealer using training adjusted R^2 only."""
    full_adj = full_model.rsquared_adj
    if V1_OVERRIDE == "full":
        return full_model, "full", "V1_OVERRIDE is 'full'."
    if V1_OVERRIDE == "reduced":
        if reduced_model is None:
            raise ValueError("V1_OVERRIDE is 'reduced', but no feature had p > 0.05.")
        return reduced_model, "reduced", "V1_OVERRIDE is 'reduced'."
    if V1_OVERRIDE is not None:
        raise ValueError("V1_OVERRIDE must be None, 'full', or 'reduced'.")

    if not dropped:
        return (
            full_model,
            "full",
            "Every training feature has p <= 0.05, so v1 is the full model.",
        )

    reduced_adj = reduced_model.rsquared_adj
    dropped_text = ", ".join(dropped)
    if reduced_adj >= full_adj:
        reason = (
            f"{dropped_text} did not earn a place (p > {P_VALUE_CUTOFF}), and "
            f"dropping {'it' if len(dropped) == 1 else 'them'} does not lower "
            f"training adjusted R^2 ({reduced_adj:.4f} vs {full_adj:.4f})."
        )
        return reduced_model, "reduced", reason

    reason = (
        f"Dropping {dropped_text} lowers training adjusted R^2 from "
        f"{full_adj:.4f} to {reduced_adj:.4f}, so v1 keeps the full model."
    )
    return full_model, "full", reason


def rule_results(name, decisions, profit):
    """Score a buy/pass rule on the 40 lots.

    Profit on a purchased car is Later_Sale_Price - Auction_Price - 1,500.
    A pass contributes nothing. A purchase loses money when that profit is negative.
    """
    bought = np.asarray(decisions) == "BUY"
    profit = np.asarray(profit, dtype=float)
    return {
        "Rule": name,
        "Cars Bought": int(bought.sum()),
        "Money-Losing Cars Bought": int((bought & (profit < 0)).sum()),
        "Total Net Profit": profit[bought].sum(),
    }


def save_test_plot(y_true, baseline_pred, v1_pred, path):
    """Actual vs predicted sale price on the test cars, baseline and v1."""
    y_true = np.asarray(y_true, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    v1_pred = np.asarray(v1_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Baseline is one price for every car, so these points form a horizontal line.
    ax.scatter(
        y_true,
        baseline_pred,
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        label="Baseline",
        zorder=2,
    )
    ax.scatter(
        y_true,
        v1_pred,
        s=28,
        c="black",
        label="v1",
        zorder=3,
    )

    lo = min(y_true.min(), baseline_pred.min(), v1_pred.min())
    hi = max(y_true.max(), baseline_pred.max(), v1_pred.max())
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="black",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        label="Perfect prediction",
        zorder=1,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Shared limits and equal aspect keep the reference line at 45 degrees.
    ax.set_aspect("equal", adjustable="box")

    dollar = FuncFormatter(lambda value, _position: f"${value:,.0f}")
    ax.xaxis.set_major_formatter(dollar)
    ax.yaxis.set_major_formatter(dollar)
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="black", length=4, labelsize=10)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    legend = ax.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color("black")

    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    sales = load_sales(SALES_PATH)

    # 70/30 of 240 cars is 168 train and 72 test. The test cars are for fit only.
    train, test = train_test_split(
        sales,
        test_size=0.30,
        train_size=0.70,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if len(train) != 168 or len(test) != 72:
        raise ValueError(f"Expected 168/72 split, got {len(train)}/{len(test)}.")

    y_train = train[TARGET]
    y_test = test[TARGET]
    print(f"Split: {len(train)} training cars, {len(test)} test cars (random_state={RANDOM_STATE}).")
    print()

    # Full model: every listed feature, training cars only.
    full_model = sm.OLS(y_train, design_matrix(train, FEATURES)).fit()
    print("Full model (training set)")
    print(full_model.summary())
    print()

    # A feature earns its place only when its training p-value is at or under 0.05.
    # The intercept is kept either way.
    dropped = [
        feature
        for feature in FEATURES
        if full_model.pvalues[feature] > P_VALUE_CUTOFF
    ]
    kept = [feature for feature in FEATURES if feature not in dropped]

    reduced_model = None
    if dropped:
        reduced_model = sm.OLS(y_train, design_matrix(train, kept)).fit()
        print(f"Reduced model, dropped {', '.join(dropped)} (training set)")
        print(reduced_model.summary())
        print()
        print("Training adjusted R^2")
        print(f"  Full model:    {full_model.rsquared_adj:.4f}")
        print(f"  Reduced model: {reduced_model.rsquared_adj:.4f}")
    else:
        print("No feature has p > 0.05. There is no smaller specification to fit.")
        print("Training adjusted R^2")
        print(f"  Full model:    {full_model.rsquared_adj:.4f}")
    print()

    v1_model, v1_name, v1_reason = choose_v1(full_model, reduced_model, dropped)
    v1_features = [name for name in v1_model.model.exog_names if name != "const"]
    print(f"v1: {v1_name} model ({', '.join(v1_features) if v1_features else 'intercept only'})")
    print(f"Why: {v1_reason}")
    print()

    # Baseline ignores the features and predicts the training-set average every time.
    baseline_price = float(y_train.mean())
    baseline_test_pred = np.full(len(test), baseline_price)
    v1_test_pred = v1_model.predict(design_matrix(test, v1_features))

    baseline_r2, baseline_rmse = training_fit_metrics(y_test, baseline_test_pred)
    v1_r2, v1_rmse = training_fit_metrics(y_test, v1_test_pred)

    print("Test-set fit (cars the model never saw)")
    print(f"  Baseline price (training mean): ${baseline_price:,.2f}")
    print(f"  Baseline  R^2: {baseline_r2:.4f}    RMSE: ${baseline_rmse:,.2f}")
    print(f"  v1        R^2: {v1_r2:.4f}    RMSE: ${v1_rmse:,.2f}")
    print()

    save_test_plot(y_test, baseline_test_pred, v1_test_pred, PLOT_PATH)
    print(f"Saved {PLOT_PATH.name}")
    print()

    # Auction lots are a decision test, not more training data.
    auction = pd.read_csv(AUCTION_PATH)
    if len(auction) != 40:
        raise ValueError(f"Expected 40 auction lots, found {len(auction)}.")
    auction = auction.copy()
    auction["Accident"] = encode_accident(auction["Accident"])

    # Realized profit uses the later retail price. The rules do not see that column.
    realized_profit = (
        auction["Later_Sale_Price"] - auction["Auction_Price"] - PREP_COST
    )
    baseline_decisions = apply_rule(
        np.full(len(auction), baseline_price),
        auction["Auction_Price"],
    )
    v1_decisions = apply_rule(
        v1_model.predict(design_matrix(auction, v1_features)),
        auction["Auction_Price"],
    )

    comparison = pd.DataFrame(
        [
            rule_results("Baseline", baseline_decisions, realized_profit),
            rule_results("v1", v1_decisions, realized_profit),
        ]
    ).set_index("Rule")
    comparison.index.name = None

    def format_dollars(amount):
        sign = "-" if amount < 0 else ""
        return f"{sign}${abs(amount):,.0f}"

    comparison["Total Net Profit"] = comparison["Total Net Profit"].map(format_dollars)

    print("Auction lots: buy if auction price + $1,500 is below the predicted sale price")
    print("Profit on a bought car = Later_Sale_Price - Auction_Price - 1,500")
    print(comparison.to_string())


if __name__ == "__main__":
    main()
