import gradio as gr
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load model and historical data
model = joblib.load("models/poissonreg.pkl")
df = pd.read_csv("footfall_735.csv")
df = df[df["meal_type"].isin(["Breakfast", "Lunch", "Dinner"])]


def preprocess(day, meal_type):
    return (
        int(day == "Saturday"),
        int(day == "Sunday"),
        int(meal_type == "Breakfast"),
        int(meal_type == "Lunch"),
        int(meal_type == "Dinner"),
    )


def get_prediction(bogo, paneer, day, guest, test, max_possible_footfall, meal_type):
    day_Saturday, day_Sunday, mt_B, mt_L, mt_D = preprocess(day, meal_type)
    X = pd.DataFrame({
        "bogo": [bogo], "paneer": [paneer],
        "day_Saturday": [day_Saturday], "day_Sunday": [day_Sunday],
        "guest": [guest], "test": [test],
        "max_possible_footfall": [max_possible_footfall],
        "meal_type_Breakfast": [mt_B],
        "meal_type_Lunch": [mt_L],
        "meal_type_Dinner": [mt_D],
    }).astype({
        "bogo": "bool", "paneer": "bool",
        "day_Saturday": "bool", "day_Sunday": "bool",
        "guest": "bool", "test": "bool",
        "max_possible_footfall": "int64",
        "meal_type_Breakfast": "bool",
        "meal_type_Lunch": "bool",
        "meal_type_Dinner": "bool",
    })
    return int(model.predict(X)[0])


def make_comparison_chart(day, meal_type, prediction):
    subset = df[(df["meal_type"] == meal_type) & (df["day"] == day)]
    hist_mean = subset["footfall"].mean() if len(subset) else 0
    hist_min  = subset["footfall"].min()  if len(subset) else 0
    hist_max  = subset["footfall"].max()  if len(subset) else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Historical Avg", "Prediction"], [hist_mean, prediction],
                  color=["#3498db", "#e67e22"], width=0.45, edgecolor="white")

    if len(subset):
        ax.errorbar(0, hist_mean,
                    yerr=[[hist_mean - hist_min], [hist_max - hist_mean]],
                    fmt="none", color="#2c3e50", capsize=10, linewidth=2)

    for bar, val in zip(bars, [hist_mean, prediction]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{val:.0f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    top = max(hist_max, prediction) * 1.25 or 100
    ax.set_ylim(0, top)
    ax.set_ylabel("Footfall", fontsize=11)
    ax.set_title(f"{meal_type} — {day}\nPrediction vs Historical (n={len(subset)})", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def make_feature_chart(bogo, paneer, day, guest, test, max_possible_footfall, meal_type):
    coeffs = model.params
    day_Saturday, day_Sunday, mt_B, mt_L, mt_D = preprocess(day, meal_type)

    impacts = {
        "Campus Pop.":   max_possible_footfall * coeffs.get("max_possible_footfall", 0),
        "Bogo Offer":    int(bogo)   * coeffs.get("bogo[T.True]", 0),
        "Paneer":        int(paneer) * coeffs.get("paneer[T.True]", 0),
        "Saturday":      day_Saturday * coeffs.get("day_Saturday[T.True]", 0),
        "Sunday":        day_Sunday   * coeffs.get("day_Sunday[T.True]", 0),
        "Guest":         int(guest) * coeffs.get("guest[T.True]", 0),
        "Test Day":      int(test)  * coeffs.get("test[T.True]", 0),
        "Breakfast":     mt_B * coeffs.get("meal_type_Breakfast[T.True]", 0),
        "Lunch":         mt_L * coeffs.get("meal_type_Lunch[T.True]", 0),
        "Dinner":        mt_D * coeffs.get("meal_type_Dinner[T.True]", 0),
    }
    # Keep only active (non-zero) features
    impacts = {k: v for k, v in impacts.items() if abs(v) > 1e-6}
    # Sort by absolute value
    impacts = dict(sorted(impacts.items(), key=lambda x: abs(x[1])))

    fig, ax = plt.subplots(figsize=(6, max(3, len(impacts) * 0.55)))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in impacts.values()]
    ax.barh(list(impacts.keys()), list(impacts.values()), color=colors, edgecolor="white")
    ax.axvline(0, color="#2c3e50", linewidth=1)
    ax.set_xlabel("Log-scale Contribution", fontsize=10)
    ax.set_title("Feature Contributions to Prediction", fontsize=11)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def make_utilization_chart(prediction, max_possible_footfall):
    pct = min(prediction / max_possible_footfall, 1.0) * 100
    color = "#2ecc71" if pct < 60 else "#f39c12" if pct < 85 else "#e74c3c"

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(
        [pct, 100 - pct],
        startangle=90,
        colors=[color, "#ecf0f1"],
        wedgeprops=dict(width=0.42, edgecolor="white"),
    )
    ax.text(0, 0, f"{pct:.0f}%", ha="center", va="center",
            fontsize=24, fontweight="bold", color="#2c3e50")
    ax.set_title("Capacity\nUtilization", fontsize=11)
    plt.tight_layout()
    return fig


def make_meal_trend_chart(meal_type):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    subset = df[df["meal_type"] == meal_type]
    means = subset.groupby("day")["footfall"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(means.index, means.values, color="#9b59b6", edgecolor="white")
    ax.set_ylabel("Avg Footfall", fontsize=10)
    ax.set_title(f"Avg {meal_type} Footfall by Day of Week", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def predict(bogo, paneer, day, guest, test, max_possible_footfall, meal_type):
    if not day or not meal_type:
        return "*Select a day and meal type, then click Predict.*", None, None, None, None

    prediction = get_prediction(bogo, paneer, day, guest, test, max_possible_footfall, meal_type)

    subset = df[(df["meal_type"] == meal_type) & (df["day"] == day)]
    hist_mean = f"{subset['footfall'].mean():.1f}" if len(subset) else "N/A"
    hist_min  = f"{int(subset['footfall'].min())}"  if len(subset) else "N/A"
    hist_max  = f"{int(subset['footfall'].max())}"  if len(subset) else "N/A"
    utilization = round(prediction / max_possible_footfall * 100, 1)

    label = "Low" if utilization < 60 else "Moderate" if utilization < 85 else "High"

    summary = f"""
## Predicted Footfall: **{prediction}**

| Metric | Value |
|---|---|
| Capacity Utilization | **{utilization}%** ({label}) |
| Historical Avg ({meal_type} / {day}) | {hist_mean} |
| Historical Min | {hist_min} |
| Historical Max | {hist_max} |
| Campus Population | {max_possible_footfall} |
"""

    fig_util   = make_utilization_chart(prediction, max_possible_footfall)
    fig_comp   = make_comparison_chart(day, meal_type, prediction)
    fig_feat   = make_feature_chart(bogo, paneer, day, guest, test, max_possible_footfall, meal_type)
    fig_trend  = make_meal_trend_chart(meal_type)

    return summary, fig_util, fig_comp, fig_feat, fig_trend


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Cafeteria Footfall Predictor") as interface:

    gr.Markdown("# Cafeteria Footfall Predictor")
    gr.Markdown("Forecast dining-hall attendance using a Poisson regression model trained on historical data.")

    with gr.Row():
        # ── Left panel: inputs ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Inputs")
            day       = gr.Radio(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                                 label="Day of the Week")
            meal_type = gr.Radio(["Breakfast","Lunch","Dinner"], label="Meal Type")
            max_ff    = gr.Slider(50, 500, value=335, step=5, label="Campus Population")

            gr.Markdown("**Special Conditions**")
            with gr.Row():
                bogo   = gr.Checkbox(label="Bogo Offer")
                paneer = gr.Checkbox(label="Paneer on Menu")
            with gr.Row():
                guest  = gr.Checkbox(label="Guest on Campus")
                test   = gr.Checkbox(label="Test / Exam Day")

            predict_btn = gr.Button("Predict", variant="primary", size="lg")

        # ── Right panel: outputs ───────────────────────────────────────────
        with gr.Column(scale=2):
            summary_md = gr.Markdown("*Select inputs and click **Predict**.*")

            with gr.Row():
                fig_util  = gr.Plot(label="Capacity Utilization")
                fig_comp  = gr.Plot(label="Prediction vs Historical")

            with gr.Row():
                fig_feat  = gr.Plot(label="Feature Contributions")
                fig_trend = gr.Plot(label="Weekly Trend")

    predict_btn.click(
        fn=predict,
        inputs=[bogo, paneer, day, guest, test, max_ff, meal_type],
        outputs=[summary_md, fig_util, fig_comp, fig_feat, fig_trend],
    )

interface.launch(share=True, theme=gr.themes.Soft())
