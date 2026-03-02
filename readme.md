# Machine Learning & Pattern Recognition Project

This project seeks to optimize the inventory management system and potentially reduce food wastage at Plaksha University by predicting cafeteria footfall using a Poisson regression model trained on historical meal data.

Featured in Plaksha yearly Magazine.

## The Team
- Alli Ajagbe
- Divith Narendra
- Soham Petkar

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/LordHarsh/MLPR.git
cd MLPR
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install gradio joblib pandas statsmodels scikit-learn matplotlib
```

### 4. Launch the web app
```bash
python model_demo.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## What the App Does

Given the following inputs:

| Input | Description |
|---|---|
| Day of the Week | Monday – Sunday |
| Meal Type | Breakfast, Lunch, or Dinner |
| Campus Population | Number of people on campus (50–500) |
| Bogo Offer | Whether a buy-one-get-one deal is active |
| Paneer on Menu | Whether paneer is being served |
| Guest on Campus | Whether external guests are present |
| Test / Exam Day | Whether it is an exam day |

The app predicts how many students will visit the cafeteria and displays:

- **Predicted footfall** with capacity utilization %
- **Comparison chart** — prediction vs historical average (with min/max range)
- **Feature contributions** — which inputs push the prediction up or down
- **Weekly trend** — average footfall by day for the selected meal type

---

## Project Structure

```
MLPR/
├── model_demo.py                  # Gradio web app (entry point)
├── models/
│   └── poissonreg.pkl             # Trained Poisson regression model
├── footfall_735.csv               # Historical dataset (735 records)
├── footfall_pred.ipynb            # Model training notebook
├── eda.ipynb                      # Exploratory data analysis
└── meow.yaml                      # Full conda environment (optional)
```
