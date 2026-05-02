# Cafeteria Footfall Prediction for Food Waste Reduction

This final-year project predicts cafeteria footfall to support better food preparation, inventory planning, and food waste reduction. The system uses historical meal data and contextual variables to estimate how many students are likely to visit the cafeteria for a given meal.

The final deployed application is a Gradio-based web app backed by a Poisson regression model trained on historical cafeteria data.

## Problem Statement

Institutional cafeterias often rely on manual estimation to decide how much food to prepare. When demand is overestimated, food is wasted. When demand is underestimated, service quality suffers.

This project addresses that problem by building a data-driven prediction system that estimates cafeteria footfall using historical and operational signals such as:

- day of the week
- meal type
- campus population
- bogo offer
- paneer on menu
- guest presence
- test or exam day

## Project Objective

- predict meal-wise cafeteria footfall from historical and contextual data
- reduce food wastage through better preparation planning
- study the effect of operational factors on student turnout
- compare multiple machine learning approaches for demand prediction
- deploy the selected model in an interactive application

## Why This Project Matters

The project connects machine learning with a practical sustainability problem. Instead of treating forecasting as a purely academic task, it applies prediction directly to cafeteria operations, where better estimates can improve planning and reduce avoidable waste.

The work is closely aligned with:

- `SDG 12`: Responsible Consumption and Production
- `SDG 9`: Industry, Innovation and Infrastructure

## Methodology

The project followed this pipeline:

1. historical cafeteria and meal-related data were collected and organized
2. exploratory data analysis was performed to study trends, seasonality, and anomalies
3. features were engineered from contextual variables such as meal type, day, offers, guests, and tests
4. multiple regression and forecasting approaches were explored in notebooks
5. Poisson regression was selected for deployment because the target variable is count-based footfall
6. the trained model was saved and integrated into a Gradio web application

## Model Selection

During experimentation, the project explored models such as:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor
- Neural Network models in Keras
- Poisson Regression

The final deployed model is a Poisson regression model stored in [poissonreg.pkl](/Users/anmolsrivastava/Documents/major-project/MLPR/models/poissonreg.pkl).

Poisson regression was chosen because:

- footfall is count data
- the model is suitable for non-negative count prediction
- the model is interpretable for operational decision-making

## Final Application

The deployed application is implemented in [model_demo.py](/Users/anmolsrivastava/Documents/major-project/MLPR/model_demo.py).

The app accepts:

- day of the week
- meal type
- campus population (`50` to `500`)
- bogo offer
- paneer on menu
- guest on campus
- test or exam day

The app displays:

- predicted footfall
- capacity utilization
- historical average, minimum, and maximum
- prediction vs historical comparison chart
- feature contribution chart
- weekly trend chart

## Dataset Summary

The main runtime dataset is [footfall_735.csv](/Users/anmolsrivastava/Documents/major-project/MLPR/footfall_735.csv).

Current project data characteristics:

- rows: `736`
- meal categories present: `Breakfast`, `Lunch`, `Dinner`, `Snacks`
- runtime app uses: `Breakfast`, `Lunch`, `Dinner`
- campus population range in training data: `70` to `500`

Important note:

- the deployed model should be used within the trained campus population range
- predictions outside the trained range are not reliable without retraining

## System Architecture

The system architecture consists of three layers:

- `Data Layer`: CSV datasets and notebook-based preprocessing workflows
- `Model Layer`: feature engineering, experimentation, and final Poisson regression training
- `Application Layer`: a Gradio web interface for real-time prediction and visualization

## Repository Structure

```text
MLPR/
├── model_demo.py                         # Final Gradio application
├── models/
│   ├── poissonreg.pkl                    # Final trained Poisson regression model
│   ├── footfall_pred.ipynb               # Model training notebook copy
│   └── proportion_pred.ipynb             # Proportion modeling notebook copy
├── footfall_735.csv                      # Main prediction dataset
├── merged_file.csv                       # Historical merged meal/footfall data
├── may.csv                               # Engineered dataset variant
├── may_till_aug_file.csv                 # Engineered dataset variant
├── eda.ipynb                             # Main exploratory analysis notebook
├── footfall_pred.ipynb                   # Poisson model training notebook
├── regression.ipynb                      # Model comparison notebook
├── footfall_prediction_models.ipynb      # Benchmarking notebook
├── footfall_prediction_models_mayonly.ipynb
├── menu.ipynb                            # Menu feature preparation
├── clean.ipynb                           # Merges menu and footfall features
├── reccomend.ipynb                       # Experimental recommendation-style modeling
├── NOTEBOOK_VIVA_GUIDE.md                # Viva preparation notes for notebooks
└── meow.yaml                             # Optional environment specification
```

## Notebooks Overview

Some notebooks are exploratory and some support the final model more directly.

Key notebooks:

- [eda.ipynb](/Users/anmolsrivastava/Documents/major-project/MLPR/eda.ipynb): exploratory data analysis and feature engineering
- [footfall_pred.ipynb](/Users/anmolsrivastava/Documents/major-project/MLPR/footfall_pred.ipynb): training workflow for the final Poisson model
- [regression.ipynb](/Users/anmolsrivastava/Documents/major-project/MLPR/regression.ipynb): comparison of multiple regression approaches
- [menu.ipynb](/Users/anmolsrivastava/Documents/major-project/MLPR/menu.ipynb) and [clean.ipynb](/Users/anmolsrivastava/Documents/major-project/MLPR/clean.ipynb): menu-feature processing
- [NOTEBOOK_VIVA_GUIDE.md](/Users/anmolsrivastava/Documents/major-project/MLPR/NOTEBOOK_VIVA_GUIDE.md): explanation of what each notebook does

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/LordHarsh/MLPR.git
cd MLPR
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install gradio joblib pandas statsmodels scikit-learn matplotlib
```

### 4. Launch the application

```bash
python model_demo.py
```

Open `http://127.0.0.1:7860` in your browser.

## Expected Output

For each prediction request, the application returns:

- predicted cafeteria footfall
- utilization percentage
- historical reference values for that day and meal
- visual comparison with historical data
- chart-based interpretation of feature influence

## Research Gap Addressed

- many cafeteria operations still depend on manual estimation rather than predictive analytics
- limited work focuses on meal-wise cafeteria demand using contextual academic and menu-related factors
- few student projects combine prediction, interpretability, and deployment into one usable tool

## Novelty

- applies machine learning specifically to cafeteria footfall prediction for food waste reduction
- combines operational variables such as meal type, population, offers, guests, and test days in one system
- delivers a working application instead of only offline model results

## Limitations

- some notebooks are exploratory and not production-polished
- parts of the notebook workflow depend on intermediate or legacy files
- the deployed model is reliable only within the observed training range
- richer real-world features such as weather, festival schedules, or transaction logs are not integrated yet

## Future Improvements

- retrain the model on a larger and more recent dataset
- incorporate additional features such as holidays, events, and weather
- add stronger evaluation reporting and model monitoring
- extend the system from prediction to quantity recommendation for each meal

## Team Collaboration

This project was completed collaboratively across data preparation, exploratory analysis, model development, evaluation, and deployment. The final outcome combines dataset engineering, machine learning experimentation, and application integration into one complete prediction workflow.

## Acknowledgment

This project was carried out as part of a final-year Machine Learning and Pattern Recognition project focused on solving a practical sustainability problem through applied machine learning.
