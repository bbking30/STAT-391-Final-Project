import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ----------------------------
# Load cleaned dataset
# ----------------------------
df = pd.read_csv("data/weather_cleaned.csv")


# ----------------------------
# Create outcome variable
# ----------------------------
df["large_error"] = (df["abs_error"] > 5).astype(int)


# ----------------------------
# Select predictors
# ----------------------------
df_model = df[
    [
        "large_error",
        "forecast_hours_before",
        "forecast_temp",
        "forecast_outlook",
        "high_or_low",
        "state"
    ]
]


# ----------------------------
# Encode categorical variables
# ----------------------------
df_model = pd.get_dummies(
    df_model,
    columns=["forecast_outlook", "high_or_low", "state"],
    drop_first=True
)


# ----------------------------
# Split predictors and outcome
# ----------------------------
X = df_model.drop("large_error", axis=1)
y = df_model["large_error"]


# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ----------------------------
# Logistic regression model
# ----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# ----------------------------
# Predictions
# ----------------------------
y_pred = model.predict(X_test)


# ----------------------------
# Model evaluation
# ----------------------------
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))


# ----------------------------
# Coefficient importance
# ----------------------------
coefficients = pd.DataFrame({
    "Predictor": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print("\nTop Predictors")
print(coefficients.head(10))


# ----------------------------
# Statsmodels version (p-values)
# ----------------------------
X_sm = sm.add_constant(X)

logit_model = sm.Logit(y, X_sm).fit()

print(logit_model.summary())
