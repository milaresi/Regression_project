import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib

# Load data
df = pd.read_csv("roblox_games.csv", sep=",")
df.drop_duplicates(inplace=True)

num_cols = ["Active", "Visits", "Favourites", "Likes", "Dislikes", "Rating"]
for col in num_cols:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)
X = df[["Active", "Visits", "Likes", "Dislikes", "Rating"]]
y = df["Favourites"]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
r2_lr = r2_score(y_test, lin_reg.predict(x_test))

# knn_reg = KNeighborsRegressor()
# knn_reg.fit(x_train, y_train)
# r2_knn = r2_score(y_test, knn_reg.predict(x_test))

# Hyperparameter tuning
best_knn_score = -1
best_knn_model = None

for k in [3, 5, 7]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    score = r2_score(y_test, knn.predict(x_test))

    if score > best_knn_score:
        best_knn_score = score
        best_knn_model = knn

r2_knn = best_knn_score
knn_reg = best_knn_model

svr = SVR(kernel="rbf")
svr.fit(x_train, y_train)
r2_svr = r2_score(y_test, svr.predict(x_test))

# Best model
scores = {"Linear Regression": r2_lr, "KNN Regression": r2_knn, "SVR": r2_svr}
best_model_name = max(scores, key=scores.get)
if best_model_name == "Linear Regression":
    best_model = lin_reg
elif best_model_name == "KNN Regression":
    best_model = knn_reg
else:
    best_model = svr

# Save model for Streamlit
joblib.dump(best_model, "Regression_app.joblib")

model_info = {
    "R2_scores": scores,
    "top_10_games": df.head(10)[["Name", "Favourites", "Likes", "Dislikes"]].to_dict(),
    "numeric_data": df[num_cols].describe().to_dict(),
}
joblib.dump(model_info, "model_info.joblib")
print(f"Training complete!")
