import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Roblox visits -Regression")
df = pd.read_csv("roblox_games.csv", sep=",")
st.write("Data preview:")
st.dataframe(df.head(10))

st.subheader("Data cleaning")
st.write("Data types")
st.write(df.dtypes)
st.write("Missing values")
st.write(df.isnull().sum())
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()
st.write(f"{duplicate_count} duplicates has been detected and removed")

numeric_Cols = df.select_dtypes(include=["float64"]).columns
df[numeric_Cols] = df[numeric_Cols].fillna(df[numeric_Cols].median())

string_cols = df.select_dtypes(include="object").columns
df[string_cols] = df[string_cols].fillna("Unknown")
num_cols = ["Active", "Visits", "Favourites", "Likes", "Dislikes"]

for col in num_cols:
    df[col] = df[col].str.replace(",", "")
    df[col] = pd.to_numeric(df[col])

st.subheader("Data preview after cleaning")
st.write(df.dtypes)
st.dataframe(df.head())

st.subheader("unqiue values")
unique_counts = df.nunique()
st.write(unique_counts)
print(unique_counts)

# EDA
st.subheader("Basic Statistics")
st.write(df.describe())

st.subheader("Bar graph plot: Favourites of top 10 games")
ten_games = df.head(10)

import matplotlib.pyplot as plt

# bargraph
fig, ax = plt.subplots()
ax.bar(ten_games["Name"], ten_games["Favourites"])
ax.ticklabel_format(style="plain", axis="y")
plt.xticks(rotation=90)
ax.set_xlabel("Game names")
ax.set_ylabel("Favourties")
st.pyplot(fig)

# boxplot
st.subheader("Boxplot: Active players")
fig, ax = plt.subplots(figsize=(6, 4))

ax.boxplot(df["Active"])
ax.set_yscale("log")
ax.set_ylabel("Active players(log scale)")
st.pyplot(fig)

# scatterplot
st.subheader("Scatterplots")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Active vs favourites
axes[0].scatter(df["Active"], df["Favourites"], s=5, alpha=0.4)
axes[0].set_xlabel("Active players")
axes[0].set_ylabel("Favourites")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_title("Active vs Favourites")

# Visits vs active
axes[1].scatter(df["Visits"], df["Active"], s=5, alpha=0.4)
axes[1].set_xlabel("Visits")
axes[1].set_ylabel("Active players")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("Visits vs Active")

st.pyplot(fig)


# correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


# print(df.describe())

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

st.header("Model Training")
x = df[["Active", "Visits", "Likes", "Dislikes", "Rating"]]
y = df["Favourites"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)
# linear
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
pred_lr = lin_reg.predict(x_test)
r2_lr = r2_score(y_test, pred_lr)

# knn
knn_reg = KNeighborsRegressor()
knn_reg.fit(x_train, y_train)
pred_knn = knn_reg.predict(x_test)
r2_knn = r2_score(y_test, pred_knn)

# svr
svr = SVR(kernel="rbf")
svr.fit(x_train, y_train)
pred_svr = svr.predict(x_test)
r2_svr = r2_score(y_test, pred_svr)

# best model
scores = {"Linear Regression": r2_lr, "KNN Regression": r2_knn, "SVR": r2_svr}

best_model_name = max(scores, key=scores.get)
if best_model_name == "Linear Regression":
    best_model = lin_reg
elif best_model_name == "KNN Regression":
    best_model = knn_reg
else:
    best_model = svr

st.write(f"The best model is: **{best_model_name}**")

st.subheader("Model accuracy : Bar graph")
results = pd.DataFrame(
    {
        "Model": ["Linear Regression", "KNN Regression", "SVR"],
        "R2_scores": [r2_lr, r2_knn, r2_svr],
    }
)

st.dataframe(results)
st.bar_chart(results.set_index("Model"))

# input
import joblib

joblib.dump(best_model, "Regression_app.joblib")
model = joblib.load("Regression_app.joblib")
st.header("prediction app")
st.header("Prediction App")

game_name = st.text_input("Enter game name")

active_players = st.number_input("Active players", min_value=0)
visits = st.number_input("Total visits", min_value=0)
likes = st.number_input("Likes", min_value=0)
dislikes = st.number_input("Dislikes", min_value=0)
rating = st.number_input("Rating (0–100)", min_value=0, max_value=100)

if st.button("Predict"):
    pred = model.predict([[active_players, visits, likes, dislikes, rating]])
    st.success(f"Predicted favourites for {game_name}: {pred[0]:.0f}")
