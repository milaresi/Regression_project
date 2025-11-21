import streamlit as st
import pandas as pd

st.title("Roblox visits -Regression")
df = pd.read_csv("roblox_games.csv", sep=",")
st.write("Data preview:")
st.dataframe(df.head())

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
st.subheader("Scatterplot: Active players VS Favourties")
fig, ax = plt.subplots()
ax.scatter(df["Active"], df["Favourites"], s=5, alpha=0.4)
ax.ticklabel_format(style="plain", axis="y")
ax.set_xlabel("Active players")
ax.set_ylabel("Favourites")
ax.set_xscale("log")
ax.set_yscale("log")
st.pyplot(fig)

# print(df.describe())

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

st.header("Model Training")
x = df[["Active"]]
y = df["Favourites"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
pred_lr = lin_reg.predict(x_test)
r2_lr = r2_score(y_test, pred_lr)
knn_reg = KNeighborsRegressor()
knn_reg.fit(x_train, y_train)
pred_knn = knn_reg.predict(x_test)
r2_knn = r2_score(y_test, pred_knn)
svr = SVR(kernel="rbf")
svr.fit(x_train, y_train)
pred_svr = svr.predict(x_test)
r2_svr = r2_score(y_test, pred_svr)
scores = {"Linear": r2_lr, "KNN": r2_knn, "SVR": r2_svr}
models = max(scores, key=scores.get)
if models == "Linear":
    best_model = lin_reg
elif models == "KNN":
    best_model = knn_reg
else:
    best_model = svr
st.write(f"The best model is {best_model}")
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
st.header("Make prediction")
user_input = st.number_input(
    "Enter number of Active players to predict favourtie:", min_value=1
)
if st.button("Predict"):
    pred = best_model.predict([[user_input]])
    st.success(f"predicted favourties:{pred[0]:.0f}")
