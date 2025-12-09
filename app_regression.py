import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.title("Roblox Visits - Regression")

# Load data
df = pd.read_csv("roblox_games.csv", sep=",")
st.subheader("Data Preview (Raw)")
st.dataframe(df.head(10))

# Missing values and duplicates
st.subheader("Data quality info (before cleaning)")
st.write("Missing values:")
st.write(df.isnull().sum())
duplicate_count = df.duplicated().sum()
st.write(f"Number of duplicates: {duplicate_count}")
# Cleaning
st.subheader("Data Cleaning")
df.drop_duplicates(inplace=True)
num_cols = ["Active", "Visits", "Favourites", "Likes", "Dislikes", "Rating"]
for col in num_cols:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

st.subheader("Data Preview (After Cleaning)")
st.dataframe(df.head(10))
# Basic statistics
st.subheader("Basic Statistics")
st.write(df[num_cols].describe())

# Load
model = joblib.load("Regression_app.joblib")
model_info = joblib.load("model_info.joblib")

# top 10 games bar chart
st.subheader("Top 10 Games:Likes and Dislikes")
top_10 = model_info["top_10_games"]
fig, ax = plt.subplots(figsize=(8, 5))

# Bar chart: likes and dislikes
ax.bar(top_10["Name"], top_10["Likes"], label="Likes")
ax.bar(top_10["Name"], top_10["Dislikes"], bottom=top_10["Likes"], label="Dislikes")
plt.xticks(rotation=90)
ax.ticklabel_format(style="plain", axis="y")
ax.set_ylabel("Counts")
ax.set_title("Likes vs Dislikes for Top 10 Games")
ax.legend()
st.pyplot(fig)

# Boxplot
st.subheader("Boxplot: Active Players")
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(df["Active"])
ax.set_yscale("log")
ax.set_ylabel("Active players (log scale)")
st.pyplot(fig)

# Scatterplots
st.subheader("Scatterplots")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df["Active"], df["Favourites"], s=5, alpha=0.4)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Active")
axes[0].set_ylabel("Favourites")
axes[0].set_title("Active vs Favourites")

axes[1].scatter(df["Visits"], df["Active"], s=5, alpha=0.4)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Visits")
axes[1].set_ylabel("Active")
axes[1].set_title("Visits vs Active")
st.pyplot(fig)

# correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
import joblib
import pandas as pd
import streamlit as st

# load the saved info
model_info = joblib.load("model_info.joblib")
scores = model_info["R2_scores"]

# best model
best_model_name = max(scores, key=scores.get)
st.write(f"The best model is: **{best_model_name}**")

# bar chart for model
results_df = pd.DataFrame(
    {"Model": list(scores.keys()), "R2_scores": list(scores.values())}
)
st.dataframe(results_df)
st.bar_chart(results_df.set_index("Model"))

# Prediction
st.subheader("Prediction App")
game_name = st.selectbox("Select a game", df["Name"].tolist())
active = st.number_input("Active Players", min_value=0)
visits = st.number_input("Visits", min_value=0)
likes = st.number_input("Likes", min_value=0)
dislikes = st.number_input("Dislikes", min_value=0)
rating = st.number_input("Rating (0–100)", min_value=0, max_value=100)


if st.button("Predict"):
    pred = model.predict([[active, visits, likes, dislikes, rating]])
    st.success(f"Predicted Favourites for '{game_name}': {pred[0]:.0f}")
