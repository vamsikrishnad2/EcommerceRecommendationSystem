
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page settings
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️"
)

st.title("E-Commerce Product Recommendation System")
st.write("Find products similar to the product you select.")

# Load product data
df = pd.read_csv("products.csv")

# Combine product details
df["features"] = (
    df["category"].fillna("") + " " +
    df["brand"].fillna("") + " " +
    df["description"].fillna("")
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["features"])

# Calculate product similarity
similarity_matrix = cosine_similarity(feature_matrix)

# Select a product
selected_product = st.selectbox(
    "Choose a product:",
    df["product_name"].tolist()
)

if st.button("Recommend Products"):
    product_index = df.index[
        df["product_name"] == selected_product
    ][0]

    similarity_scores = list(
        enumerate(similarity_matrix[product_index])
    )

    # Sort by similarity, excluding the selected product
    similarity_scores = sorted(
        similarity_scores,
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = [
        (index, score)
        for index, score in similarity_scores
        if index != product_index
    ][:5]

    st.subheader("Recommended Products")

    for index, score in recommendations:
        st.write("###", df.iloc[index]["product_name"])
        st.write("Category:", df.iloc[index]["category"])
        st.write("Brand:", df.iloc[index]["brand"])
        st.write("Similarity:", round(score * 100, 2), "%")
        st.divider()
