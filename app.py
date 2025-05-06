import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Your updated metadata
data = [
    {
        "dashboard": "Single Series Channels Report",
        "tab": "Channel View",
        "filters": "Weekstarday, Channel, Territory, Platform, Select Metric",
        "view": "Weekly Trend",
        "columns": "Weeksytar day, Revenue, Minutes Streamed, Sessions, Impressions, Views"
    },
    {
        "dashboard": "Single Series Channels Report",
        "tab": "Channel View",
        "filters": "Weekstarday, Channel, Territory, Platform, Select Metric",
        "view": "Relative Weeks Trends",
        "columns": "Relative week, Revenue, Minutes Streamed, Sessions, Impressions, Views"
    },
    {
        "dashboard": "AVOD Dashboard",
        "tab": "Title Lifespan",
        "filters": "Platform, Content Type, Corporate Grade, Title, Avg Quintile Score, Consistency, Bucket, Sub Bucket",
        "view": "Title Performance By Month",
        "columns": "Date, Revenue, Average Platform Revenue"
    },
    {
        "dashboard": "AVOD Dashboard",
        "tab": "Revenue Distribution",
        "filters": "Date, Platform, Content Type",
        "view": "Platform Title Distribution",
        "columns": "Revenue Bucket, Distinct Title Count"
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Combine view + columns for similarity search
df["search_text"] = df["view"] + " " + df["columns"] + " " + df["tab"]

# Function to find best match
def find_best_match(user_query, df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df["similarity_score"] = similarities
    return df.sort_values(by="similarity_score", ascending=False).iloc[0]

# Streamlit UI
st.title("📊 Dashboard View Finder")

user_query = st.text_input("Ask for a view or describe what you're looking for:")

if user_query:
    match = find_best_match(user_query, df)
    st.subheader("🔍 Best Match")
    st.markdown(f"**Dashboard:** {match['dashboard']}")
    st.markdown(f"**Tab:** {match['tab']}")
    st.markdown(f"**View:** {match['view']}")
    st.markdown(f"**Filters:** {match['filters']}")
    st.markdown(f"**Columns:** {match['columns']}")
