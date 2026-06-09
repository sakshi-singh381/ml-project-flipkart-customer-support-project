import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Flipkart Support Dashboard",
    layout="wide"
)

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* Headings */
h1, h2, h3 {
    color: #00ffcc;
}

/* Metrics cards */
[data-testid="metric-container"] {
    background-color: #1a1f2b;
    border: 1px solid #2a2f3a;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,255,204,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Buttons */
.stDownloadButton button {
    background-color: #00ffcc;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/Customer_support_data (1).csv")
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# ---------------- SENTIMENT ANALYSIS ----------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['customer_remarks'].apply(get_sentiment)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🔍 Filters")

category_filter = st.sidebar.multiselect(
    "Select Category",
    df['category'].unique(),
    default=df['category'].unique()
)

city_filter = st.sidebar.multiselect(
    "Select City",
    df['customer_city'].unique(),
    default=df['customer_city'].unique()
)

filtered_df = df[
    (df['category'].isin(category_filter)) &
    (df['customer_city'].isin(city_filter))
]

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>📊 Flipkart Customer Support Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>LabMentix Internship Project</h4>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- KPI METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🔥 Total Tickets", len(filtered_df))
col2.metric("⭐ Avg CSAT", round(filtered_df['csat_score'].mean(), 2))
col3.metric("⏱ Avg Handling Time", round(filtered_df['connected_handling_time'].mean(), 2))
col4.metric("🌍 Unique Cities", filtered_df['customer_city'].nunique())

st.markdown("---")

# ---------------- CATEGORY CHART ----------------
st.subheader("📊 Top Complaint Categories")

category_counts = (
    filtered_df['category']
    .value_counts()
    .rename_axis('category')
    .reset_index(name='count')
)

fig1 = px.bar(
    category_counts,
    x='category',
    y='count',
    color='category',
    text='count'
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- CSAT DISTRIBUTION ----------------
st.subheader("⭐ CSAT Distribution")

fig2 = px.histogram(
    filtered_df,
    x='csat_score',
    nbins=10
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- SENTIMENT CHART ----------------
st.subheader("💬 Sentiment Distribution")

sent_count = (
    filtered_df['sentiment']
    .value_counts()
    .rename_axis('sentiment')
    .reset_index(name='count')
)

fig3 = px.pie(
    sent_count,
    names='sentiment',
    values='count'
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- DOWNLOAD ----------------
st.download_button(
    "⬇ Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_data.csv",
    "text/csv"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(" Built with Streamlit | Flipkart Support Analytics Dashboard")