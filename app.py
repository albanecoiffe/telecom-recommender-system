import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Telecom Offer Recommender", layout="wide")

st.title("📡 Telecom Offer Recommender System")

st.markdown("""
This Streamlit app presents:
- An overview of the telecom dataset
- The business objective of the recommender system
- A summary of the collaborative filtering model used
- Evaluation results comparing predictions with and without bootstrapping
""")

# Load preprocessed data
@st.cache_data
def load_data():
    s3_link='https://s3.amazonaws.com/projex.dezyre.com/recommender-system-for-telecom-products/materials/Telecom_data.csv'
    df = pd.read_csv("https://www.dropbox.com/scl/fi/ck3cjqomhsk98u92etl58/processed_telecom_offer_data.csv?rlkey=m32cp8ew5ecewfegew5dkx6i0&st=17ks8qko&dl=1")
    df_boot = pd.read_csv("offer_recommendation_bootstrap.csv")
    df_noboot = pd.read_csv("offer_recommendation_without_bootstap.csv")
    return df, df_boot, df_noboot

df, df_boot, df_noboot = load_data()

st.header("📁 Overview of the Telecom Dataset")
st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])
st.dataframe(df.head())

st.header("🎯 Business Objective")
st.markdown("""
The goal is to recommend the most relevant telecom offers to customers in order to:
- Increase customer satisfaction and engagement
- Reduce churn rate
- Maximize revenue by providing personalized promotions
""")

st.header("🧠 Model Overview")
st.markdown("""
We use a **collaborative filtering** approach to generate recommendations based on user similarity.

Collaborative filtering uses historical user-item interactions to recommend items (here, telecom offers).
There are two main types:
- **User-based filtering**: finds customers with similar behaviors and preferences
- **Item-based filtering**: finds items (offers) that are often co-used or similarly rated

In our case, we use **user-based collaborative filtering** with distance-based similarity measures such as:
- Cosine similarity
- Euclidean distance
- Manhattan distance

### 🔍 How it works:
1. We extract customer features (excluding ID, month, and offer)
2. For a given customer/month, we compute distances to all other customers
3. We select the `n` most similar non-churned customers
4. From their received offers, we calculate the most frequent ones
5. We recommend the top 3 offers exceeding a given minimal threshold

This technique helps personalize recommendations based on past customer behaviors and churn history.
""")


st.header("📊 Offer Predictions Overview")
st.markdown("""
We compare **two models** in this analysis:
- 🔹 A model with **bootstrapping**, which aggregates recommendations over multiple samples to improve stability
- 🔸 A simpler model **without bootstrapping**, which provides deterministic recommendations based on the nearest neighbors
""")

st.subheader("Offer Overlap Distribution (Bootstrap vs. No-Bootstrap)")
import re

def match_distribution(df_boot, df_noboot):
    def count_common(row):
        set1 = set(re.findall(r"[A-Z]", row['offers']))
        set2 = set(re.findall(r"[A-Z]", df_noboot.loc[row.name, 'offers']))
        return len(set1 & set2)

    df_boot['n_common'] = df_boot.apply(count_common, axis=1)
    return df_boot['n_common'].value_counts().sort_index()

match_counts = match_distribution(df_boot.copy(), df_noboot)

fig2, ax2 = plt.subplots(figsize=(2.5, 2))
sns.barplot(x=match_counts.index, y=match_counts.values, ax=ax2, palette="viridis")
ax2.set_title("Offer Overlap Distribution (Bootstrap vs. No-Bootstrap)")
ax2.set_xlabel("# of Common Offers (out of 3)")
ax2.set_ylabel("Number of Customers")
st.pyplot(fig2)
st.markdown("""**Observation**:  
- Most customer predictions have **2 or 3 common offers**, confirming that the recommendations are **mostly consistent**.
- Only **a few cases (1 or 0)** show model disagreement — likely due to noise or borderline similarity scores.

This supports the idea that the bootstrap-enhanced model is robust and not introducing erratic recommendations.""")

st.subheader("Top-1 Offer Frequency by Model")

# Frequency comparison
def prepare_top1_freq(df_boot, df_noboot):
    boot = df_boot['offers'].str.extractall(r"([A-Z])").groupby(level=0).first()
    noboot = df_noboot['offers'].str.extractall(r"([A-Z])").groupby(level=0).first()
    freq_boot = boot[0].value_counts().sort_index()
    freq_noboot = noboot[0].value_counts().sort_index()
    return pd.DataFrame({"Bootstrap": freq_boot, "No Bootstrap": freq_noboot}).fillna(0)

freq_df = prepare_top1_freq(df_boot, df_noboot)
#st.dataframe(freq_df)

fig, ax = plt.subplots(figsize=(3, 2))
freq_df.plot(kind='bar', ax=ax)
plt.title("Top-1 Offer Frequency (Bootstrap vs. No Bootstrap)")
plt.xlabel("Offer")
plt.ylabel("Count")
plt.xticks(rotation=0)
st.pyplot(fig)


st.header("🧪 Try It Yourself")
st.markdown("Select a customer and month to view recommended offers.")

customer_options = df_boot['Customer ID'].unique()
selected_customer = st.selectbox("Select Customer ID", sorted(customer_options))

month_options = df_boot[df_boot['Customer ID'] == selected_customer]['Month'].unique()
selected_month = st.selectbox("Select Month", sorted(month_options))

result_boot = df_boot[(df_boot['Customer ID'] == selected_customer) & (df_boot['Month'] == selected_month)]
result_noboot = df_noboot[(df_noboot['Customer ID'] == selected_customer) & (df_noboot['Month'] == selected_month)]

st.subheader("📌 Recommendations with Bootstrap")
if not result_boot.empty:
    st.write(result_boot['offers'].values[0])
else:
    st.warning("No recommendation found for this customer/month in bootstrap version.")

st.subheader("📌 Recommendations without Bootstrap")
if not result_noboot.empty:
    st.write(result_noboot['offers'].values[0])
else:
    st.warning("No recommendation found for this customer/month in no-bootstrap version.")

st.header("✅ Summary")
st.markdown("""
- **Model type**: User-based collaborative filtering
- **Evaluation**: Comparison between models with and without bootstrapping
- **Observed results**:
    - Top-1 match rate ~53%
    - Exact match (3/3 offers) ~18%
    - Confidence (common offers) ~76%
    - Precision@1 ~94%

This analysis suggests that bootstrapping improves confidence and stability without drastically altering the recommendation space.
""")
