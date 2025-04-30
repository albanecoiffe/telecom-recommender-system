# 📡 Telecom Offer Recommender System

[Streamlit page](https://telecom-recommender-system.streamlit.app/)     

This project implements a recommender system for telecom products and services, designed to suggest the most relevant offers to customers based on their behavior, usage patterns, and churn history.

The solution includes:
- Collaborative filtering-based modeling
- Comparative evaluation of two versions: with and without bootstrapping
- An interactive Streamlit dashboard for exploration and demonstration

---

## 🎯 Business Objective

The goal is to increase customer satisfaction, reduce churn, and maximize revenue by recommending personalized offers to telecom users.

---

## 🧠 Model Description

We use **user-based collaborative filtering**, where customers are compared based on their features (demographics, usage, etc.). Recommendations are derived from the most similar past customers.

Key points:
- Similarity is computed using distance measures: **cosine**, **euclidean**, or **manhattan**
- The model identifies the top `n` similar non-churned customers
- Most frequent offers from this group are selected and filtered by a minimal prevalence threshold
- Two versions of the model:
  - 🔸 **Without bootstrapping**: a single prediction from nearest neighbors
  - 🔹 **With bootstrapping**: aggregates predictions over multiple subsamples for robustness

---

## 🚀 Streamlit App

Launch the interactive app with:

```bash
streamlit run app.py
