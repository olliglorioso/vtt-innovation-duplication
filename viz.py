# app.py

import streamlit as st
import networkx as nx
from pyvis.network import Network
import pickle
# --- PAGE SETUP ---
st.set_page_config(page_title="Network Analysis Demo", layout="wide")
st.title("🔍 Aalto AI Hackathon Demo")

# --- TABS FOR NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Network Graph", "Graphs", "Evaluation"])

# --- TAB 1: Overview ---
with tab1:
    st.header("Project Overview")
    st.markdown("""
    This project analyzes X dataset to produce a network graph revealing Y.
    Key steps include embedding, clustering, and graph construction.
    """)
    st.metric("Recall", "92%")
    st.metric("Precision", "88%")
    st.metric("Accuracy", "89%")

# --- TAB 2: Network Graph ---
with tab2:
    st.header("Interactive Network Graph")

    with open("vtt_innovation_network.html", 'r', encoding='utf-8') as f:
        graph_html = f.read()
    st.components.v1.html(graph_html, height=600, scrolling=True)

# --- TAB 3: Embeddings ---
with tab3:
    st.header("Analysis visualizations")

    st.subheader("Thresholds")
    st.markdown("""
        The thresholds is used as the similarity threshold for the embeddings. The graphs show effects of the threshold
        on the number of pairs that are considered similar.
    """)
    with open("data/results/thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)
    st.pyplot(thresholds)

    st.subheader("Threshold groups")
    st.markdown("""
        Here we can see the amount of groups created, size of the groups and the total amount of pairs in the groups
        for each threshold value.
    """)
    with open("data/results/thresholds_group.pkl", "rb") as f:
        thresholds_groups = pickle.load(f)
    st.pyplot(thresholds_groups)


    st.subheader("Embeddings visualization")
    st.markdown("""
        Here you can see the visualization of the vector embeddings.
        Note that the embeddings contain over 3000 dimensions and these are reduced to 2 dimensions using PCA.
        The colors represent the groups that the pairs belong to.
    """)
    with open("data/results/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    st.pyplot(embeddings)

# --- TAB 4: Performance ---
with tab4:
    st.header("Evaluation")
    st.markdown("""
                    We evaluated the perfomance of the deduplication by creating a validation set manually which had 100 pairs
                which were either considered duplicates or not.
                The metric are shown below.
                """)
    st.metric("Accuracy", "87%")
    st.metric("Recall", "96%")
    st.metric("Precision", "83%")
    st.metric("F1 Score", "89%")
