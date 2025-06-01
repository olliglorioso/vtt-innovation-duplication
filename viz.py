import streamlit as st
import pickle
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="VTT Innovation Duplication Analysis", 
    layout="wide",
    page_icon="🔬"
)
st.title("🔬 VTT Innovation Duplication Analysis")

# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_pickle_file(file_path):
    """Load pickle file with error handling"""
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# --- TABS FOR NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🌐 Network Graph", "📊 Analysis", "📈 Evaluation"])

# --- TAB 1: Overview ---
with tab1:
    st.header("Innovation Duplication Analysis")
    st.markdown("""
    **Our Three-Step Approach:**
    
    **Step 1: Group Similar Innovations using embeddings**
    - Generate semantic embeddings from innovation descriptions and titles
    - Use similarity thresholds to identify potential duplicate clusters
    - Scale analysis across thousands of innovation records
    
    **Step 2: Validate Groups with LLM**
    - Azure OpenAI reviews each cluster for false positives
    - Removes incorrectly grouped innovations with detailed reasoning
    - Ensures high precision while maintaining recall
    
    **Step 3: Aggregate Results with LLM**
    - LLM combines information from multiple sources about the same innovation
    - Creates unified innovation profiles preserving all source details
    - Maintains full traceability while consolidating descriptions
    """)
    
    st.markdown("---")
    st.markdown("**📍 Navigate through the tabs above to explore detailed results and interactive visualizations.**")

# --- TAB 2: Network Graph ---
with tab2:
    st.header("🌐 Interactive Innovation Network")
    st.markdown("Explore relationships between innovations, organizations, and development patterns.")
    
    try:
        if os.path.exists("vtt_innovation_network.html"):
            with open("vtt_innovation_network.html", 'r', encoding='utf-8') as f:
                graph_html = f.read()
            st.components.v1.html(graph_html, height=600, scrolling=True)
        else:
            st.error("Network graph file not found. Please run the analysis first.")
    except Exception as e:
        st.error(f"Error loading network graph: {str(e)}")

# --- TAB 3: Analysis ---
with tab3:
    st.header("📊 Threshold & Embedding Analysis")

    # Threshold Analysis
    st.subheader("🎯 Similarity Threshold Effects")
    st.markdown("""
    Similarity thresholds determine how strict the clustering is. Higher thresholds = fewer, more confident groups.
    """)
    
    thresholds = load_pickle_file("./thresholds.pkl")
    if thresholds is not None:
        st.pyplot(thresholds)
    
    # Group Analysis
    st.subheader("👥 Group Formation Analysis")
    st.markdown("""
    This shows how threshold values affect group size distribution and total clustering behavior.
    """)
    
    thresholds_groups = load_pickle_file("./thresholds_group.pkl")
    if thresholds_groups is not None:
        st.pyplot(thresholds_groups)

    # Embedding Visualization
    st.subheader("🎨 Embedding Space Visualization")
    st.markdown("""
    **3000+ dimensional embeddings** reduced to 2D using PCA. Colors represent duplicate groups.
    Points close together = semantically similar innovations.
    """)
    
    embeddings = load_pickle_file("./embeddings.pkl")
    if embeddings is not None:
        st.pyplot(embeddings)

# --- TAB 4: Evaluation ---
with tab4:
    st.header("📈 Performance Evaluation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Key Metrics")
        st.metric("Accuracy", "87%", delta="High overall performance")
        st.metric("Recall", "96%", delta="Excellent duplicate detection")
        st.metric("Precision", "83%", delta="Good false positive control")
        st.metric("F1 Score", "89%", delta="Strong balanced performance")
    
    with col2:
        st.subheader("🎯 Validation Approach")
        st.markdown("""
        **Manual Validation Set:**
        - 100 carefully labeled innovation pairs
        - Mix of true duplicates and distinct innovations
        """)
    
    st.markdown("---")
    st.info("""
    **Note on Results:** The dataset contains many "obvious" cases which inflate accuracy scores. 
    However, with 100 manually validated pairs, these metrics demonstrate reliable performance. 
    Manual validation was limited by the time-intensive nature of expert review.
    """)