# VTT Innovation Duplication Analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Setup Environment

```bash
# Clone and navigate to project
cd vtt-innovation-duplication

# Install dependencies with uv
uv sync
```

### 2. Create Required Directories and Files

```bash
# unzip data.zip
unzip data.zip

# Create results directory
mkdir -p data/results

# Create Azure config (see setup section below)
mkdir -p data/keys
```

### 3. Configure Azure OpenAI (Required for OpenAI embeddings)

Create `data/keys/azure_config.json` with your Azure OpenAI credentials:

```json
{
  "gpt-4.1-mini": {
    "api_base": "https://your-resource.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview",
    "api_version": "2025-01-01-preview",
    "api_key": "your-api-key-here",
    "deployment": "gpt-4.1-mini",
    "eval_deployment": "gpt-4.1-mini",
    "emb_deployment": "text-embedding-3-large"
  }
}
```

## 📊 Data Pipeline

### Step 1: Data Analysis (Jupyter Notebook)

Run the main analysis in the Jupyter notebook:

```bash
uv run jupyter lab
# Open and run main.ipynb
```

**What it does:**

#### **Data Extraction & Preprocessing**

- Loads VTT domain relationships (`df_relationships_vtt_domain.csv`)
- Loads comparison domain relationships (`df_relationships_comp_url.csv`)
- Filters for "DEVELOPED_BY" relationships
- Identifies documents with VTT presence
- Creates text for comparison: `"source id - source description | Developed by target english_id"`
- Outputs: `data/results/df_combined.csv` and `data/results/df_comp_domain_vtt_present.csv`

#### **Embedding Generation**

- **Multiple Provider Support**: Local SentenceTransformer models and Azure OpenAI embeddings
- **Extensible Architecture**: Easy to add new embedding providers through abstract base class
- **Model Options**:
  - Local: `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, etc.
  - OpenAI: `text-embedding-3-large`
- Generates semantic embeddings for the "text_to_compare" field
- Outputs: `data/results/embeddings.csv` with embedding vectors and metadata

#### **Similarity Analysis & Grouping**

- Calculates cosine similarity matrices between all embeddings
- Tests multiple similarity thresholds (0.6 to 0.99) to analyze grouping behavior
- Groups similar pairs into connected components using graph traversal (DFS)
- Analyzes relationship between threshold, number of groups, and group sizes
- Identifies potential innovation duplications above specified threshold (default: 0.8)

#### **Visualization & Analysis**

- **Network Graph Generation**: Creates interactive network graphs showing innovation relationships
- Generates threshold analysis charts showing:
  - Number of groups vs threshold
  - Number of similar pairs vs threshold
  - Mean group size vs threshold
- **Embedding Space Visualization**: 2D projections of high-dimensional embeddings

#### **LLM-Based Validation**

- Uses Azure OpenAI (GPT-4.1-mini) to validate potential duplicate groups
- **False Positive Detection**: Identifies and removes incorrectly grouped innovations
- **Structured Validation**: Uses Pydantic models for consistent reasoning output
- **Reasoning Capture**: Stores detailed reasoning for each validation decision

#### **Unified Innovation Analysis**

- **Organization Relationship Mapping**: Tracks all organizations involved with innovation groups
- **Role Classification**: Identifies VTT's role as developer, collaborator, or unknown
- **Comprehensive Innovation Profiles**: Creates unified records with aliases, descriptions, and organizational relationships

#### **Final Outputs**

- `data/results/duplicates.csv` - All identified duplicates
- `data/results/groups_final.json` - Initial similarity groups
- `data/results/enhanced_groups.json` - Validated groups after LLM review
- `data/results/results_with_reasoning.json` - LLM validation results with reasoning
- `data/results/unified_innovations_5.json` - Final unified innovation records
- `data/results/vtt_unified_innovations.csv` - CSV format of unified innovations

### Step 2: Evaluation Framework

#### **Validation dataset creation**

Used dedupe's interactive labeling CLI to manually label 100 pairs of innovations as duplicate or distinct. This was used as the validation dataset and saved as: `innovation_dedupe_validation.json`.

**Comprehensive Clustering Evaluation:**

```bash
uv run evaluate_clustering.py data/validation/innovation_dedupe_validation.json data/results/enhanced_groups.json -o data/results/clustering_evaluation_results.json
```

**Features:**

- **Pairwise Metrics**: Precision, recall, F1-score, and accuracy for pair-wise duplicate detection
- **Cluster-level Metrics**: Adjusted Rand Index, Normalized Mutual Information, Homogeneity, Completeness, V-measure
- **Confusion Matrix Analysis**: Detailed breakdown of true/false positives and negatives
- **Ground Truth Comparison**: Validates clustering results against manually labeled data
- **Detailed Error Analysis**: Shows examples of correctly and incorrectly clustered pairs
- **Cross-validation Support**: Handles different data formats and composite key matching

### Step 3: Launch Visualization Dashboard

```bash
uv run streamlit run viz.py
```

**Features:**

- **Interactive Network Graphs**: Explore innovation relationships with node/edge filtering
- **Embedding Visualizations**: 2D projections of high-dimensional embedding space with group coloring
- **Threshold Analysis**: Interactive charts showing impact of similarity thresholds
- **Performance Metrics Dashboard**: Real-time evaluation results and confusion matrices
- **Multi-tab Interface**: Organized views for different analysis aspects
- **Exportable Results**: Download analysis results and visualizations

## File explanations

### Data Extractor (`data_extractor.py`)

- Prepare and filter innovation relationship data, identifying documents containing VTT

### Embedding Generator (`embedding_generator.py`)

- Convert text descriptions to semantic vectors using multiple embedding providers (local and Azure OpenAI)

### Fuzzy Deduplicator (`fuzzy_deduplicator.py`)

- Interactive deduplication using machine learning with manual labeling and confidence scoring

### Evaluation Framework (`evaluate_clustering.py`)

- Comprehensive validation of clustering results with multiple metrics and ground truth comparison

### Visualization (`viz.py`)

- Interactive Streamlit dashboard for exploring results with network graphs, threshold analysis, and evaluation metrics
