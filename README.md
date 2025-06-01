# VTT Innovation Duplication Analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
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

#### **Fuzzy Deduplication System** _(Validation Dataset Creation)_

- **Interactive Manual Labeling**: Used to create ground truth validation dataset through manual pair labeling
- **Dedupe Library Integration**: Uses advanced record linkage techniques with field-specific variables
- **Training Data Generation**: Creates labeled positive and negative duplicate pairs for evaluation
- **Comprehensive Field Analysis**: Analyzes multiple fields (ID, description, relationships, targets)
- **Confidence Scoring**: Provides confidence scores for each potential duplicate pair
- **Cross-document Duplicate Detection**: Identifies duplicates across different data sources
- Creates training data saved as: `innovation_dedupe_validation.json`

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
- `innovation_duplicates.csv` - Fuzzy deduplication results (if using alternative method)

### Step 2: Evaluation Framework

**Comprehensive Clustering Evaluation:**

```bash
uv run python evaluate_clustering.py ground_truth.json predictions.json
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

## 📁 Project Structure

```
vtt-innovation-duplication/
├── data/
│   ├── dataframes/           # Input CSV files
│   │   ├── df_relationships_vtt_domain.csv
│   │   ├── df_relationships_comp_url.csv
│   │   ├── df_combined.csv   # Generated by data_extractor.py
│   │   └── embeddings.csv    # Generated by embedding_generator.py
│   ├── results/              # Output files (create this folder)
│   │   ├── duplicates.csv
│   │   ├── groups_final.json
│   │   ├── enhanced_groups.json
│   │   ├── unified_innovations_5.json
│   │   └── *.pkl             # Visualization cache files
│   └── keys/                 # API keys (create this folder)
│       └── azure_config.json # Azure OpenAI config
├── data_extractor.py         # Step 1: Data preparation
├── embedding_generator.py    # Step 2: Embedding generation
├── fuzzy_deduplicator.py     # Alternative: Interactive fuzzy deduplication
├── evaluate_clustering.py    # Evaluation framework
├── viz.py                   # Step 3: Streamlit dashboard
├── main.ipynb              # Jupyter notebook for analysis
├── pyproject.toml          # uv configuration
└── README.md
```

## 🔬 Analysis Components

### Data Extractor (`data_extractor.py`)

- **Purpose**: Prepare and filter innovation relationship data
- **Key Method**: `filter_vtt_present_docs()` - identifies documents containing VTT
- **Multiple Output Formats**: Filtered and unfiltered datasets for different analysis needs
- **Document Prefixing**: Adds source identifiers (VTT vs COMP) for tracking
- **Output**: Combined dataset with VTT and comparison domains

### Embedding Generator (`embedding_generator.py`)

- **Purpose**: Convert text descriptions to semantic vectors
- **Provider Architecture**: Extensible system supporting multiple embedding sources
- **Methods**:
  - **Local**: SentenceTransformer models (offline, fast, free)
  - **OpenAI**: Azure OpenAI text-embedding-3-large (API-based, high-quality)
- **Batch Processing**: Handles large datasets efficiently with progress tracking
- **Output**: High-dimensional embeddings for similarity analysis with metadata

### Fuzzy Deduplicator (`fuzzy_deduplicator.py`)

- **Purpose**: Interactive deduplication using machine learning
- **Training Interface**: Manual labeling system for building custom models
- **Model Persistence**: Save/load trained models for consistent results
- **Field-Specific Analysis**: Handles different data types (text, strings, IDs)
- **Confidence Scoring**: Provides reliability measures for each duplicate detection
- **Cross-Document Detection**: Identifies duplicates spanning multiple data sources

### Evaluation Framework (`evaluate_clustering.py`)

- **Purpose**: Comprehensive validation of clustering results
- **Multiple Metrics**: Both pairwise and cluster-level evaluation measures
- **Ground Truth Integration**: Compares results against manually validated data
- **Error Analysis**: Detailed breakdown of classification errors
- **Flexible Input**: Handles different clustering result formats
- **Exportable Reports**: Saves evaluation results in JSON format

### Visualization (`viz.py`)

- **Purpose**: Interactive dashboard for exploring results
- **Tabs**:
  - **Overview**: Project metrics and summary statistics
  - **Network Graph**: Interactive innovation relationship networks with filtering
  - **Graphs**: Threshold analysis and embedding visualizations
  - **Evaluation**: Model performance metrics and validation results
- **Caching**: Efficient loading of pre-computed visualizations
- **Export Capabilities**: Download charts and analysis results

## 🎯 Project Purpose

This project was developed for analyzing innovation duplication patterns in research organizations. The comprehensive toolkit provides:

1. **Multi-Method Duplication Detection**:

   - Semantic similarity using embeddings
   - Interactive fuzzy matching with machine learning
   - Manual validation with LLM assistance

2. **Comprehensive Evaluation**:

   - Multiple evaluation metrics
   - Ground truth validation
   - Error analysis and debugging tools

3. **Network Analysis**:

   - Innovation relationship mapping
   - Organizational collaboration patterns
   - Cross-document innovation tracking

4. **Interactive Exploration**:

   - Web-based dashboard
   - Real-time filtering and analysis
   - Exportable results and visualizations

5. **Production-Ready Pipeline**:
   - Extensible architecture
   - Multiple data source support
   - Scalable processing for large datasets

## 🔧 Development

### Adding Dependencies

```bash
uv add package-name
```

### Development Dependencies

```bash
uv sync --dev  # Includes jupyter, pytest, black, flake8
```

### Running Different Analysis Methods

```bash
# Main embedding-based analysis
uv run jupyter lab  # Run main.ipynb

# Alternative fuzzy deduplication
uv run python fuzzy_deduplicator.py

# Evaluation framework
uv run python evaluate_clustering.py ground_truth.json results.json

# Visualization dashboard
uv run streamlit run viz.py
```

### Custom Embedding Providers

Extend the `EmbeddingProvider` abstract base class to add new embedding sources:

```python
class CustomEmbeddingProvider(EmbeddingProvider):
    def encode(self, texts: List[str]) -> np.ndarray:
        # Implement your embedding logic
        pass

    @property
    def name(self) -> str:
        return "custom_provider"
```
