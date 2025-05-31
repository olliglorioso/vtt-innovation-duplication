#!/usr/bin/env python3
"""
Embedding Generator for Innovation Analysis

This script generates embeddings for innovation data using different approaches:
- Local embeddings using SentenceTransformer models
- OpenAI/Azure OpenAI embeddings
- Extensible architecture for additional embedding providers

Input file: data/dataframes/df_combined.csv
Output file: data/dataframes/embeddings.csv

Usage:
    python embedding_generator.py --method local
    python embedding_generator.py --method openai
"""

import argparse
import json
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the embedding provider"""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using SentenceTransformer models"""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Initialized local embedding model: {model_name}")
        except ImportError:
            raise ImportError("sentence-transformers is required for local embeddings. Install with: pip install sentence-transformers")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local SentenceTransformer model"""
        logger.info(f"Generating embeddings for {len(texts)} texts using local model")
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        
        # Convert to numpy array if it's a tensor
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    @property
    def name(self) -> str:
        return f"local_{self.model_name}"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI/Azure OpenAI embedding provider"""
    
    def __init__(self, model_key: str = 'gpt-4o-mini'):
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError:
            raise ImportError("langchain-openai is required for OpenAI embeddings. Install with: pip install langchain-openai")
        
        self.config_path = 'data/keys/azure_config.json'
        self.model_key = model_key
        self.embedding_model = self._initialize_embeddings()
        logger.info(f"Initialized OpenAI embedding model with key: {model_key}")
    
    def _initialize_embeddings(self):
        """Initialize Azure OpenAI embeddings client"""
        with open(self.config_path, 'r') as jsonfile:
            config = json.load(jsonfile)
        
        # Extract base URL (remove the specific deployment path)
        api_base = config[self.model_key]['api_base'].split('/openai/deployments')[0]
        
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            api_key=config[self.model_key]['api_key'],
            azure_endpoint=api_base,
            api_version=config[self.model_key]['api_version'],
            azure_deployment=config[self.model_key]['emb_deployment']
        )
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        logger.info(f"Generating embeddings for {len(texts)} texts using OpenAI API")
        
        # Process in batches to handle API limits
        batch_size = 100
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    @property
    def name(self) -> str:
        return f"openai_{self.model_key}"


class EmbeddingGenerator:
    """Main class for generating embeddings from innovation data"""
    
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
    
    def generate_embeddings(self, input_file: str, output_file: str):
        """Generate embeddings and save to CSV"""
        logger.info(f"Loading data from {input_file}")
        
        # Load the data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from input file")
        
        # Prepare text data
        texts = df["text_to_compare"].tolist()
        logger.info(f"Prepared {len(texts)} texts for embedding generation")
        
        if len(texts) == 0:
            logger.warning("No texts found after filtering. Check your data format.")
            return
        
        # Generate embeddings
        embeddings = self.provider.encode(texts)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Create output dataframe
        output_df = df.copy()
        
        # Add embedding columns
        # Store embeddings as a single column containing lists
        output_df['embedding'] = embeddings.tolist()
        
        # Combine original data with embeddings
        result_df = output_df.copy()
        
        # Add metadata
        result_df['embedding_provider'] = self.provider.name
        result_df['embedding_dimension'] = embeddings.shape[1]
        
        # Save to CSV
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved embeddings to {output_file}")
        
        return result_df


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for innovation data")
    parser.add_argument("--method", choices=["local", "openai"], required=True,
                        help="Embedding method to use")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Model name for local embeddings (default: all-mpnet-base-v2)")
    parser.add_argument("--model-key", default="gpt-4.1-mini",
                        help="Model key for OpenAI embeddings (default: gpt-4.1-mini)")
    
    args = parser.parse_args()
    
    # Hardcoded input and output paths
    input_file = "data/dataframes/df_combined.csv"
    output_file = "data/dataframes/embeddings.csv"
    
    # Validate input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} does not exist")
        return
    
    # Initialize embedding provider
    try:
        if args.method == "local":
            provider = LocalEmbeddingProvider(args.model)
        elif args.method == "openai":
            # Check if config file exists
            config_path = 'data/keys/azure_config.json'
            if not Path(config_path).exists():
                logger.error(f"Config file {config_path} does not exist")
                return
            provider = OpenAIEmbeddingProvider(args.model_key)
        else:
            logger.error(f"Unknown method: {args.method}")
            return
    except Exception as e:
        logger.error(f"Failed to initialize embedding provider: {e}")
        return
    
    # Generate embeddings
    generator = EmbeddingGenerator(provider)
    
    try:
        generator.generate_embeddings(input_file, output_file)
        logger.info("Embedding generation completed successfully")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise


if __name__ == "__main__":
    main() 