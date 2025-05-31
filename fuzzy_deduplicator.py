import pandas as pd
import pickle
import os
import json
from tqdm import tqdm
from typing import Dict, Optional
import dedupe
import dedupe.variables

class FuzzyDeduplicator:
    """Object-oriented fuzzy deduplication system for Innovation entities."""
    
    def __init__(self, csv_file: str):
        """
        Initialize the FuzzyDeduplicator.
        
        Args:
            csv_file: Path to CSV file with relationship data (from data_extractor.py)
        """
        self.csv_file = csv_file
        self.settings_file = 'innovation_dedupe_settings'
        self.training_file = 'innovation_dedupe_validation.json'
        self.output_file = 'innovation_duplicates.csv'
        self.threshold = None
        
        # Will be populated during processing   
        self.deduper = None
        self.duplicates_df = None

    @staticmethod
    def clean_text_for_dedupe(text):
        """Clean text data to remove problematic characters for dedupe"""
        if pd.isna(text) or text is None:
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove null characters and other control characters
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\r', ' ')   # Replace carriage returns with spaces
        text = text.replace('\n', ' ')   # Replace newlines with spaces
        text = text.replace('\t', ' ')   # Replace tabs with spaces
        
        # Remove other control characters (ASCII 0-31 except space)
        text = ''.join(char for char in text if ord(char) >= 32 or char == ' ')
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # Return cleaned text - since we pre-filter, we should have valid content
        cleaned = text.strip()
        return cleaned if cleaned else 'MISSING'  # Replace any remaining empty strings

    def analyze_training_data(self) -> Optional[Dict]:
        """Analyze and display information about the saved training data"""
        if not os.path.exists(self.training_file):
            print(f"Training file {self.training_file} not found.")
            return None
        
        print(f"\nAnalyzing training data from {self.training_file}...")
        
        try:
            with open(self.training_file, 'r') as f:
                training_data = json.load(f)
            
            # Count matches and distinct pairs
            match_count = len(training_data.get('match', []))
            distinct_count = len(training_data.get('distinct', []))
            total_labeled = match_count + distinct_count
            
            print(f"Training data summary:")
            print(f"  - Total labeled pairs: {total_labeled}")
            print(f"  - Duplicate pairs (matches): {match_count}")
            print(f"  - Distinct pairs (non-matches): {distinct_count}")
            
            if total_labeled > 0:
                match_ratio = match_count / total_labeled * 100
                print(f"  - Match ratio: {match_ratio:.1f}%")
            
            # Show sample pairs - handle the dedupe JSON format with __class__ and __value__
            if match_count > 0:
                print(f"\nSample duplicate pairs:")
                for i, pair_data in enumerate(training_data['match'][:3]):
                    print(f"  Pair {i+1}:")
                    # Handle dedupe's tuple format: {"__class__": "tuple", "__value__": [record1, record2]}
                    if isinstance(pair_data, dict) and "__value__" in pair_data:
                        record1, record2 = pair_data["__value__"]
                    else:
                        record1, record2 = pair_data
                    
                    print(f"    Record 1: {record1.get('source_english_id', 'N/A')} - {record1.get('source_description', 'N/A')[:50]}...")
                    print(f"    Record 2: {record2.get('source_english_id', 'N/A')} - {record2.get('source_description', 'N/A')[:50]}...")
                    print()
            
            if distinct_count > 0:
                print(f"Sample distinct pairs:")
                for i, pair_data in enumerate(training_data['distinct'][:3]):
                    print(f"  Pair {i+1}:")
                    # Handle dedupe's tuple format: {"__class__": "tuple", "__value__": [record1, record2]}
                    if isinstance(pair_data, dict) and "__value__" in pair_data:
                        record1, record2 = pair_data["__value__"]
                    else:
                        record1, record2 = pair_data
                    
                    print(f"    Record 1: {record1.get('source_english_id', 'N/A')} - {record1.get('source_description', 'N/A')[:50]}...")
                    print(f"    Record 2: {record2.get('source_english_id', 'N/A')} - {record2.get('source_description', 'N/A')[:50]}...")
                    print()
            
            return training_data
            
        except Exception as e:
            print(f"Error reading training file: {e}")
            return None

    def find_innovation_duplicates(self, csv_file: str) -> Optional[pd.DataFrame]:
        """Use dedupe library to find duplicate Innovation entries"""
        
        print(f"Loading data from CSV file: {csv_file}")
        df_relationships_comp_url = pd.read_csv(csv_file)
        
        # Filter for Innovation source types only
        innovation_df = df_relationships_comp_url[df_relationships_comp_url['source type'] == 'Innovation'].copy()
        
        print(f"Found {len(innovation_df)} Innovation entries before filtering")
        
        # Filter out rows with missing/empty values in key fields
        key_fields = ['source id', 'source english_id', 'source description']
        
        # Remove rows where key fields are NaN, None, or empty strings
        print("Filtering out rows with missing/empty key fields...")
        original_count = len(innovation_df)
        
        for field in key_fields:
            # Remove NaN/None values
            innovation_df = innovation_df[innovation_df[field].notna()]
            # Remove empty strings and whitespace-only strings
            innovation_df = innovation_df[innovation_df[field].astype(str).str.strip() != '']
        
        print(f"After filtering: {len(innovation_df)} entries remaining ({original_count - len(innovation_df)} removed)")
        
        if len(innovation_df) == 0:
            print("No Innovation entries found after filtering!")
            return None
        
        # Prepare data for dedupe - convert to list of dictionaries
        data_dict = {}
        print("Cleaning and preparing data...")
        
        for idx, row in innovation_df.iterrows():
            data_dict[idx] = {
                'source_id': self.clean_text_for_dedupe(row['source id']),
                'source_english_id': self.clean_text_for_dedupe(row['source english_id']),
                'source_description': self.clean_text_for_dedupe(row['source description']),
                'relationship_description': self.clean_text_for_dedupe(row['relationship description']),
                'target_id': self.clean_text_for_dedupe(row['target id']),
                'target_english_id': self.clean_text_for_dedupe(row['target english_id']),
                'document_number': str(row['Document number']),
                'source_company': self.clean_text_for_dedupe(row.get('Source Company', 'Unknown')),
            }
        
        # Validate no null characters remain
        print("Validating cleaned data...")
        null_char_found = False
        
        for record_id, record in data_dict.items():
            for field, value in record.items():
                if '\x00' in value:
                    print(f"Warning: Null character found in record {record_id}, field {field}")
                    null_char_found = True
        
        if not null_char_found:
            print("✓ Data cleaning successful - no null characters found")
        
        print(f"Prepared {len(data_dict)} records for deduplication")
        
        # Check if we have enough data for deduplication
        if len(data_dict) < 2:
            print("Error: Need at least 2 records for deduplication")
            return None
        elif len(data_dict) < 10:
            print("Warning: Very small dataset - results may not be reliable")
        
        print("Setting up dedupe...")
        
        # Define fields for deduplication using dedupe 3.0+ format
        # Using Text for longer fields and String for shorter identifiers
        fields = [
            dedupe.variables.Text('source_id', has_missing=True),
            dedupe.variables.Text('source_english_id', has_missing=True),
            dedupe.variables.Text('source_description', has_missing=True),
            dedupe.variables.Text('relationship_description', has_missing=True),
            dedupe.variables.Text('target_id', has_missing=True),
            dedupe.variables.Text('target_english_id', has_missing=True),
            dedupe.variables.String('document_number', has_missing=True),
        ]
        
        # Track whether we're using a pre-trained model
        using_pretrained_model = False
        
        # Check if we have trained settings file first
        if os.path.exists(self.settings_file):
            print('Reading from saved settings file...')
            with open(self.settings_file, 'rb') as f:
                self.deduper = dedupe.StaticDedupe(f)
            using_pretrained_model = True
        else:
            # Create deduper for training
            self.deduper = dedupe.Dedupe(fields)
            
            # Prepare training with existing training data if available
            print("Preparing training data...")
            
            # Adjust sample size based on dataset size
            sample_size = min(15000, len(data_dict) * (len(data_dict) - 1) // 4)  # At most 1/4 of all possible pairs
            print(f"Using sample size: {sample_size}")
            
            if os.path.exists(self.training_file):
                print(f'Reading existing training data from {self.training_file}...')
                with open(self.training_file, 'r') as f:
                    self.deduper.prepare_training(data_dict, training_file=f, sample_size=sample_size)
            else:
                print('No existing training data found, preparing fresh training set...')
                self.deduper.prepare_training(data_dict, sample_size=sample_size)
            
            # Need to do active learning
            print("\nStarting active learning...")
            print("You will be asked to label pairs as duplicates or not.")
            print("Type 'y' for yes (duplicate), 'n' for no (not duplicate), 'u' for unsure, 's' to stop, 'f' to finish")
            print("NOTE: The document_number field will now be visible during comparison!")
            
            # Redirect stdin for interactive training
            dedupe.console_label(self.deduper)
            
            # Save the raw training data (labeled pairs) after manual labeling
            print(f"Saving raw training data to {self.training_file}...")
            with open(self.training_file, 'w') as f:
                self.deduper.write_training(f)
            
            # Train the model
            print("Training model...")
            self.deduper.train()
            
            # Save settings
            with open(self.settings_file, 'wb') as f:
                self.deduper.write_settings(f)
        
        # Set threshold - handle differently for pre-trained vs newly trained models
        if self.threshold is not None:
            threshold = self.threshold
            print(f"Using provided threshold: {threshold}")
        elif using_pretrained_model:
            # For pre-trained models, use a reasonable default threshold
            threshold = 0.5
            print(f"Using default threshold for pre-trained model: {threshold}")
        else:
            # For newly trained models, calculate optimal threshold
            threshold = self.deduper.threshold(data_dict, recall_weight=1)
            print(f"Calculated optimal threshold: {threshold}")
        
        # Find duplicates
        print("Finding duplicate clusters...")
        
        # Use partition method which works for both Dedupe and StaticDedupe
        clustered_dupes = self.deduper.partition(data_dict, threshold)
        
        print(f'Found {len(clustered_dupes)} duplicate clusters')
        
        # Process results
        duplicate_results = []
        for (cluster_id, cluster) in enumerate(clustered_dupes):
            id_set, scores = cluster
            cluster_d = [data_dict[c] for c in id_set]
            
            # Get original dataframe indices
            df_indices = list(id_set)
            
            for record_id, score in zip(id_set, scores):
                duplicate_results.append({
                    'cluster_id': cluster_id,
                    'record_id': record_id,
                    'confidence_score': score,
                    'document_number': data_dict[record_id]['document_number'],
                    'source_company': data_dict[record_id]['source_company'],
                    'source_id': data_dict[record_id]['source_id'],
                    'source_english_id': data_dict[record_id]['source_english_id'],
                    'source_description': data_dict[record_id]['source_description'][:100] + '...' if len(data_dict[record_id]['source_description']) > 100 else data_dict[record_id]['source_description'],
                    'target_english_id': data_dict[record_id]['target_english_id']
                })
        
        # Convert to dataframe
        duplicates_df = pd.DataFrame(duplicate_results)
        
        # Sort by cluster_id and confidence score
        duplicates_df = duplicates_df.sort_values(['cluster_id', 'confidence_score'], ascending=[True, False])
        
        print("\nDuplicate clusters found:")
        if len(duplicates_df) > 0:
            print(duplicates_df.groupby('cluster_id').size().describe())
            
            # Analyze cross-document vs same-document duplicates
            cross_document_clusters = 0
            same_document_clusters = 0
            
            for cluster_id in duplicates_df['cluster_id'].unique():
                cluster_data = duplicates_df[duplicates_df['cluster_id'] == cluster_id]
                unique_docs = cluster_data['document_number'].nunique()
                
                if unique_docs > 1:
                    cross_document_clusters += 1
                else:
                    same_document_clusters += 1
            
            print(f"\nCluster analysis:")
            print(f"  - Clusters spanning multiple documents: {cross_document_clusters}")
            print(f"  - Clusters within same document: {same_document_clusters}")
            print(f"  - Cross-document ratio: {cross_document_clusters / (cross_document_clusters + same_document_clusters) * 100:.1f}%")
        
        # Save results
        duplicates_df.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")
        
        # Show sample results
        print("\nSample duplicate clusters:")
        for cluster_id in duplicates_df['cluster_id'].unique()[:5]:
            cluster_data = duplicates_df[duplicates_df['cluster_id'] == cluster_id]
            unique_docs = cluster_data['document_number'].nunique()
            unique_companies = cluster_data['source_company'].nunique()
            
            print(f"\nCluster {cluster_id} ({len(cluster_data)} records, {unique_docs} documents, {unique_companies} companies):")
            for _, row in cluster_data.iterrows():
                print(f"  - Score: {row['confidence_score']:.3f}")
                print(f"    Doc: {row['document_number']} | Company: {row['source_company']}")
                print(f"    ID: {row['source_id']}")
                print(f"    English ID: {row['source_english_id']}")
                print(f"    Description: {row['source_description']}")
                print(f"    Target: {row['target_english_id']}")
                print()
        
        self.duplicates_df = duplicates_df
        return duplicates_df

    def run_deduplication(self, csv_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Main method to run the complete deduplication process.
        
        Args:
            csv_file: Path to CSV file with relationship data (if None, uses the one from constructor)
            
        Returns:
            DataFrame containing duplicate clusters or None if no duplicates found
        """
        print("Starting Innovation deduplication process...")
        
        # Use provided csv_file or fall back to the one from constructor
        source_file = csv_file if csv_file is not None else self.csv_file
        
        # Check if CSV file exists
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"CSV file not found: {source_file}")
        
        # Find duplicates
        duplicates_df = self.find_innovation_duplicates(source_file)
        
        if duplicates_df is not None:
            print(f"\nDeduplication complete! Found {len(duplicates_df)} duplicate records in {duplicates_df['cluster_id'].nunique()} clusters.")
        else:
            print("No duplicates found or no Innovation entries available.")
        
        # Analyze the training data that was created/used
        self.analyze_training_data()
        
        print("\nFiles created:")
        print(f"- {self.output_file}: Duplicate clusters found")
        print(f"- {self.training_file}: Raw manual labeling data")
        print(f"- {self.settings_file}: Trained model settings")
        
        return duplicates_df

    def get_duplicates_summary(self) -> Dict:
        """Get a summary of duplicate findings"""
        if self.duplicates_df is None:
            return {"error": "No duplicates found yet. Run deduplication first."}
        
        return {
            "total_duplicate_records": len(self.duplicates_df),
            "total_clusters": self.duplicates_df['cluster_id'].nunique(),
            "cross_document_clusters": len([
                cluster_id for cluster_id in self.duplicates_df['cluster_id'].unique()
                if self.duplicates_df[self.duplicates_df['cluster_id'] == cluster_id]['document_number'].nunique() > 1
            ]),
            "average_cluster_size": self.duplicates_df.groupby('cluster_id').size().mean(),
            "files_generated": [self.output_file, self.training_file, self.settings_file]
        }


def main():
    """Main function to run the deduplication process - kept for backward compatibility"""
    csv_file = "./data/results/df_combined.csv"
    deduplicator = FuzzyDeduplicator(csv_file)
    deduplicator.run_deduplication()


# Convenience function to work with data_extractor
def run_deduplication_with_extractor(extractor_output_file: str = "./data/results/df_combined.csv") -> Optional[pd.DataFrame]:
    """
    Convenience function to run deduplication with CSV file from data_extractor.py
    
    Args:
        extractor_output_file: Path to the CSV file created by data_extractor.py
        
    Returns:
        DataFrame containing duplicate clusters or None if no duplicates found
    """
    deduplicator = FuzzyDeduplicator(extractor_output_file)
    return deduplicator.run_deduplication()


if __name__ == "__main__":
    main()