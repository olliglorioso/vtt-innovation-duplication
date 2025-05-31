import pandas as pd

class Extractor():
    def __init__(self):
        pass
        
    def create_text_to_compare(self, df):
        text_to_compare = df["source id"] + " - " + df["source description"].fillna("") + " | Developed by " + df["target english_id"].fillna("")
        
        return text_to_compare   

    def filter_vtt_present_docs(self, df):
        """Filter to only include documents where VTT is present in any way"""
        vtt_present_docs = df[
            (df['target id'] == 'FI26473754') |          # VTT's official ID as target
            (df['source id'] == 'FI26473754') |          # VTT's official ID as source
            (df['target english_id'].str.contains('VTT', case=False, na=False)) |  # VTT in target name
            (df['source english_id'].str.contains('VTT', case=False, na=False)) |  # VTT in source name
            (df['target description'].str.contains('VTT', case=False, na=False)) | # VTT in target description
            (df['source description'].str.contains('VTT', case=False, na=False)) | # VTT in source description
            (df['relationship description'].str.contains('VTT', case=False, na=False)) # VTT in relationship description
        ]['Document number'].unique()
        
        # Return all rows from documents where VTT is present
        return df[df['Document number'].isin(vtt_present_docs)] 
    
    def extract(self, location="./data/results/df_combined.csv"):
        # Extract the CSV files
        vtt_domain_df = pd.read_csv("./data/dataframes/df_relationships_vtt_domain.csv")
        vtt_domain_df = vtt_domain_df[vtt_domain_df["relationship type"] == "DEVELOPED_BY"].copy()
        vtt_domain_df["text_to_compare"] = self.create_text_to_compare(vtt_domain_df)

        comp_domain_df = pd.read_csv("./data/dataframes/df_relationships_comp_url.csv")
        comp_domain_df = comp_domain_df[comp_domain_df["relationship type"] == "DEVELOPED_BY"].copy()
        comp_domain_df = comp_domain_df[comp_domain_df["source type"] != "Organization"]
        comp_domain_df["text_to_compare"] = self.create_text_to_compare(comp_domain_df)

        print("Comp domain df length:", len(comp_domain_df), " - VTT domain df length:", len(vtt_domain_df))
        print("Example of text to compare:", vtt_domain_df.iloc[0]["text_to_compare"])        
        
        vtt_domain_df["Document number"] = "VTT" + vtt_domain_df["Document number"].astype(str)
        comp_domain_df["Document number"] = "COMP" + comp_domain_df["Document number"].astype(str)

        df_combined = pd.concat([vtt_domain_df, comp_domain_df], ignore_index=True)
        df_combined.to_csv(location)

        # Hack 1: Create another exactly the same df, but without filtering developed_by and organization
        # Save it to location + .no_filter.csv
        new_vtt_domain_df = pd.read_csv("./data/dataframes/df_relationships_vtt_domain.csv")
        new_comp_domain_df = pd.read_csv("./data/dataframes/df_relationships_comp_url.csv")
        # These wont need embeddings so just combine them after giving the IDs
        new_vtt_domain_df["Document number"] = "VTT" + new_vtt_domain_df["Document number"].astype(str)
        new_comp_domain_df["Document number"] = "COMP" + new_comp_domain_df["Document number"].astype(str)
        new_df_combined = pd.concat([new_vtt_domain_df, new_comp_domain_df], ignore_index=True)
        new_df_combined.to_csv(location.replace(".csv", ".no_filter.csv"), index=False)
        
        comp_domain_vtt_present = self.filter_vtt_present_docs(comp_domain_df)
        comp_domain_vtt_present.to_csv("./data/results/df_comp_domain_vtt_present.csv", index=False)
        
