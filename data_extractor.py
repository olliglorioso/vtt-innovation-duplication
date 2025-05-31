import pandas as pd

class Extractor():
    def __init__(self):
        pass
        
    def create_text_to_compare(self, df):
        text_to_compare = df["source id"] + " - " + df["source description"].fillna("") + " | Developed by " + df["target english_id"].fillna("")
        return text_to_compare   
    
    def extract(self, location="./data/results/df_combined.csv"):
        # Extract the CSV files
        vtt_domain_df = pd.read_csv("./data/dataframes/df_relationships_vtt_domain.csv")
        comp_domain_df = pd.read_csv("./data/dataframes/df_relationships_comp_url.csv")
        
        # Include both DEVELOPED_BY and COLLABORATION relationships
        vtt_domain_df = vtt_domain_df[
            vtt_domain_df["relationship type"].isin(["DEVELOPED_BY", "COLLABORATION"])
        ].copy()
        
        comp_domain_df = comp_domain_df[
            comp_domain_df["relationship type"].isin(["DEVELOPED_BY", "COLLABORATION"])
        ].copy()
        
        # Create text to compare for ALL rows (both Innovation and Organization sources)
        vtt_domain_df["text_to_compare"] = self.create_text_to_compare(vtt_domain_df)
        comp_domain_df["text_to_compare"] = self.create_text_to_compare(comp_domain_df)
        
        # FIXED: Only drop duplicates for INNOVATIONS, not for COLLABORATIONS
        # For innovations, drop duplicates on source description
        vtt_innovations = vtt_domain_df[vtt_domain_df["source type"] == "Innovation"].drop_duplicates(subset="source description", keep="first")
        comp_innovations = comp_domain_df[comp_domain_df["source type"] == "Innovation"].drop_duplicates(subset="source description", keep="first")
        
        # For collaborations, keep ALL records (no duplicate dropping)
        vtt_collaborations = vtt_domain_df[vtt_domain_df["source type"] == "Organization"]
        comp_collaborations = comp_domain_df[comp_domain_df["source type"] == "Organization"]
        
        # Combine innovations and collaborations separately, then merge
        vtt_domain_df = pd.concat([vtt_innovations, vtt_collaborations], ignore_index=True)
        comp_domain_df = pd.concat([comp_innovations, comp_collaborations], ignore_index=True)
        
        # Combine and save ALL relationship data
        df_combined = pd.concat([vtt_domain_df, comp_domain_df], ignore_index=True)
        df_combined.to_csv(location, index=False)
        
        #comp_domain_vtt_present.to_csv("./data/results/df_comp_domain_vtt_present.csv", index=False)