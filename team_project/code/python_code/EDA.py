#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
#%%
df1=pd.read_csv(r"Data\depmap_export.csv")
df2=pd.read_csv(r"Data\RNAi_subset.csv")
merged_df = pd.merge(df1, df2, on='depmap_id', how='inner')
merged_df = merged_df.drop(columns=['cell_line_display_name_y', 'lineage_2_x','lineage_3_x', 'lineage_6_x', 'lineage_4_x', 'lineage_1_y', 'lineage_2_y', 'lineage_3_y', 'lineage_6_y', 'lineage_4_y'])
merged_df.to_csv("processed_df_with_all_target_genes.csv")
#%%
df = pd.read_csv("processed_df_with_all_target_genes.csv", index_col=0)
df = df.sort_values(by='lineage_1_x')
cancerlist=['CNS/Brain', 
            'Breast', 
            'Skin', 
            'Ovary/Fallopian Tube', 
            'Esophagus/Stomach']

targets = ['PARP1',
           'BRCA1',
           'BRCA2',
           'TP53',
           'BRIP1']

# %%
cellno=[]
for i in cancerlist:
    cellno.append(merged_df[merged_df['lineage_1_x']==i].shape[0])
    
plt.bar(cancerlist, cellno, width=0.8)
plt.ylabel("No. of cell lines")
plt.xticks(rotation=45) 
plt.show()
#%%
df = df[["depmap_id", "lineage_1_x",'PARP1', 'BRCA1','BRCA2','TP53','BRIP1' ]]
#%%
df_melted = df.melt(
    id_vars=['depmap_id', 'lineage_1_x'], # Columns to keep (Identifiers)
    value_vars=targets,                 # Columns to unpivot (The Genes)
    var_name='Gene',                  # New column name for headers
    value_name='Dependency_Score'     # New column name for values
)

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df_melted,
    x='Gene',
    y='Dependency_Score',
    hue='lineage_1_x',
    palette='viridis',
    cut=0 # Prevents the violin from extending past the data range
)

# Add reference lines
plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Non-essential (0)')
plt.axhline(-0.5, color='red', linestyle='--', alpha=0.5, label='Essential (-1)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Gene Dependency Scores across Lineages')
plt.xlabel('Gene')
plt.ylabel('Dependency Score')
plt.tight_layout()

#%%
for i in cancerlist:
    for j in targets:
        safe_i = str(i).replace("/", "_").replace("\\", "_")
        df1 = df[df['lineage_1_x']==i]
        plt.figure(figsize=(10, 6))
        plt.axvline(x=-0.5, color='red', linestyle='--', label="Dependency Threshold (-0.5)")
        sns.histplot(data=df1, x=j, kde=True, bins=30)
        plt.title(i+"\t"+j)
        filename = f"imgs/{safe_i+j}.png"
        plt.savefig(filename)
        plt.show()