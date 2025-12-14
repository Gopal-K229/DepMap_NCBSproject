#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

#%%
df1=pd.read_csv(r"Data\depmap_export.csv")
df2=pd.read_csv(r"Data\RNAi_subset.csv")
# %%
outcomegene= ['PARP1', 'BRCA1','BRCA2','TP53','BRIP1']
for i in outcomegene:
    df2[i].plot(kind='hist', bins=20, title='Distribution of Value Column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
#%%
cancerlist=['CNS/Brain', 'Breast', 'Skin', 'Ovary/Fallopian Tube',
       'Esophagus/Stomach']

#%%
# os.makedirs("imgs", exist_ok=True)

# for j in cancerlist:
#     moddf = df2[df2['lineage_1'] == j]

#     # clean lineage name for filename
#     safe_j = str(j).replace("/", "_").replace("\\", "_")

#     for i in outcomegene:
#         safe_i = str(i).replace("/", "_").replace("\\", "_")

#         plt.figure(figsize=(6,4))
#         moddf.boxplot(column=i)
#         plt.title(f"{i} in {j}")
#         plt.tight_layout()

#         # save directly inside imgs/
#         filename = f"imgs/{safe_i}_{safe_j}.png"
#         # plt.savefig(filename)
#         plt.close()

# %%
merged_df = pd.merge(df1, df2, on='depmap_id', how='inner')


merged_df = merged_df.rename(columns={
    "Omics Absolute CN Gene Public 24Q4 TP53": "tp53ex"
})
for i in cancerlist:
    moddf = merged_df[merged_df['lineage_1_x'] == i]
    # Drop rows where either value is missing
    plot_df = moddf.dropna(subset=["tp53ex", "TP53", "lineage_1_x"])
    safe_i = str(i).replace("/", "_").replace("\\", "_")
    plt.figure(figsize=(8,6))

    # Scatterplot colored by lineage
    sns.scatterplot(
        data=plot_df,
        x="tp53ex",
        y="TP53",
        hue="lineage_1_x",
        palette="tab10",
        alpha=0.8,
        s=50
    )

    # Labels and title
    plt.xlabel("tp53 expression")
    plt.ylabel("Dependency")
    plt.title("APC2 CN vs TP53 | Colored by Lineage")

    plt.tight_layout()
    plt.title(f"TP53 in {i}")
    plt.tight_layout()

    filename = f"imgs/{safe_i}.png"
    plt.savefig(filename)
    plt.close()
    plt.show()
# %%
merged_df = pd.merge(df1, df2, on='depmap_id', how='inner')


merged_df = merged_df.rename(columns={
    "Expression Public 25Q3 TP53": "tp53ex",
    "Expression Public 25Q3 CD274": "cd274ex"

})
for i in cancerlist:
    moddf = merged_df[merged_df['lineage_1_x'] == i]
    # Drop rows where either value is missing
    plot_df = moddf.dropna(subset=["tp53ex", "cd274ex", "lineage_1_x"])
    safe_i = str(i).replace("/", "_").replace("\\", "_")
    plt.figure(figsize=(8,6))

    # Scatterplot colored by lineage
    sns.scatterplot(
        data=plot_df,
        x="tp53ex",
        y="cd274ex",
        hue="lineage_1_x",
        palette="tab10",
        alpha=0.8,
        s=50
    )

    # Labels and title
    plt.xlabel("tp53 expression")
    plt.ylabel("Dependency")
    plt.title("APC2 CN vs TP53 | Colored by Lineage")

    plt.tight_layout()
    plt.title(f"TP53 in {i}")
    plt.tight_layout()

    # filename = f"imgs/{safe_i}.png"
    # plt.savefig(filename)
    # plt.close()
    plt.show()