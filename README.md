# DepMap Multi-Omics Analysis of Cancer Cell lines

## NCBS 2025

A lightweight playground for modeling gene dependency using DepMap multi-omics data. The goal is to explore how copy-number, expression, and damaging mutations relate to RNAi-derived gene dependency across selected cancer types.

### Repository Structure

- `Code/`: analysis scripts and notebooks
- `Data/`: input CSVs downloaded from DepMap
- `Results/`: figures, tables, and model artifacts
- `tmp/`: scratch space and intermediate files

## Data

- Predictors: `depmap_export_2025-12-01 13_24_31.776975_subsetted.csv`  
  Source: https://depmap.org/portal/data_page/?tab=customDownloads

### Custom Filters

Cancer types: Breast, Ovary, Esophagus/stomach, CNS/Brain, Skin

Genes: 307 common cancer genes (TP53 PIK3CA KRAS BRAF NRAS HRAS PTEN APC CTNNB1 RB1 CDKN2A ARID1A IDH1 IDH2 TET2 DNMT3A JAK2 NPM1 FLT3 KIT PDGFRA EGFR ERBB2 ERBB3 ERBB4 MET ALK ROS1 RET NTRK1 NTRK2 NTRK3 FGFR1 FGFR2 FGFR3 FGFR4 PDGFRB IGF1R AXL DDR2 EPHA2 EPHB4 VEGFA KDR RAF1 MAP2K1 MAP2K2 MAPK1 MAPK3 PIK3R1 PIK3CB PIK3CD PIK3CG AKT1 AKT2 AKT3 MTOR TSC1 TSC2 RPTOR RICTOR STK11 CDK4 CDK6 CDK2 CCND1 CCND2 CCND3 CCNE1 CDKN1A CDKN1B CDKN2B CDKN2C RBM10 MDM2 MDM4 CCNB1 WEE1 CHEK1 CHEK2 ATM ATR BUB1B AURKA AURKB PLK1 BRCA1 BRCA2 PALB2 RAD51 RAD51B RAD51C RAD51D RAD52 RAD54L FANCA FANCC FANCD2 FANCE FANCF FANCG FANCL MSH2 MSH6 MSH3 MLH1 PMS2 PMS1 POLE POLD1 ERCC1 ERCC2 XPA XPC LIG1 LIG3 PARP1 PARP2 BLM WRN MUTYH BCL2 BCL2L1 BCL2L11 BAX BAK1 MCL1 CASP8 CASP9 FAS FASLG CFLAR MYC MYCN MYCL NFE2L2 HIF1A EPAS1 STAT3 STAT5B NOTCH1 NOTCH2 NOTCH3 CREBBP EP300 RELA NFKB1 NFKB2 ESR1 ESR2 PGR AR GATA3 FOXA1 SOX2 SOX9 RUNX1 RUNX1T1 TCF3 TCF4 TCF7L2 ARID1B ARID2 SMARCA4 SMARCB1 SMARCD1 SMARCE1 PBRM1 BRD4 KMT2A KMT2C KMT2D KDM6A KDM5A KDM5C EZH2 SUZ12 EED DNMT1 TET1 TET3 ASXL1 ASXL2 BCOR BCORL1 NSD1 NSD2 SETD2 AXIN1 AXIN2 DVL2 DVL3 LRP5 LRP6 APC2 GSK3B LEF1 SMO PTCH1 PTCH2 GLI1 GLI2 GLI3 SUFU BMP2 BMP4 TGFB1 TGFBR1 TGFBR2 ACVR1 ACVR2A SMAD2 SMAD3 SMAD4 SMAD7 CDH1 CDH2 CDH11 EPCAM VIM SNAI1 SNAI2 TWIST1 ZEB1 ZEB2 MMP2 MMP9 ITGA6 ITGB3 ITGB1 COL1A1 COL2A1 COL3A1 SELP CXCR4 CXCL12 B2M HLA-A HLA-B HLA-C HLA-DPA1 HLA-DPB1 HLA-DQA1 HLA-DQB1 HLA-DRB1 CD274 PDCD1 PDCD1LG2 CTLA4 LAG3 TIGIT HAVCR2 JAK1 JAK3 STAT1 IFNGR1 IFNGR2 CALR MPL CBL CSF3R SF3B1 SRSF2 U2AF1 ZRSR2 SETBP1 ETNK1 GATA1 GATA2 IKZF1 IKZF3 PAX5 BCL6 BCL7A BIRC3 CARD11 CD79A CD79B ETV6 JARID2 PHF6 PRDM1 TCL1A ETV1 ETV4 ETV5 FLI1 ERG EWSR1 FUS TFE3 TFEB TCF12 NUP214 NUP98 BCR ABL1 ABL2 PDGFRL PRKACA PRKACB SS18 CIC DUX4)

### Features (per gene)

- Absolute copy-number (CN)
- Expression
- Damaging mutations

- Outcome: `RNAi_(Achilles+DRIVE+Marcotte,_DEMETER2)_subsetted-4.csv`  
  Source: https://depmap.org/portal/data_page/?tab=customDownloads

The Outcome file contains the RNA interference (silencing) results for cell-line experiments across five genes (BRCA1 BRCA2 TP53 BRIP1 PARP1). Choose one, and use the predictors to model the cell line dependency on that gene. (the lower value, the more dependent. >-0.5 generally means the cell line dies when the gene is silenced).

## Research Questions (Ideas)

- Which cancer types show the strongest dependency for each target gene?
- Do CN, expression, and damaging mutations contribute complementary or redundant signal?
- Are predictors from pathway partners (e.g., HR genes for BRCA targets) most informative?
- How stable are feature rankings across CV folds and random seeds?
- Can a model trained on one cancer type transfer to another (domain shift)?

## Analysis Plan

- Make a script that loads data.

  - Make sure you understand the structure of both predictors and outcome, then join it to a single table based on the depmap ID.

- How many cell lines are there per cancer type?

  - What does this mean for our modelling of the data?
  - Make summary statistics

- Choose a gene from the outcome table and identify the best predictors of dependency on that gene

  - Select a model or tool
  - Visualize the output

- Train a predictive model of gene dependency

  - Select a model
  - Consider the difference between mutation, expression, and CN data.
  - Select tools for optimizing and quantifying efficacy. Cross validation? Make sure to leave out test data to evaluate your final model.

- If time, try unsupervised learning to identify patterns in the data.
# DepMap_NCBSproject
