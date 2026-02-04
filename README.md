# ToxCML: A Hybrid mfCoQ‑RASAR-Based Platform Integrating Consensus QSAR and Read-Across for Comprehensive Multi-Endpoint Toxicity Assessment

**Authors**  
Fauzan Syarif Nursyafi¹, Muhammad Adnan Pramudito², Yunendah Nur Fuadah³, Rahmafatin Nurul Izza², Abdul Latif Fauzan⁵, and Ki Moo Lim¹˒⁴˒⁵*  

¹ Computational Medicine Lab, Department of Medical IT Convergence Engineering, Kumoh National Institute of Technology, Gumi, 39177, Republic of Korea  
² Computational Medicine Lab, Department of IT Convergence Engineering, Kumoh National Institute of Technology, Gumi, 39177, Republic of Korea  
³ Telecommunication Engineering Study Program, School of Electrical Engineering, Telkom University Main Campus, Bandung, Indonesia  
⁴ Computational Medicine Lab, Department of Biomedical Engineering, Kumoh National Institute of Technology, Gumi, 39177, Republic of Korea  
⁵ Meta Heart Co., Ltd, Gumi, 39253, Republic of Korea  

\*Corresponding author: kmlim@kumoh.ac.kr  

ToxCML is a large-scale hybrid **mfCoQ‑RASAR** (multi-feature Consensus quantitative Read-Across Structure–Activity Relationship) platform that explicitly integrates consensus QSAR and consensus read-across into a **weight-optimized hybrid predictor** for multi-endpoint toxicity assessment. The framework is designed to provide chemically contextualized, applicability-domain–aware predictions to support large-scale toxicity screening, hazard prioritization, and reduction of animal testing.

---
## Framework overview

![mfCoQ‑RASAR framework](Figure1_mfCoQ-RASAR%20Framework.png)

## ToxCML web platform

![ToxCML web design](Web_ToxCML%20Design.png)


## Study overview

Conventional animal-based toxicity testing is costly, time-consuming, and ethically constrained, motivating robust in silico alternatives. In this work, ToxCML:

- Curates and standardizes SMILES-based toxicity datasets following QSAR-ready best practices (validity checks, salt/solvent removal, parent structure selection, charge normalization, tautomer handling, stereochemical consistency, duplicate resolution, and label harmonization).  
- Computes multiple complementary molecular representations using RDKit and CDK: MACCS keys, Morgan circular fingerprints, APF, RDKit fingerprints, and physicochemical descriptors.  
- Trains **descriptor-specific QSAR models** (Random Forest, XGBoost, SVM) with **10-fold scaffold-based cross-validation**, selects the best model per descriptor using a composite score  
  \(S_m = (AUC_m + BACC_m)/2\), and builds an **Sm-weighted consensus QSAR**.  
- Implements **multi-fingerprint read-across** (k-NN similarity-weighted using Tanimoto on MACCS, Morgan, APF, RDKit fingerprints) and builds an **Sf-weighted consensus RA** using  
  \(S_f = (AUC_f + BACC_f)/2\).  
- Constructs a hybrid **mfCoQ‑RASAR** ensemble that combines consensus QSAR and consensus RA probabilities via a linear integration:  
  \(P_{mfCoQ\text{-}RASAR} = w\,P_{QSAR}^{cons} + (1-w)\,P_{RA}^{cons}\), with **\(w\)** optimized by 10-fold scaffold-CV on the training set.  
- Applies a **tiered applicability-domain (AD) framework** (QSAR AD, RA AD, and mfCoQ‑RASAR AD as their intersection) to flag in-domain vs out-of-domain predictions.  

Across 18 toxicity endpoints and evaluation on unseen test or external validation sets, mfCoQ‑RASAR models achieve AUC ≈ 0.86–0.99 and BACC ≈ 0.73–0.98, consistently outperforming individual consensus QSAR and consensus read-across while preserving interpretability and AD transparency.

---

## Toxicity endpoints

ToxCML covers 18 curated toxicity endpoints spanning acute systemic toxicity, organ-specific toxicity, and drug-induced safety liabilities.

| No. | Endpoint | Brief definition |
|----:|----------|------------------|
| 1 | **AMES Mutagenicity** | Bacterial mutagenicity (e.g., *Salmonella typhimurium*), used as a screening assay for genotoxic potential. |
| 2 | **Acute Dermal Toxicity** | Systemic adverse effects or mortality after single/short-term dermal exposure, typically LD₅₀-based. |
| 3 | **Acute Inhalation Toxicity** | Systemic toxicity following short-term inhalation exposure to gases, vapors, or aerosols. |
| 4 | **Acute Oral Toxicity** | Systemic effects or death after a single oral dose, often categorized by LD₅₀ classes. |
| 5 | **Carcinogenicity** | Tumorigenic potential under repeated or chronic exposure. |
| 6 | **Cardiotoxicity** | Structural or functional cardiac injury (contractility, conduction, rhythm). |
| 7 | **DILI (Drug-Induced Liver Injury)** | Drug-related liver damage ranging from enzyme elevations to acute liver failure. |
| 8 | **Developmental Toxicity** | Adverse embryo–fetal outcomes (malformations, growth retardation, prenatal death). |
| 9 | **Drug-Induced Nephrotoxicity (DIN)** | Renal injury affecting glomerular/tubular function or clearance. |
| 10 | **Eye Irritation** | Reversible/irreversible ocular damage following direct exposure. |
| 11 | **FDA MDD** | Systemic toxicity at FDA-oriented maximum daily dose thresholds. |
| 12 | **Hematotoxicity** | Toxicity to blood and the hematopoietic system, including altered cell counts or marrow function. |
| 13 | **Hepatotoxicity** | Structural or functional liver injury (e.g., enzyme elevations, cholestasis, hepatocellular damage). |
| 14 | **Mitochondrial Toxicity** | Impaired mitochondrial function (e.g., oxidative phosphorylation disruption, oxidative stress). |
| 15 | **Neurotoxicity** | Adverse effects on CNS or PNS, impacting neuronal structure or function. |
| 16 | **Respiratory Toxicity** | Toxic effects in the respiratory tract (airway inflammation, bronchoconstriction, lung parenchymal damage). |
| 17 | **Skin Irritation** | Non-immunologic inflammatory skin response (erythema, edema) after topical exposure. |
| 18 | **Skin Sensitization** | Immune-mediated allergic contact dermatitis after repeated exposure. |

---

## Repository structure

This repository mirrors your local ToxCML layout:

- `README.md`  
  Main documentation (study overview, workflow, and usage).

- `Dataset/`  
  - QSAR-ready curated datasets for the 18 endpoints (SMILES + binary `Outcome` labels) after structural and label curation.  
  - For some endpoints, scaffold-based train/test splits are already provided.

- `Molecular Descriptor Computation_Preprosesing data (1).ipynb`  
  - QSAR-ready curation and descriptor computation:  
    - Validates SMILES and removes unparsable/chemically implausible structures.  
    - Standardization: removes salts/solvents, selects parent structures, normalizes charges, applies harmonized cleanup rules, and treats tautomers/stereochemistry consistently.  
    - Identifies duplicates via canonical SMILES/InChIKey and removes records with conflicting labels.  
  - Outputs:  
    - MACCS (166-bit), Morgan (1024-bit), APF (1024-bit), RDKit fingerprints (2048-bit).  
    - 49 RDKit/CDK-based physicochemical descriptors.

- `Training_Consensus QSAR_Fingerprint_10foldCrossvalidation (1).ipynb`  
  - Trains fingerprint-based QSAR models (RF, XGB, SVM) on MACCS, Morgan, APF.  
  - Uses **10-fold Bemis–Murcko scaffold-based CV** for hyperparameter tuning and performance estimation.  
  - For each descriptor–algorithm model \(m\), computes \(AUC_m\), \(BACC_m\), and the composite score  
    \(S_m = (AUC_m + BACC_m)/2\).  
  - Selects one best model per descriptor (highest \(S_m\)) and saves the model and its \(S_m\).

- `Training_Consensus QSAR_PhysicochemicalProperties_10foldCrossvalidation (1).ipynb`  
  - Trains QSAR models on physicochemical descriptors (RF, XGB, SVM) with 10-fold scaffold-CV.  
  - Computes and stores \(S_m\) for each descriptor–algorithm combination and selects the best physicochemical QSAR model per endpoint.

- `Model Performance Evaluation 1_Consensus QSAR.ipynb`  
  - Builds **consensus QSAR** by aggregating the best MACCS, Morgan, APF, and physicochemical models using **Sm-based weights**:  
    \[
    P_{QSAR}^{cons}(i) = \sum_m w_m P_m(i), \quad w_m = \frac{S_m}{\sum_k S_k}.
    \]  
  - Evaluates consensus QSAR on strictly unseen test/external sets using AUC, BACC, ACC, SEN, SPE, and 95% bootstrap confidence intervals.

- `Model Performance Evaluation 2_Consensus Read-Across Evaluation.ipynb`  
  - Implements **k-NN similarity-weighted read-across** using MACCS, Morgan, APF, and RDKit fingerprints:  
    - Computes Tanimoto similarity, selects k nearest neighbors, and derives similarity-weighted toxicity probabilities.  
  - Provides baseline RA performance per fingerprint and per endpoint.

- `Weight Optimization_Consensus Read Across.ipynb`  
  - **Required step for building consensus RA and mfCoQ‑RASAR.**  
  - Uses 10-fold scaffold-CV on the training set to compute \(AUC_f\), \(BACC_f\), and  
    \(S_f = (AUC_f + BACC_f)/2\) for each fingerprint type \(f\).  
  - Derives **Sf-based weights** \(v_f = S_f / \sum_g S_g\) and builds the final **Sf-weighted consensus RA**:  
    \[
    P_{RA}^{cons}(q) = \sum_f v_f P_f(q).
    \]  
  - These consensus RA probabilities are then used as the RA component in the mfCoQ‑RASAR integration.

- `mfCoQ-RASAR_Evaluation_Framework.ipynb`  
  - Constructs and evaluates the **hybrid mfCoQ‑RASAR** predictor:  
    \[
    P_{mfCoQ\text{-}RASAR}(i; w) = w\,P_{QSAR}^{cons}(i) + (1-w)\,P_{RA}^{cons}(i).
    \]  
  - Performs **10-fold Bemis–Murcko scaffold-CV on the training set** to optimize the global weight \(w\):  
    - In each fold \(f\), trains consensus QSAR and consensus RA on 9 folds and generates \(P_{QSAR}^{cons,f}\), \(P_{RA}^{cons,f}\) on the validation fold.  
    - For each candidate \(w\), computes  
      \(P_{mfCoQ\text{-}RASAR}^{(f,w)}\), then \(AUC_{f,w}\), \(BACC_{f,w}\), and  
      \(S_{f,w} = (AUC_{f,w} + BACC_{f,w})/2\).  
    - Averages across folds:  
      \(S_w = \frac{1}{K} \sum_f S_{f,w}\) and selects  
      \(w^* = \arg\max_w S_w\).  
  - After obtaining \(w^*\), retrains consensus QSAR and consensus RA on the full training set and applies the fixed \(w^*\) to test/external sets, reporting performance (AUC, BACC, ACC, SEN, SPE, 95% CI).

- `AD Analysis/` and `AD Analysis.ipynb`  
  - Contain **pre-computed applicability-domain annotations** (INSIDE/OUTSIDE) for QSAR, RA, and mfCoQ‑RASAR per endpoint.  
  - The notebook describes:  
    - QSAR AD: maximum Tanimoto similarity (Morgan, MACCS, APF) and leverage in physicochemical descriptor space, combined via majority voting.  
    - RA AD: mean similarity to k neighbors per fingerprint and cut-offs derived from the training set distribution.  
    - mfCoQ‑RASAR AD: the intersection of QSAR AD and RA AD.

- `SHAP/` and `SHAP Analysis (2).ipynb`  
  - SHAP-based explainable AI analysis for top-performing descriptor-specific QSAR models per endpoint.  
  - Uses TreeExplainer for RF/XGB and KernelExplainer for SVM.  
  - Provides fragment contribution maps (Morgan-based) projecting important fingerprint bits back onto molecular structures.

- `.gitattributes`  
  - Git configuration for line endings and handling of large/binary files.

---

## Workflow: reproducing the mfCoQ‑RASAR framework

1. **Data curation and preprocessing**  
   - Run `Molecular Descriptor Computation_Preprosesing data (1).ipynb`.  
   - Input: raw endpoint datasets in `Dataset/`.  
   - Output: QSAR-ready datasets plus all fingerprints and physicochemical descriptors.

2. **Descriptor-specific QSAR model development**  
   - Run `Training_Consensus QSAR_Fingerprint_10foldCrossvalidation (1).ipynb` for MACCS/Morgan/APF models (RF/XGB/SVM, scaffold-CV, compute \(S_m\)).  
   - Run `Training_Consensus QSAR_PhysicochemicalProperties_10foldCrossvalidation (1).ipynb` for physicochemical-based models (scaffold-CV, \(S_m\)).

3. **Consensus QSAR evaluation**  
   - Use `Model Performance Evaluation 1_Consensus QSAR.ipynb` to build Sm-weighted consensus QSAR and evaluate on test/external sets with AUC, BACC, ACC, SEN, SPE, and 95% confidence intervals.

4. **Read-across development and Sf-based weighting (required)**  
   - Run `Model Performance Evaluation 2_Consensus Read-Across Evaluation.ipynb` to implement multi-fingerprint k-NN RA and obtain per-fingerprint RA probabilities.  
   - Run `Weight Optimization_Consensus Read Across.ipynb` to compute \(S_f\) via 10-fold scaffold-CV and build the **Sf-weighted consensus RA** used in mfCoQ‑RASAR.

5. **Hybrid mfCoQ‑RASAR construction and evaluation**  
   - Run `mfCoQ-RASAR_Evaluation_Framework.ipynb` to:  
     - Optimize \(w\) using 10-fold scaffold-CV on the training set.  
     - Retrain consensus QSAR and consensus RA on the full training set.  
     - Evaluate mfCoQ‑RASAR with \(w^*\) on test/external sets with full metrics and 95% bootstrap confidence intervals.

6. **Applicability domain and explainability**  
   - Use `AD Analysis.ipynb` and `AD Analysis/` to analyze AD and filter predictions by in-domain vs out-of-domain status.  
   - Use `SHAP Analysis (2).ipynb` and `SHAP/` to interpret QSAR predictions at descriptor and fragment level.

---

## How to cite

If you use this repository, the models, or any derived results in your work, please cite the corresponding ToxCML mfCoQ‑RASAR manuscript once published. Pending formal citation, you may reference it as:

> Nursyafi FS, Pramudito MA, Fuadah YN, Izza RN, Fauzan AL, Lim KM.  
> **ToxCML: A Hybrid mfCoQ‑RASAR-Based Platform Integrating Consensus QSAR and Read-Across for Comprehensive Multi-Endpoint Toxicity Assessment.** Manuscript in preparation.
