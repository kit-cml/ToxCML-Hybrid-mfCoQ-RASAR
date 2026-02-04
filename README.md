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

- Curates and standardizes SMILES-based toxicity datasets following QSAR-ready best practices 
  (validity checks, salt/solvent removal, parent structure selection, charge normalization, 
  tautomer handling, stereochemical consistency, duplicate resolution, and label harmonization).

- Computes multiple complementary molecular representations using RDKit and CDK: 
  MACCS keys, Morgan circular fingerprints, APF, RDKit fingerprints, and physicochemical descriptors.

- Trains **descriptor-specific QSAR models** (Random Forest, XGBoost, SVM) with 
  **10-fold scaffold-based cross-validation**, selects the best model per descriptor using the 
  composite score:

  Sm = (AUCm + BACCm) / 2

  and builds an Sm-weighted consensus QSAR.

- Implements **multi-fingerprint read-across** (k-NN similarity-weighted using Tanimoto on MACCS, 
  Morgan, APF, RDKit fingerprints) and builds an Sf-weighted consensus RA using:

  Sf = (AUCf + BACCf) / 2

- Constructs a hybrid **mfCoQ‑RASAR** ensemble that combines consensus QSAR and consensus RA 
  probabilities via a linear integration:

  P_mfCoQ-RASAR = w * P_QSAR,cons + (1 - w) * P_RA,cons

  where the global weight w is optimized by 10-fold scaffold-CV on the training set.

- Applies a **tiered applicability-domain (AD) framework** (QSAR AD, RA AD, and mfCoQ‑RASAR AD 
  as their intersection) to flag in-domain vs out-of-domain predictions.


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

# ToxCML: mfCoQ-RASAR Framework

This repository provides the complete implementation of the **mfCoQ-RASAR** framework, a hybrid toxicity prediction platform that integrates **consensus QSAR**, **similarity-weighted read-across**, and their optimized coupling under a rigorous scaffold-based validation scheme.

The framework is designed to deliver reliable, interpretable, and applicability-domain-aware predictions across multiple toxicity endpoints while maintaining broad chemical-space coverage.

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
    - Validates SMILES and removes unparsable or chemically implausible structures.  
    - Standardization: removes salts and solvents, selects parent structures, normalizes charges, applies harmonized cleanup rules, and treats tautomers and stereochemistry consistently.  
    - Identifies duplicates via canonical SMILES or InChIKey and removes records with conflicting labels.  
  - Outputs:  
    - MACCS (166-bit), Morgan (1024-bit), APF (1024-bit), RDKit fingerprints (2048-bit).  
    - 49 RDKit/CDK-based physicochemical descriptors.

- `Training_Consensus QSAR_Fingerprint_10foldCrossvalidation (1).ipynb`  
  - Trains fingerprint-based QSAR models (RF, XGB, SVM) on MACCS, Morgan, and APF fingerprints.  
  - Uses **10-fold Bemis–Murcko scaffold-based cross-validation** for hyperparameter tuning and performance estimation.  
  - For each descriptor–algorithm model $m$, computes $AUC_m$, $BACC_m$, and the composite score:

    $$
    S_m = \frac{AUC_m + BACC_m}{2}
    $$

  - Selects one best-performing model per descriptor (highest $S_m$) and saves the model together with its $S_m$.

- `Training_Consensus QSAR_PhysicochemicalProperties_10foldCrossvalidation (1).ipynb`  
  - Trains QSAR models on physicochemical descriptors (RF, XGB, SVM) using 10-fold scaffold-based cross-validation.  
  - Computes $S_m$ for each descriptor–algorithm combination and selects the best physicochemical QSAR model per endpoint.

- `Model Performance Evaluation 1_Consensus QSAR.ipynb`  
  - Builds the **consensus QSAR** model by aggregating the best MACCS, Morgan, APF, and physicochemical QSAR models using **$S_m$-based weights**:

    $$
    P_{QSAR}^{cons}(i) = \sum_m w_m \, P_m(i),
    \qquad
    w_m = \frac{S_m}{\sum_k S_k}
    $$

  - Evaluates consensus QSAR on strictly unseen test or external sets using AUC, BACC, ACC, SEN, SPE, and 95% bootstrap confidence intervals.

- `Model Performance Evaluation 2_Consensus Read-Across Evaluation.ipynb`  
  - Implements **k-nearest-neighbor similarity-weighted read-across** using MACCS, Morgan, APF, and RDKit fingerprints.  
  - Computes Tanimoto similarity, selects the $k$ nearest neighbors, and derives similarity-weighted toxicity probabilities.  
  - Provides baseline read-across performance per fingerprint and per endpoint.

- `Weight Optimization_Consensus Read Across.ipynb`  
  - **Required step for building consensus read-across and mfCoQ-RASAR.**  
  - Uses 10-fold scaffold-based cross-validation on the training set to compute $AUC_f$, $BACC_f$, and:

    $$
    S_f = \frac{AUC_f + BACC_f}{2}
    $$

    for each fingerprint type $f$.  
  - Derives **$S_f$-based weights** $v_f = S_f / \sum_g S_g$ and constructs the final **$S_f$-weighted consensus read-across**:

    $$
    P_{RA}^{cons}(q) = \sum_f v_f \, P_f(q)
    $$

  - These consensus read-across probabilities are used as the read-across component in the mfCoQ-RASAR model.

- `mfCoQ-RASAR_Evaluation_Framework.ipynb`  
  - Constructs and evaluates the **hybrid mfCoQ-RASAR** predictor:

    $$
    P_{\mathrm{mfCoQ\text{-}RASAR}}(i; w)
    = w \, P_{QSAR}^{cons}(i) + (1 - w) \, P_{RA}^{cons}(i)
    $$

  - Performs **10-fold Bemis–Murcko scaffold-based cross-validation** on the training set to optimize the global weight $w$:  
    - In each fold $f$, trains consensus QSAR and consensus read-across on 9 folds and generates $P_{QSAR}^{cons,f}$ and $P_{RA}^{cons,f}$ on the validation fold.  
    - For each candidate weight $w$, computes:

      $$
      S_{f,w} = \frac{AUC_{f,w} + BACC_{f,w}}{2}
      $$

    - Averages performance across folds:

      $$
      S_w = \frac{1}{K} \sum_f S_{f,w}
      $$

    - Selects the optimal coupling weight:

      $$
      w^* = \operatorname*{arg\,max}_w S_w
      $$

  - After identifying $w^*$, retrains consensus QSAR and consensus read-across on the full training set and applies the fixed $w^*$ to test or external sets, reporting AUC, BACC, ACC, SEN, SPE, and 95% bootstrap confidence intervals.

- `AD Analysis/` and `AD Analysis.ipynb`  
  - Contain pre-computed **applicability-domain (AD)** annotations (INSIDE or OUTSIDE) for QSAR, read-across, and mfCoQ-RASAR predictions.  
  - QSAR AD is defined using maximum Tanimoto similarity (Morgan, MACCS, APF) and leverage in physicochemical descriptor space, combined by majority voting.  
  - Read-across AD is defined based on the mean similarity to the $k$ nearest neighbors per fingerprint with thresholds derived from training-set distributions.  
  - mfCoQ-RASAR AD is defined as the intersection of QSAR AD and read-across AD.

- `SHAP/` and `SHAP Analysis (2).ipynb`  
  - SHAP-based explainable AI analysis for the top-performing descriptor-specific QSAR models per endpoint.  
  - Uses TreeExplainer for RF and XGB models and KernelExplainer for SVM models.  
  - Provides fragment contribution maps (Morgan-based) by projecting important fingerprint bits back onto molecular structures.

- `.gitattributes`  
  - Git configuration for line endings and handling of large or binary files.

---

## Workflow: reproducing the mfCoQ-RASAR framework

1. **Data curation and preprocessing**  
   Run `Molecular Descriptor Computation_Preprosesing data (1).ipynb` to generate QSAR-ready datasets and molecular descriptors.

2. **Descriptor-specific QSAR model development**  
   Train fingerprint-based and physicochemical QSAR models using scaffold-based cross-validation and compute $S_m$ scores.

3. **Consensus QSAR evaluation**  
   Build the $S_m$-weighted consensus QSAR model and evaluate it on unseen test or external datasets.

4. **Read-across development and weighting**  
   Perform multi-fingerprint read-across and derive the $S_f$-weighted consensus read-across model.

5. **Hybrid mfCoQ-RASAR construction**  
   Optimize the QSAR–read-across coupling weight $w$, retrain models on the full training set, and evaluate final performance.

6. **Applicability-domain analysis and explainability**  
   Analyze prediction reliability using applicability-domain definitions and interpret QSAR predictions using SHAP.


---

## How to cite

If you use this repository, the models, or any derived results in your work, please cite the corresponding ToxCML mfCoQ‑RASAR manuscript once published. Pending formal citation, you may reference it as:

> Nursyafi FS, Pramudito MA, Fuadah YN, Izza RN, Fauzan AL, Lim KM.  
> **ToxCML: A Hybrid mfCoQ‑RASAR-Based Platform Integrating Consensus QSAR and Read-Across for Comprehensive Multi-Endpoint Toxicity Assessment.** Manuscript in preparation.
