# ToxCML: A Hybrid mfCoQ-RASAR Platform for Multi-Endpoint Toxicity Prediction

## Authors
Fauzan Syarif Nursyafi¹, Muhammad Adnan Pramudito², Yunendah Nur Fuadah³, Rahmafatin Nurul Izza², Abdul Latif Fauzan⁵, and Ki Moo Lim¹˒⁴˒⁵*

¹ Computational Medicine Lab, Department of Medical IT Convergence Engineering, Kumoh National Institute of Technology, Gumi 39177, Republic of Korea  
² Computational Medicine Lab, Department of IT Convergence Engineering, Kumoh National Institute of Technology, Gumi 39177, Republic of Korea  
³ Telecommunication Engineering Study Program, School of Electrical Engineering, Telkom University Main Campus, Bandung, Indonesia  
⁴ Computational Medicine Lab, Department of Biomedical Engineering, Kumoh National Institute of Technology, Gumi 39177, Republic of Korea  
⁵ Meta Heart Co., Ltd., Gumi 39253, Republic of Korea  

\*Corresponding author: kmlim@kumoh.ac.kr  

---

## Overview

ToxCML is a large-scale hybrid **mfCoQ-RASAR** (multi-feature Consensus quantitative Read-Across Structure–Activity Relationship) platform that explicitly integrates consensus QSAR and consensus read-across within a single, weight-optimized predictive framework for multi-endpoint toxicity prediction. The platform is designed to provide chemically contextualized and applicability-domain–aware predictions, supporting large-scale toxicity screening, hazard prioritization, and the reduction of animal testing.

---

## Study Overview

Conventional animal-based toxicity testing is costly, time-consuming, and ethically constrained, motivating the development of robust in silico alternatives. In this work, ToxCML:

- Curates and standardizes SMILES-based toxicity datasets following QSAR-ready best practices, including structural validity checks, salt and solvent removal, parent structure selection, charge normalization, tautomer handling, stereochemical consistency, duplicate resolution, and label harmonization.
- Computes multiple complementary molecular representations using RDKit and CDK, including MACCS keys, Morgan circular fingerprints, APF fingerprints, RDKit fingerprints, and physicochemical descriptors.
- Trains descriptor-specific QSAR models using Random Forest, XGBoost, and Support Vector Machines with **10-fold Bemis–Murcko scaffold-based cross-validation**, selects the best-performing model per descriptor using a composite score  
  \( S_m = (AUC_m + BACC_m) / 2 \), and constructs an **Sm-weighted consensus QSAR**.
- Implements **multi-fingerprint similarity-based read-across** using k-nearest neighbors with Tanimoto similarity (MACCS, Morgan, APF, RDKit) and builds an **Sf-weighted consensus read-across** using  
  \( S_f = (AUC_f + BACC_f) / 2 \).
- Constructs a hybrid **mfCoQ-RASAR** model by linearly integrating consensus QSAR and consensus read-across predictions:  
  \( P_{mfCoQ\text{-}RASAR} = w\,P_{QSAR}^{cons} + (1-w)\,P_{RA}^{cons} \),  
  where the global weight \( w \) is optimized via 10-fold scaffold-based cross-validation on the training set.
- Applies a **tiered applicability-domain (AD) framework**, consisting of QSAR AD, read-across AD, and their intersection as the mfCoQ-RASAR AD, to distinguish in-domain from out-of-domain predictions.

Across 18 toxicity endpoints and evaluations on strictly unseen test or external validation sets, mfCoQ-RASAR models achieve AUC values of approximately 0.86–0.99 and BACC values of approximately 0.73–0.98, consistently outperforming individual consensus QSAR and consensus read-across models while preserving interpretability and applicability-domain transparency.

---

## Toxicity Endpoints

ToxCML covers 18 curated toxicity endpoints spanning acute systemic toxicity, organ-specific toxicity, and drug-induced safety liabilities.

| No. | Endpoint | Brief definition |
|----:|----------|------------------|
| 1 | **AMES Mutagenicity** | Bacterial mutagenicity assay (e.g., *Salmonella typhimurium*) for genotoxicity screening. |
| 2 | **Acute Dermal Toxicity** | Systemic adverse effects following short-term dermal exposure (LD₅₀-based). |
| 3 | **Acute Inhalation Toxicity** | Toxicity after short-term inhalation of gases, vapors, or aerosols. |
| 4 | **Acute Oral Toxicity** | Adverse systemic effects following a single oral dose. |
| 5 | **Carcinogenicity** | Tumorigenic potential under chronic or repeated exposure. |
| 6 | **Cardiotoxicity** | Structural or functional cardiac injury affecting rhythm or contractility. |
| 7 | **DILI** | Drug-induced liver injury ranging from enzyme elevations to liver failure. |
| 8 | **Developmental Toxicity** | Adverse effects on embryo–fetal development. |
| 9 | **Drug-Induced Nephrotoxicity** | Toxic injury to renal structure or function. |
| 10 | **Eye Irritation** | Ocular damage following direct exposure. |
| 11 | **FDA MDD** | Toxicity at FDA-oriented maximum daily dose thresholds. |
| 12 | **Hematotoxicity** | Toxic effects on blood cells or hematopoietic tissues. |
| 13 | **Hepatotoxicity** | Structural or functional liver injury. |
| 14 | **Mitochondrial Toxicity** | Disruption of mitochondrial function or bioenergetics. |
| 15 | **Neurotoxicity** | Adverse effects on the central or peripheral nervous system. |
| 16 | **Respiratory Toxicity** | Toxic effects on the respiratory tract or lungs. |
| 17 | **Skin Irritation** | Non-immunologic inflammatory skin responses. |
| 18 | **Skin Sensitization** | Immune-mediated allergic contact dermatitis. |

---

## Repository Structure

The repository follows the operational workflow of the ToxCML framework:

- `README.md`  
  Main documentation describing the study overview, workflow, and usage instructions.

- `Dataset/`  
  QSAR-ready curated datasets for all 18 endpoints, including SMILES and binary outcome labels. Scaffold-based training/test splits are provided for selected endpoints.

- `Molecular Descriptor Computation_Preprocessing data.ipynb`  
  QSAR-ready data curation and molecular descriptor computation.

- `Training_Consensus_QSAR_Fingerprint_10foldCrossvalidation.ipynb`  
  Descriptor-specific QSAR model training using scaffold-based cross-validation.

- `Training_Consensus_QSAR_PhysicochemicalProperties_10foldCrossvalidation.ipynb`  
  QSAR modeling based on physicochemical descriptors.

- `Model_Performance_Evaluation_Consensus_QSAR.ipynb`  
  Construction and evaluation of the Sm-weighted consensus QSAR.

- `Model_Performance_Evaluation_Consensus_ReadAcross.ipynb`  
  Similarity-based read-across modeling and Sf-weighted consensus construction.

- `mfCoQ-RASAR_Evaluation_Framework.ipynb`  
  Hybrid mfCoQ-RASAR construction, weight optimization, and evaluation.

- `AD_Analysis/` and `AD_Analysis.ipynb`  
  Applicability-domain analysis for QSAR, read-across, and mfCoQ-RASAR models.

- `SHAP/` and `SHAP_Analysis.ipynb`  
  SHAP-based explainability analysis for descriptor-specific QSAR models.

---

## How to Cite

If you use this repository, the models, or any derived results in your work, please cite the corresponding ToxCML mfCoQ-RASAR manuscript once published. Until then, you may cite it as:

> Nursyafi FS, Pramudito MA, Fuadah YN, Izza RN, Fauzan AL, Lim KM.  
> **ToxCML: A hybrid mfCoQ-RASAR platform integrating consensus QSAR and read-across for multi-endpoint toxicity prediction.** Manuscript in preparation.
