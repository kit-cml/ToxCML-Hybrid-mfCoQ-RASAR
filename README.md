# ToxCML: A Hybrid q‑RASAR Framework-Based Platform Integrating Consensus QSAR and Read-Across for Enhanced Comprehensive Multi-Endpoint Toxicity Assessment

**Authors**  
Fauzan Syarif Nursyafi¹, Muhammad Adnan Pramudito², Yunendah Nur Fuadah³, Rahmafatin Nurul Izza², Abdul Latif Fauzan⁵, and Ki Moo Lim¹˒⁴˒⁵*  

¹ Computational Medicine Lab, Department of Medical IT Convergence Engineering, Kumoh National Institute of Technology, Gumi, 39177, Republic of Korea.  
² Computational Medicine Lab, Department of IT Convergence Engineering, Kumoh National Institute of Technology, Gumi, 39177, Republic of Korea.  
³ Telecommunication Engineering Study Program, School of Electrical Engineering, Telkom University Main Campus, Bandung, Indonesia.  
⁴ Computational Medicine Lab, Department of Biomedical Engineering, Kumoh National Institute of Technology, Gumi, 39177, Republic of Korea.  
⁵ Meta Heart Co., Ltd, Gumi, 39253, Republic of Korea.  

*Corresponding author: [kmlim@kumoh.ac.kr](mailto:kmlim@kumoh.ac.kr)

ToxCML is a large-scale hybrid **q‑RASAR** framework-based platform that integrates consensus QSAR and consensus read-across into a weight-optimized workflow for multi-endpoint toxicity prediction. It is designed to provide chemically contextualized, applicability-domain–aware predictions that can support large-scale toxicity screening, hazard prioritization, and reduction of animal testing.

## Study overview

Conventional animal-based toxicity testing is time-consuming, expensive, and ethically challenging, motivating the development of computational in silico methods. In this project, ToxCML combines:
- Machine-learning–based consensus QSAR models built on multiple molecular representations (MACCS, Morgan, APF, RDKit fingerprints, and physicochemical descriptors) using Random Forest, XGBoost, and Support Vector Machines (SVM).
- Similarity-based consensus read-across models.
- Pre-computed tiered applicability-domain (AD) annotations for each compound and endpoint (INSIDE vs OUTSIDE).
- Integration of consensus QSAR and consensus read-across into a unified hybrid q‑RASAR framework.

The hybrid q‑RASAR framework predicts **18 toxicity endpoints** for **53,378 unique chemicals**, achieving strong performance on unseen test sets and external validation sets (AUC ≈ 0.86–0.99; ACC ≈ 0.75–0.98). Across endpoints, hybrid q‑RASAR consistently outperforms its individual components, with consensus QSAR remaining highly competitive and consensus read-across providing complementary discriminatory information.

## Toxicity endpoints

ToxCML covers 18 curated toxicity endpoints spanning acute systemic toxicity, organ-specific toxicity, and drug-induced safety liabilities relevant to regulatory and drug discovery contexts.

| No. | Endpoint | Brief definition |
|----:|----------|------------------|
| 1 | **AMES Mutagenicity** | Potential of a compound to induce gene mutations in bacterial test systems (e.g., *Salmonella typhimurium*), serving as an initial indicator of genotoxicity. |
| 2 | **Acute Dermal Toxicity** | Adverse systemic effects or mortality following single or short-term exposure via the skin, typically categorized using dermal LD₅₀ data. |
| 3 | **Acute Inhalation Toxicity** | Toxic responses after brief inhalation exposure to gases, vapors, or aerosols, reflecting systemic hazard via the respiratory route. |
| 4 | **Acute Oral Toxicity** | Systemic adverse effects or death after a single oral dose, classified according to LD₅₀-based regulatory categories. |
| 5 | **Carcinogenicity** | Ability of a compound to induce or promote tumor formation following repeated or long-term exposure. |
| 6 | **Cardiotoxicity** | Structural or functional damage to the heart, including effects on contractility, conduction, or rhythm. |
| 7 | **DILI (Drug-Induced Liver Injury)** | Liver damage caused by drugs, ranging from mild enzyme elevations to severe hepatic failure. |
| 8 | **Developmental Toxicity** | Adverse effects on embryo–fetal development, such as malformations, growth retardation, or prenatal death. |
| 9 | **Drug Induced Nephrotoxicity (DIN)** | Kidney injury caused by drugs, affecting glomerular or tubular function and renal clearance. |
| 10 | **Eye Irritation** | Reversible or irreversible ocular damage following direct contact with a test substance. |
| 11 | **FDA MDD** | Maximum Daily Dose–oriented systemic toxicity endpoint, where compounds are labeled according to toxicity status at FDA-relevant maximum daily dose (MDD) levels. |
| 12 | **Hematotoxicity** | Toxic effects on the blood and hematopoietic system, including changes in blood cell counts or bone marrow function. |
| 13 | **Hepatotoxicity** | Structural or functional liver injury (e.g., enzyme elevations, cholestasis, or hepatocellular damage) caused by chemicals or drugs. |
| 14 | **Mitochondrial Toxicity** | Impairment of mitochondrial function, such as disrupted oxidative phosphorylation or increased oxidative stress, leading to cellular dysfunction. |
| 15 | **Neurotoxicity** | Adverse effects on the central or peripheral nervous system, impacting neuronal structure or function. |
| 16 | **Respiratory Toxicity** | Toxic effects on the respiratory system, including airway inflammation, bronchoconstriction, or parenchymal lung damage. |
| 17 | **Skin Irritation** | Non-immunologic inflammatory response of the skin (e.g., erythema, edema) after short-term topical exposure. |
| 18 | **Skin Sensitization** | Immune-mediated allergic skin response (allergic contact dermatitis) following repeated exposure to a sensitizing chemical. |

## Repository structure

- `Dataset/`  
  Curated datasets (SMILES and binary toxicity labels) for the 18 toxicity endpoints used to develop and validate the models.  

- `AD Analysis/`  
  Pre-computed applicability-domain outputs for each endpoint, indicating whether each compound is classified as **INSIDE** or **OUTSIDE** the final q‑RASAR AD (no AD code is provided, only the annotated results).  

- `Molecular Descriptor Computation_Preprosesing data.ipynb`  
  Notebook for data preprocessing and molecular feature computation:
  - Input: SMILES-based datasets from `Dataset/`.  
  - Output: MACCS, Morgan, APF, RDKit fingerprints and physicochemical descriptors for each compound.  

- `Training_Consensus QSAR_Fingerprint_10foldCrossvalidation.ipynb`  
  Notebook for training fingerprint-based QSAR models (e.g., Random Forest, XGBoost, SVM using MACCS, Morgan, APF):  
  - Performs 10‑fold cross-validation per fingerprint–algorithm combination.  
  - Produces prediction probabilities that will be integrated into consensus QSAR.  

- `Training_Consensus QSAR_PhysicochemicalProperties_10foldCrossvalidation.ipynb`  
  Notebook for training descriptor-based QSAR models using physicochemical properties:  
  - Trains multiple algorithms with 10‑fold cross-validation.  
  - Generates consensus QSAR probabilities from descriptor-based models.  

- `Model Performance Evaluation 1_Consensus QSAR.ipynb`  
  Notebook for evaluating consensus QSAR on unseen test or external validation sets:  
  - Computes ACC, AUC, sensitivity, specificity, and confidence intervals.  

- `Model Performance Evaluation 2_Consensus Read-Across Evaluation.ipynb`  
  Notebook for building and evaluating consensus read-across models:  
  - Uses k‑nearest neighbor–type similarity (e.g., Tanimoto on MACCS, Morgan, APF, RDKit fingerprints).  
  - Aggregates fingerprint-specific read-across probabilities into a consensus read-across score.  

- `q-RASAR_Evaluation_Framework.ipynb`  
  Notebook for constructing and evaluating the **hybrid q‑RASAR** framework:  
  - Integrates consensus QSAR and consensus read-across probabilities via a linear weighting scheme.  
  - Optimizes the weight to jointly maximize predictive power (e.g., ACC and AUC).  

- `.gitattributes`  
  Git configuration for handling specific file types and line endings.

## Workflow: step-by-step

The typical workflow for reproducing the ToxCML hybrid q‑RASAR framework is:

### 1. Molecular descriptor and fingerprint computation

**Notebook:**  
`Molecular Descriptor Computation_Preprosesing data.ipynb`  

**Input:**  
- SMILES-based datasets from `Dataset/` (18 endpoints, 53,378 unique compounds).

**What this step does:**  
- Performs structure and label preprocessing (e.g., SMILES validation, duplicate removal, label harmonization).  
- Computes multiple molecular representations for each compound:
  - MACCS keys.  
  - Morgan circular fingerprints.  
  - Atom Pair (APF) fingerprints.  
  - RDKit fingerprints.  
  - Physicochemical descriptors.

**Output:**  
- Feature matrices (fingerprints + physicochemical descriptors) ready for model development.

### 2. Development of the consensus QSAR framework

This stage trains QSAR models separately on fingerprint-based features and physicochemical descriptors, then aggregates them into a consensus QSAR scheme.

#### 2.1 Training fingerprint-based QSAR models

**Notebook:**  
`Training_Consensus QSAR_Fingerprint_10foldCrossvalidation.ipynb`  

**What this step does:**  
- Trains QSAR models using MACCS, Morgan, and APF fingerprints with machine-learning algorithms (e.g., Random Forest, XGBoost, SVM).  
- Applies 10‑fold cross-validation to optimize and assess each fingerprint–algorithm combination.  
- Produces predicted probabilities for each endpoint and configuration.  

#### 2.2 Training physicochemical descriptor–based QSAR models

**Notebook:**  
`Training_Consensus QSAR_PhysicochemicalProperties_10foldCrossvalidation.ipynb`  

**What this step does:**  
- Trains QSAR models using physicochemical descriptors as input features.  
- Performs 10‑fold cross-validation to evaluate predictive stability and select high-performing models.  
- Outputs predicted probabilities for the descriptor-based QSAR models.
  
#### 2.3 Consensus QSAR model evaluation

**Notebook:**  
`Model Performance Evaluation 1_Consensus QSAR.ipynb`  

**What this step does:**  
- Integrates predicted probabilities from selected fingerprint- and descriptor-based models into a **consensus QSAR** probability (e.g., arithmetic mean across models/representations).  
- Evaluates consensus QSAR performance on unseen test sets or external validation sets.  
- Reports ACC, AUC, sensitivity, specificity, and 95% confidence intervals, showing that consensus QSAR is competitive across endpoints.

### 3. Development of the consensus read-across framework

**Notebook:**  
`Model Performance Evaluation 2_Consensus Read-Across Evaluation.ipynb`  

**What this step does:**  
- Implements read-across predictions using similarity between query and training compounds:
  - Computes similarity metrics (e.g., Tanimoto) on MACCS, Morgan, APF, and RDKit fingerprints.  
  - Selects k nearest neighbors and estimates read-across probabilities from their toxicity labels.  
- Aggregates fingerprint-specific read-across probabilities into a **consensus read-across** score.  
- Evaluates consensus read-across using ACC, AUC, and other metrics on unseen data, demonstrating its complementary predictive and discriminatory contribution relative to QSAR.

### 4. Hybrid q‑RASAR integration of consensus QSAR and read-across

**Notebook:**  
`q-RASAR_Evaluation_Framework.ipynb`  

**What this step does:**  
- Integrates the **consensus QSAR** probability and **consensus read-across** probability into a single hybrid **q‑RASAR** probability using a linear weighting scheme:
  - \( P_{\text{q‑RASAR}} = w \cdot P_{\text{QSAR,consensus}} + (1 - w) \cdot P_{\text{RA,consensus}} \).  
- Performs grid search over the weight \( w \) to maximize a joint performance score (e.g., combining AUC and ACC).  
- Evaluates hybrid q‑RASAR on unseen test and external validation sets, showing that the hybrid model consistently outperforms individual consensus QSAR and consensus read-across across the 18 endpoints.

## Applicability domain and chemical space analysis

**Folder:**  
`AD Analysis/`  

**What this step contains:**  
- Pre-computed applicability-domain annotations for all compounds and endpoints, indicating whether each compound is **INSIDE** or **OUTSIDE** the final q‑RASAR AD.  
- These AD labels can be used to filter predictions and interpret model reliability.

## How to cite

If you use this repository, the code, or any derived models in your work, please cite (on-going publication).
