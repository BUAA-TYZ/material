# **ML-Guided Design of Dual-Dynamic Crosslinked Networks (DDCNs)**

Official implementation of the QSPR framework for:  
3D Printing of Hierarchical Dual-Dynamic Crosslinked Networks (DDCNs) Featuring Dynamic Covalent Bonds and Machine Learning-Optimized Coordination.

**Repository URL:** [https://github.com/BUAA-TYZ/material](https://github.com/BUAA-TYZ/material)

This repository provides the **Quantitative Structure–Property Relationship (QSPR)** framework, datasets, and computational workflows utilized to accelerate the discovery of optimal coordination crosslinkers for high-resolution DLP 3D printing.

## **🧪 1\. Problem & Solution**

### **The Challenge**

The development of next-generation dynamic polymers requires navigating an enormous combinatorial chemical space to identify ideal metal-ligand (M-L) coordination nodes. Traditional trial-and-error screening is **time-consuming**, **resource-intensive**, and inefficient when balancing conflicting properties such as mechanical strength, optical performance, and resin compatibility.

### **Our Solution: ML-Guided Screening**

To address this challenge, we developed a data-driven QSPR workflow powered by **Random Forest Regression (RFR)**. Unlike "black-box" approaches, this framework was specifically designed to:

* **Bypass Experimental Bottlenecks:** Predict material performance solely based on **9 theoretical molecular descriptors**, enabling rapid pre-screening without synthesis.  
* **Quantify Trade-offs:** Optimize a composite **Performance Index (PI)** that rigorously balances optical (Quantum Yield), mechanical (Bond Strength), and processing (Solubility) requirements.  
* **Elucidate Mechanisms:** Reveal the **"Ligand-Metal Synergism"** and **"Electronic-Electrostatic Driving Forces"** governing material properties.

## **🎯 2\. Methodology**

### **Target Definition: Performance Index (PI)**

To train the model, we constructed a ground-truth dataset where the target variable (PI) uses an **equal-weighting strategy** to balance conflicting engineering requirements:

PI \= 1/3×F \+ 1/3×S \+ 1/3×M

* F **(FluorescenceQuantum Yield):** Experimental fluorescence efficiency (In-situ probe).  
* S **(Bond Shortening):** Thermodynamic strength of the coordination node (Derived from **CCDC** crystal data).  
* M **(Compatibility):** Ligand solubility in acrylate resin (Calculated via **SwissADME**).

**Note:** This formula is used **ONLY** to generate labels for the training set. The machine learning model learns to predict this PI directly from theoretical physicochemical descriptors.

### **Feature Engineering**

The model utilizes 9 fundamental physicochemical descriptors. Feature importance analysis reveals a distinct **"Dual Electronic-Electrostatic Driving Force"**, where the top features account for the majority of the decision weight:

1. **Optical Gatekeeper:** **Antenna Effect (**A**)** ensures efficient ligand-to-metal energy transfer.  
2. **Thermodynamic Foundation:** **Ligand Electronegativity (**Xdonor**)** and **Ionic Potential (**IP**)** govern the electrostatic interaction strength of the coordination node, which is critical for mechanical robustness.  
3. **Quantum Identifier:** **f-electron count (**nf**)** defines the intrinsic emission color and energy levels.

## **📊 3\. Model Performance & Discovery**

The framework prioritizes **screening efficiency (ranking fidelity)** over absolute regression accuracy in low-performance regions.

### **Validation Metrics (Test Set)**

* **RMSE \= 0.043:** Extremely low absolute error (\< 5% of theoretical range), confirming the model can accurately map theoretical features to the composite PI.  
* R2 **\= 0.71:** Demonstrates robust predictive capability, effectively mitigating noise arising from complex fluorescence quenching mechanisms in non-emissive samples.

### **Screening Outcome**

Without prior experimental data, the computational screening successfully highlighted **Terbium-Acetylacetonate (Tb-ACAC)** as the optimal coordination node (**Rank 1**, Predicted PI: 0.805). This ML-guided discovery enabled the DDCN system to achieve high mechanical performance and excellent 3D printability.

Additionally, **Zinc-Acetylacetonate (Zn-ACAC)** (Predicted PI: 0.697) was identified as a structural control to decouple optical functionalities from mechanical reinforcement.

## **🛠️ 4\. Usage**

### **Prerequisites**

* Python 3.8+  
* scikit-learn  
* pandas, numpy  
* matplotlib, openpyxl

### **Quick Start**

1. **Clone the repository:**  
   `git clone https://github.com/BUAA-TYZ/material.git` `cd material`
2. **Install dependencies:**  
   `pip install \-r requirements.txt`
3. Run the pipeline (Train & Screen):  
   The main script trains the Random Forest model, evaluates it on the test set, and predicts scores for all candidates.  
   `python main.py \--model rf`
   * *Outputs:* Feature importance plots, predicted vs. actual plots, and the final ranked candidate list (outputs/predicted\_with\_all\_scores.xlsx).

## **📂 Repository Structure**

* data/: Contains the M-L library (material.xlsx) with physicochemical descriptors.  
* src/:  
  * models.py: Implementation of Random Forest Regressor configuration.  
  * pipeline.py: End-to-end training and evaluation workflow.  
  * data\_utils.py: Data preprocessing and stratification logic.  
* outputs/: Generated figures and prediction tables.

## **📄 Citation**

If you use this code or data in your research, please cite our paper:

(Insert your full paper citation here)