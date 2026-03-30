# **Data-Driven Optimization of Metal-Ligand Coordination via Machine Learning**

**Official implementation of the machine learning framework for: "Rational Design of Hierarchical Dual-Dynamic Crosslinked Networks via Machine Learning-Optimized Coordination"**

🔗 **Repository URL:** [https://github.com/BUAA-TYZ/material](https://github.com/BUAA-TYZ/material)

This repository provides the fully self-contained machine learning framework, raw datasets, and computational workflows utilized to accelerate the discovery of optimal coordination crosslinkers for advanced digital manufacturing. It includes the complete source code for model training and a forward-prediction engine for unseen molecular candidates.

## **🧪 1\. Problem & Solution**

### **The Challenge**

Integrating robust hindered urea bonds (HUBs) with metal-ligand (M-L) coordination bonds poses a multi-objective optimization challenge, since the resulting network must exhibit mechanical stiffness, dynamic tunability, and chemical compatibility simultaneously. Conventional trial-and-error approaches are highly inefficient when exploring the vast combinatorial space of potential M-L crosslinkers.

### **Our Solution: Machine Learning-Guided Screening**

To address this challenge, we developed a data-driven workflow. This framework was specifically designed to:

* **Bypass Experimental Bottlenecks:** Predict the structural and thermodynamic performance of candidate materials by training the model on the underlying physicochemical relationships.  
* **Establish a Universal Metric:** We formulated a scalarized Performance Index (![][image1])，*where CN \= Coordination Number, S \= binding Strength, M \= Compatibility.*  
* **Enable Extrapolative Predictability:** Successfully mapped virtual ![][image2] scores to actual macroscopic mechanical trends via forward-prediction blind tests.

## **🧠 2\. The Machine Learning Workflow**

1. **Data Curation & Preprocessing:** A comprehensive dataset of metal-ligand complexes was curated. The model is informed by **8 fundamental physicochemical descriptors**, including intrinsic elemental properties (e.g., ionic radius, electronegativity) and structural parameters (e.g., bite angle, steric hindrance).  
2. **Model Training:** A robust **Random Forest Regressor (RFR)** was deployed to map the non-linear structure-property relationships and predict the composite ![][image2] score.  
3. **Forward Prediction (Blind Test):**  
   The trained RFR model was utilized to computationally screen uncharacterized molecular candidates (e.g., Zr-ACAC, Cu-ACAC). The virtual algorithmic ranking perfectly monotonically aligned with the actual macroscopic tensile strengths (Tb \> Zr \> Zn \> Cu), providing support for the applicability of our framework.

## **⚙️ 3\. Installation & Setup**

### **Prerequisites**

* Python 3.8+  
* pip (Python package installer)

### **Installation**

Clone the repository and install the required dependencies:

git clone \[https://github.com/BUAA-TYZ/material.git\](https://github.com/BUAA-TYZ/material.git)  
cd material  
pip install \-r requirements.txt

## **🚀 4\. Usage**

To train the Random Forest Regressor and generate predictions, simply run the main script from the root directory:

python main.py

**What this command does:**

1. Loads the training dataset (material.xlsx) and the blind test dataset (test.xlsx).  
2. Trains the Random Forest Regressor (RFR) on the training set.  
3. Evaluates model performance and feature importances.  
4. Performs forward prediction on the unseen candidates.  
5. Automatically generates and saves all corresponding figures and data tables into the outputs/ directory.

## **📁 5\. Repository Structure & Outputs**

This repository contains the complete source code and raw Excel datasets required to reproduce the study. The structure is organized as follows:

material/  
├── data/                               \# Contains the raw Excel datasets  
│   ├── material.xlsx                   \# Primary training dataset containing known M-L descriptors  
│   └── test.xlsx                       \# Unseen candidate dataset for the forward-prediction "blind test" (e.g., Zr, Cu)  
├── src/                                \# Core source code modules  
│   ├── config.py                       \# Configuration and hyperparameter settings  
│   ├── data\_utils.py                   \# Data loading and preprocessing scripts  
│   ├── metrics\_utils.py                \# Evaluation metrics calculation  
│   ├── models.py                       \# Machine learning model initializations  
│   ├── pipeline.py                     \# ML training and prediction pipeline  
│   └── plot\_utils.py                   \# Visualization and plotting functions  
├── main.py                             \# Main executable script to run the entire workflow  
├── requirements.txt                    \# Python dependencies  
└── outputs/                            \# Directory automatically generated after running main.py  
    ├── predicted\_with\_all\_scores.xlsx  \# Full training dataset with the newly predicted PI scores  
    ├── test\_predicted.xlsx             \# Predicted results specifically for the test/unseen candidates (Blind Test outputs)  
    ├── figures/                        \# High-resolution plots:  
    │   ├── pred\_vs\_actual.png          \# Parity plot evaluating model accuracy  
    │   ├── residual\_distribution.png   \# Distribution of prediction errors  
    │   ├── feature\_importance.png      \# RFR feature importance analysis  
    │   ├── cumulative\_importance.png   \# Cumulative feature importance curve  
    │   └── top10\_predicted\_vs\_actual.png \# Bar chart of the top candidate materials  
    └── tables/                         \# Raw CSV data corresponding to the generated figures  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKQAAAAYCAYAAAB0vVZPAAAFYElEQVR4Xu2aS4gdRRSG75CIis9AxmFeXXceMAwqEgZRwYX4QkFciGBwIgRcuBIkEF+rBHHhUhGEoAQRceFAdBGM4iIiSHQ2BhwC0cAYorMQGRCzUEn0/2+fujn33Orbr5m+IPVB0X3POVV96tTpqq6aabUikUgkEolEIpFIJBKJbAPtdnuvc+7fQSVJkudtPQLdirVlmZ2dvcnaNswO+HFG/PkB5T2UVd5DN0I5+v2cN87zH7JHrc3MzMwd2qYpMBaPif+Hcf8i7v/G/X5cL1nb7YYxNHH52dpoYP+dsT9gbbpAuUwjdHKXli8tLV1FORr7Qss10L8rNtdYXcOMTE9PPy6dfdoqp6am5kW3NjExsVvrJLh/if4jrfNAftnKmgRjsw/lrJXDrx9RLlh5U+DZGy594TPjg/jeBf0xie/rVt/DwsLCDTD6msZWR+SBWbpFlN8HOdMU8OES/WTiWZ0H+n/07KjkawwUyoVQXyVGH1p5UfDM962sLPTLThgiX4b8JSvPw+XMaEXASjHGuPnZ0uo98O9zJ3mE+1mr70HNKqtWR0QXfBjkb4n+uNU1iV8O0Nl3rE7DQbCzo8g3GChc72E7aG+v1jNG0C9pWRnqJiQ/EaR/oYQ8gJdw0srzYCysrCxMRMaNsZG49a2S0B1tyadSVh714JNKKlp2SkNnrIK4dKpm3dJv6FYiPv4GP261Oo0MwogRM1jHGUwM7LXSnxPagDEKJXJR6iakGvCnrA4+347LTivPQ2JRC7SxwrjJy5z1wqyOjY1dJ2OUv5K6dMllY30zAORfUsdvSasj8hAuk6Xf0K2CL4P4/6DVFYH91nWdbIhaapDxe8PfV6FuQnLQfaylr99ydmr1v1yFcVuTkBd5HR0dvV786skhyF5rpZOa3xjmr6Sqo0dU+VNkn1l7j3xXdepaXRZw+CuU8yXKM7YNi5MXqlVxcPCMo3qp8ZsfyA55GX6v+fsq1E1IAh/2+HjrkjVZ5OHqJ2RnZfE/JGbd73MmKWK7T3QnUS4neZOGSzcl7Fj+VGpwsjNHWbe6JvEDY+VFQd3NgKyz426nx2KM0bK1CTE3N3cLbMcD5eOAbJz2to0icJmGb5/SR5eza5XZte/ZKL8EZOOwv9m2EQLJda9OMPriXzx5qU9rnZPPIi8LojY061aXh7uyoVmxuiYRHzpLxyA424YC4gLLMWxPsF25PpEEPmdCuCtHR4WLbaMo6khu4OwL/WH7zEElCRwtheBsmKgdM+peRDnJe1l1XlE6tv2q/52Jk+OepMKmxMlxT+40vM3QBwbDyi0uMBNyo+ICM4wcZ3QGCOWbVsXPAU9e0gygu+GyCiL+ZR8wD8DVXLKd+Yzhb5SNycnJKeTEQ17OpKWflGv7INKh9SqbEqmbGawQcO4BlCeLFiSGs21Y4MPb9KU1IGmS9K8ae6zcvuUa2G9KH3OTPY+qCYkV7E6XrkChXXTnBGRY35DOrCzso8Trey13spJqWRAMxC5poPQbhkDdxrrJkGdHj/hy1h6Kc7CYjAjWfi33oN5m1oCiziMuXQFCx2GlqJqQfLaM0REtd+l37Tn2TcvL4GokJCaKh1H/mJY52VPYeLp0Je37LOqizoR6SlJg2UZg77P1WNrmIHkIcGnzf79mX86LX2/Mz8/fqA35t2roThn/D2kbD/vFQ2krL0vVhIRvf8D/q3E9KL7+6tK/SP2Ecr+1L4OrkJCoc1rHjUXpFtHPF3gvnzznrK0b4p84hwH/qWIRyfgJrqcQnLutwbCokZDdjQDun+Ugo6038XOHMquEq5CQkf8JSKKXrWzYICE/sLJIJBKJRCKRDP4DoAwjvTTDXKIAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAABX0lEQVR4Xu2TMUvEQBCFYyeoZUiTZBMRwoGdWGhtJ1hZCJYWtoedP8A/YKm/wcJCsLAQbISr7w/YWNgLKl5875xI9uW87WzMB0PYeW9udud2o6in509wzp0h6jnxjrjUOoL8zQx/rb4oTdO1PM/3i6K4pgHfA64ZWB+1ine1Frlt8009qB1yrb4fYBohJppH4aKzHatGyrJMmiaqdTDjnearqlpB/oF6kiRLqiN/YrX3qnnEcbxMY5Zle6pxfPYjn6oR5Me/1XrANKAR81xVDblba3KhGkH+FfE8q9bD2Q3TPE5xag2uVDMWTD9XwYM74E7sJE9NYP2IGKi/DXwb1mSuj8YdmCaIsWohUHPIJrwcqnk4u54YzbFqIdz3te+MuYOdouZ9Vy2EjWqkeQ97aDSGdyPY+wlPwP50NvlQLQTexTpr8d1UbQpfbnOCdvC2qFeB70XrEG+ILfX2/EO+ABgShr5FIdK1AAAAAElFTkSuQmCC>