# 📐 AV Safety Metric Validation

### **Status: Proof of Concept for Autonomous Safety & Validation**

This module is a custom Python implementation designed to bridge **Transportation Engineering constraints** with **Foundation Model reasoning** for autonomous vehicles. This work directly supports the validation and data curation challenges addressed by the **NVIDIA Alpamayo-R1** and **Sim2Val** frameworks.

---

## 🌟 The Problem & The Solution

**Problem:** Autonomous driving models need explicit, real-time metrics for geometric constraints to avoid safety failures (the "long tail" problem). Raw sensor logs (Quaternions) are unusable for direct causal reasoning.

**Solution:** This module automatically converts complex, raw kinematics data into a clean, single engineering metric—**Instantaneous Roadway Curvature ($\kappa$)**—which is a critical Causal Factor for a VLA model's safety decisions.

---

## 📊 Feature Validation: Curvature Profile

The plot below shows the output of the custom-coded feature tracking a sharp turn (the sustained dip in curvature) against the Ground Truth data for a sample clip from the NVIDIA Physical AI Dataset. The tight alignment validates the robustness of the custom kinematics code.



### Results Summary
* **The Event:** A sustained left turn, reaching approximately **-0.05 rad/m** (a safety-critical geometric constraint).
* **Proof:** The Calculated Feature (blue line) tracks the Ground Truth (red line) perfectly through the entire maneuver, validating the math.

---

## 🛠️ Project Details & Technical Stack

| Component | Description | Relevance |
| :--- | :--- | :--- |
| **Core Innovation** | **Curvature Feature Extraction:** Calculates $\kappa$ from raw Quaternions, Velocity vectors, and Timestamps using advanced kinematics (e.g., `numpy.unwrap`). | **Causal Reasoning:** Provides the explicit geometric input needed for the **Alpamayo-R1 Chain of Causation (CoC)** framework. |
| **Data Source** | NVIDIA Physical AI Autonomous Vehicles Dataset (Ego Motion Logs). | **Validation/Sim2Real:** Proves capability in handling industry-standard sensor data formats (Parquet). |
| **Primary Code** | `curvature_calculator.py` | Python (Pandas/NumPy) |

### Files in Repository:
1.  `curvature_calculator.py`: The Python script containing the `calculate_curvature_feature` function.
2.  `Curvature_Feature_Analysis_Final.csv`: The output data file containing the `curvature_feature` column.
3.  `Curvature_Profile_Plot.png`: The visualization of the validation result.

## ⚙️ Setup and Run Instructions

This project requires a standard Python environment optimized for data science.

1.  **Environment:** Requires Python 3.9+ and a standard Conda/Venv environment.
2.  **Dependencies:** `pandas`, `numpy`, `pyarrow`, `matplotlib`.
3.  **Data:** Requires a downloaded `.egomotion.parquet` file from the NVIDIA Physical AI Dataset (available on Hugging Face).
