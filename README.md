# DeepViscosity
DeepViscosity is an ensemble deep learning ANN model developed to predict high-concentration monoclonal antibody viscosity classes (Low <= 20 cP, High > 20 cP). The model utilized 30 spatial properties (descriptors) obtained from DeepSP surrogate model as features for training. It was trained based on 229 mAbs.

# How to use DeepViscosity to Predict Low or High Class Viscosity

## Option 1 - Google colab notebook
- Prepare your input file according to the format DeepViscosity_input.csv
- Run the notebook file DeepViscosity_predictor.ipynb
- DeepViscosity Classes (as well as the DeepSP spatial properties) for sequences inputed, will be populated and saved to a csv file - DeepViscosity_classes.csv.

## Option 2 - Linux environment 
- git clone https://github.com/Lailabcode/DeepViscosity.git
- cd DeepViscosity
- create an environment and install necessary package
	- conda create -n deepViscosity python=3.9.13
	- source activate deepViscosity
	- conda install -c bioconda anarci
	- pip install keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4 joblib dill

- Prepare your input file according to the format DeepViscosity_input.csv
- Run the python file deepviscosity_predictor.py
- DeepViscosity Classes (as well as the DeepSP spatial properties) for sequences inputed, will be obtained and saved to a csv file - DeepViscosity_classes.csv.

## 🔄 Update: Migration from ANARCI to ANARCII

AbDev has transitioned from **ANARCI** to **ANARCII** for antibody sequence numbering.

Install via:

```bash
pip install anarcii
```

### Why this change?
- pip installable
- Improved compatibility with modern Python environments  
- Simplified installation (no legacy HMMER dependency)  
- Active maintenance  

### Important Note

Due to differences in numbering logic and backend implementation, minor variations in IMGT residue assignments may occur.

These changes may propagate to:

- Descriptor calculations  
- Feature engineering steps  
- Downstream prediction outputs  

For strict reproducibility of earlier results, ensure the same numbering backend is used.

---


# Citation
> Lateefat A. Kalejaye, Jia-Min Chu, I-En Wu, Bismark Amofah, Amber Lee, Mark Hutchinson, Chacko Chakiath, Andrew Dippel, Gilad Kaplan, Melissa Damschroder, Valentin Stanev, Maryam Pouryahya, Mehdi Boroumand, Jenna Caldwell, Alison Hinton, Madison Kreitz, Mitali Shah, Austin Gallegos, Neil Mody and Pin-Kuang Lai. 
> *Accelerating high-concentration monoclonal antibody development with large-scale viscosity data and ensemble deep learning.*
> mAbs, 2025.
> [https://doi.org/10.1080/19420862.2025.2483944](https://www.tandfonline.com/doi/full/10.1080/19420862.2025.2483944)
