# electrospray-shap
SHAP Analysis of Electrospinning Parameters for Predicting Particle Size

# Electrospray Particle Size Prediction with SHAP Analysis

## Purpose of this Code
This code is designed to predict the diameter of particles generated through electrospray processes using a pre-trained XGBoost model. It also performs SHAP (SHapley Additive exPlanations) analysis to explain the model's predictions, providing insights into the importance and interaction of various features influencing the particle size.

## Libraries/Dependencies Needed
To run this code, you will need the following Python libraries:

- `shap`
- `pickle`
- `pandas`
- `matplotlib`
- `numpy`
- `xgboost`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install shap pickle5 pandas matplotlib numpy xgboost scikit-learn
```

## Instructions on How to Use

1. **Clone the Repository:**
   - Clone this repository to your local machine using:
     ```bash
     git clone https://github.com/AriadneD/electrospray-shap.git
     ```

2. **Navigate to the Project Directory:**
   - Change to the directory where the repository is cloned:
     ```bash
     cd <repository-directory>
     ```

3. **Ensure All Dependencies are Installed:**
   - Install the required Python libraries as listed in the dependencies section above.

4. **Prepare the Data:**
   - Ensure your data file is formatted in agreement with the example `Formatted_ML_EHD_Data.csv` and placed in the correct directory as expected by the script.

5. **Run the Code:**
   - Execute the Python script that performs SHAP analysis and generates visualizations:
     ```bash
     python ship.py
     ```
   - This script will load the model, process the data, and create various SHAP plots.

6. **View the Results:**
   - The generated visualizations will be saved in the `shap` directory within the project. You can open these PNG files to analyze the model's predictions.

## Variable and Meanings in the Data File

| Name          | Meaning                                                       | Unit    |
|---------------|---------------------------------------------------------------|---------|
| Polymer_Conc  | Concentration of the PLGA polymer in the solution             | w/v%    |
| Solvent       | Solvent used in electrospray solution                         | -       |
| FR            | Flow rate of the syringe pump                                 | mL/h    |
| Vol           | Voltage                                                       | kV      |
| Distance      | Collection distance, the distance between gauge and collector | mm      |
| Gauge         | Gauge size (needle size)                                      | mm      |
| Diameter_Mean | Diameter of the particle                                      | micrometers |

### Solvent Abbreviations:
- DMF 
- DMA 
- DCM 
- Acetone 
- ACN 
- TFE 
- DMC 
- Chloroform 
- Toluene 
- AceticAcid 
- EA 
- THF 
- MeOH 
- EtOH 
- Water

## Data Access
The models and the data used for training the models and running the SHAP analysis are sourced from the research conducted by Wang, F. et al. in the paper titled "Machine learning predicts electrospray particle size." 

You can access the data from their FigShare repository:
[https://doi.org/10.6084/m9.figshare.25040459.v1](https://doi.org/10.6084/m9.figshare.25040459.v1)

You can access the models from the GitHub repository:
[https://github.com/FrankWanger/ML_EHD/tree/main](https://github.com/FrankWanger/ML_EHD/tree/main)

## Original Paper Attribution
This code and analysis are based on the work presented in:

**Wang, F. et al. Machine learning predicts electrospray particle size.**  
Materials & Design 219, 110735 (2022).  
[https://doi.org/10.1016/j.matdes.2022.110735](https://doi.org/10.1016/j.matdes.2022.110735)
