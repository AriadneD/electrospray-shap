# Import necessary libraries
import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained XGBoost model
with open('data/model.pkl', 'rb') as model_file:
    Model = pickle.load(model_file)

# Load the validation dataset
val_data = pd.read_csv('Formatted_ML_EHD_Data.csv')

# Define the solvent list
sol_list = ['DMF','DMA','DCM','Acetone','ACN','TFE','DMC','Chloroform','Toluene','AceticAcid','EA','THF','MeOH','EtOH','Water']

# Data Loading 
Polymer_wt = val_data.iloc[:,0]
val_data['Polymer_wt'] = Polymer_wt
Processing_Params = val_data.iloc[:,1:]

# Prepare raw features excluding 'Diameter_Mean'
X_raw = Processing_Params.drop(columns=['Diameter_Mean']).values.tolist()

# Extract fingerprints
fingerprint_file = Model['sol_feat']
FPs = []
for entry in X_raw:
    # Find featurization by index in the solvent list
    FP = fingerprint_file[sol_list.index(entry[0])]
    entry.pop(0)
    FPs.append(FP)

# Experiment Feature Standardizing
X_std = Model['feature'].transform(X_raw)
FPs = Model['fingerprint'].transform(FPs)

# Fingerprint Scaling
FPs = FPs/np.sqrt(len(FPs[0]))

# Concatenate Other Features with FPs
X_val = np.hstack([X_std, FPs])

# Define the feature names
feature_names = Processing_Params.drop(columns=['Diameter_Mean']).columns.tolist()
fingerprint_names = [f'FP_{i}' for i in range(FPs.shape[1])]
all_feature_names = feature_names + fingerprint_names

# Create a SHAP explainer using TreeExplainer for XGBoost
explainer = shap.TreeExplainer(Model['model'])

# Calculate SHAP values
shap_values = explainer.shap_values(X_val)

# Calculate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_val)

# Generate summary plot (ranking and value impact on model)
shap.summary_plot(shap_values, X_val, feature_names=all_feature_names, show=False)
plt.savefig('shap/shap_summary_plot.png')  # Save the summary plot
plt.show()

# Dependence plots for multiple features with line of best fit
features = ['Distance', 'Solvent', 'Vol', 'Gauge', 'FR']
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i, feature in enumerate(features):
    row, col = divmod(i, 3)
    shap.dependence_plot(feature, shap_values, X_val, feature_names=all_feature_names, ax=axes[row, col], show=False)
    
    # Extract SHAP values and feature values for fitting
    shap_vals = shap_values[:, all_feature_names.index(feature)]
    feature_vals = X_val[:, all_feature_names.index(feature)]
    
    # Handle NaNs or Infinites
    valid_mask = np.isfinite(feature_vals) & np.isfinite(shap_vals)
    feature_vals = feature_vals[valid_mask]
    shap_vals = shap_vals[valid_mask]

    # Calculate line of best fit
    try:
        z = np.polyfit(feature_vals, shap_vals, 1)
        p = np.poly1d(z)
        axes[row, col].plot(np.sort(feature_vals), p(np.sort(feature_vals)), "r--")
    except np.linalg.LinAlgError:
        print(f"Skipping line fit for {feature}")

plt.tight_layout()
plt.savefig('shap/shap_dependence_plots.png')
plt.show()

# Interaction plots for specified feature pairs with line of best fit
feature_pairs = [
    ('Distance', 'Solvent'), 
    ('Distance', 'Vol'), 
    ('Distance', 'Gauge'), 
    ('Solvent', 'Vol'), 
    ('Solvent', 'Gauge'), 
    ('Vol', 'Gauge')
]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i, (feature1, feature2) in enumerate(feature_pairs):
    row, col = divmod(i, 3)
    shap.dependence_plot((feature1, feature2), shap_interaction_values, X_val, feature_names=all_feature_names, ax=axes[row, col], show=False)
    
    # Extract SHAP interaction values and feature values for fitting
    shap_vals = shap_interaction_values[:, all_feature_names.index(feature1), all_feature_names.index(feature2)]
    feature_vals = X_val[:, all_feature_names.index(feature1)]
    
    # Handle NaNs or Infinites
    valid_mask = np.isfinite(feature_vals) & np.isfinite(shap_vals)
    feature_vals = feature_vals[valid_mask]
    shap_vals = shap_vals[valid_mask]

    # Calculate line of best fit
    try:
        z = np.polyfit(feature_vals, shap_vals, 1)
        p = np.poly1d(z)
        axes[row, col].plot(np.sort(feature_vals), p(np.sort(feature_vals)), "r--")
    except np.linalg.LinAlgError:
        print(f"Skipping line fit for {feature1} and {feature2}")

plt.tight_layout()
plt.savefig('shap/shap_interaction_plots.png')
plt.show()

# Generate decision plot (cumulative impact, shows prediction pathway)
all_feature_names.remove('Solvent') # Remove the feature variable
shap.decision_plot(explainer.expected_value, shap_values, X_val, feature_names=all_feature_names, show=False)
plt.savefig('shap/shap_decision_plot.png')  # Save the decision plot
plt.show()
