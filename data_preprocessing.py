import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

input_file = "/Users/alberthua/Desktop/ML_Project/cereal.csv"  # Full path to your file
data = pd.read_csv(input_file)

# Group manufacturers into 3 categories
data['mfr_group'] = data['mfr'].map({
    'K': 'Kelloggs',
    'G': 'General Mills'
}).fillna('Other')

# Drop original 'mfr' & 'name' cols
data = data.drop(columns=['mfr', 'name', 'type'])

# Separate features and target
X = data.drop(columns=['mfr_group'])
y = data['mfr_group']

# Handle missing values in features
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize numeric features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# Encode the target column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Combine features and encoded target
preprocessed_data = X_scaled.copy()
preprocessed_data['mfr_group'] = y_encoded

# Save label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Save processed data
output_file = "/Users/alberthua/Desktop/ML_Project/output/preprocessed_data.csv"
preprocessed_data.to_csv(output_file, index=False)

print(f"Preprocessed data saved as {output_file}")
