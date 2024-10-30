import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Dummy data generation
np.random.seed(0)
n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.randint(18, 45, size=n_samples),
    'BMI': np.random.randint(18, 35, size=n_samples),
    'Sports_Type': np.random.choice(['Football', 'Basketball', 'Cricket'], size=n_samples),
    'Hours_Played_Per_Week': np.random.randint(1, 20, size=n_samples),
    'Injury_History': np.random.choice(['Yes', 'No'], size=n_samples),
    'Maintenance_Need': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
    'Time_to_Maintenance': np.random.randint(1, 365, size=n_samples)
})

encoder = OneHotEncoder(sparse=False)
data_encoded = pd.DataFrame(encoder.fit_transform(data[['Sports_Type', 'Injury_History']]))
data_encoded.columns = encoder.get_feature_names(['Sports_Type', 'Injury_History'])
data_final = pd.concat([data[['Age', 'BMI', 'Hours_Played_Per_Week', 'Time_to_Maintenance']], data_encoded], axis=1)

from lifelines import CoxPHFitter

# Fit the Cox model
C = CoxPHFitter()
C.fit(data_final, duration_col='Time_to_Maintenance', event_col='Maintenance_Need')

# Summary of the model
print(C.summary())

# Print the coefficients
print(C.params_)

# Predict survival function for 1 year and 2 years
t = np.array([1, 2])
survival_func = C.predict_survival_function(data_final, t)
survival_func.head()
