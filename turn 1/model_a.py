import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_dd

# Step 1: Generate Dummy Data
np.random.seed(0)
data = pd.DataFrame({
    'player_id': range(1, 21),
    'training_hours': np.random.randint(10, 50, size=20),
    'rest_days': np.random.randint(1, 8, size=20),
    'injury_status': [0] * 10 + [1] * 10,  # 10 injured, 10 non-injured
    'time_to_event': np.random.randint(1, 100, size=20),  # Censored time to event (in weeks)
})

# Step 2: Convert data to required format for CoxPHFitter
T = data['time_to_event']
E = data['injury_status'].astype(bool)  # Event indicator (0=censored, 1=event)
data = data.drop(columns=['time_to_event', 'injury_status'])

# Step 3: Fit the Cox Proportional Hazards Model
cph = CoxPHFitter()
cph.fit(data, duration_col=T, event_col=E)

# Step 4: Summarize the model
print(cph.summary())

# Step 5: Make predictions (hazard ratios and survival curves)
new_data = pd.DataFrame({
    'training_hours': [20, 30],
    'rest_days': [3, 5]
})

hazard_ratios = cph.predict_survival_function(new_data).plot()

survival_curves = cph.predict_survival_function(new_data, times=[1, 10, 20, 30, 40, 50])
print(survival_curves)
