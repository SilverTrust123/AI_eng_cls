import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_NAME = 'peach.csv'

try:
    df = pd.read_csv(FILE_NAME)
    print(f"loading successful: {FILE_NAME}")
except FileNotFoundError:
    print(f"path issue: {FILE_NAME}")
    exit() 
except Exception as e:
    print(f"reading file issue: {e}")
    exit()


# preprocessing
# (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Manure'], prefix='Manure', drop_first=True)
# none < low < medium < high (0, 1, 2, 3)
phosphorus_mapping = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
df.loc[:, 'Phosphorus_Encoded'] = df['Phosphorus'].map(phosphorus_mapping)
df = df.drop('Phosphorus', axis=1)

features = ['Phosphorus_Encoded', 'Manure_without']
X = df[features]

y = df['Growth (CM)']

# data split(80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
# model training using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
print("\nstart training the model...")
model.fit(X_train, y_train)
print("completed model training.")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nmodel evaluation results:")
print(f"MSE: {mse:.2f}")
print(f"R-squared: {r2:.4f}") # R-squared

importances = model.feature_importances_
feature_names = ['Phosphorus Level (0-3)', 'Manure (1=without)'] 
sorted_indices = np.argsort(importances)[::-1]

print("\nfeature importances:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]*100:.2f}%")

# for i in range(len(X.columns)):
#     for j in range(len(y.columns)):

#     new_conditions = pd.DataFrame({
#         'Phosphorus_Encoded': [i], 
#         'Manure_without': [j]
#     })

#     predicted_growth = model.predict(new_conditions)
#     print(f"\n new result for Ph")
#     print(f"predict 'Medium Phosphorus' and  'with Manure' height: {predicted_growth[0]:.2f} CM")

print("\n all possible predictions based on different conditions:")
print("=============================================")

# Phosphorus_Encoded: 0=none, 1=low, 2=medium, 3=high
phosphorus_levels = [0, 1, 2, 3]

# Manure_without: 0=with Manure, 1=without Manure
manure_conditions = [0, 1]
def decode_conditions(p_encoded, m_encoded):
    p_map = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'}
    m_map = {0: 'With Manure', 1: 'Without Manure'}
    return p_map[p_encoded], m_map[m_encoded]

results = []
p_labels = ['None', 'Low', 'Medium', 'High']
m_labels = ['With Manure', 'Without Manure']


for p in phosphorus_levels:
    for m in manure_conditions:
        new_conditions = pd.DataFrame({
            'Phosphorus_Encoded': [p], 
            'Manure_without': [m]
        })

        predicted_growth = model.predict(new_conditions)[0]
        p_label, m_label = decode_conditions(p, m)
        
        results.append({
            'Phosphorus_Encoded': p,
            'Manure_without': m,
            'Predicted_Growth': predicted_growth,
            'Phosphorus_Label': p_label,
            'Manure_Label': m_label
        })
        
        print("Condition prediction:")
        print(f"Condition: Phosphate fertilizer = {p_label}, Fertilizer = {m_label}")
        print(f" Predicted growth height: **{predicted_growth:.2f} CM**")
        print
        
results_df = pd.DataFrame(results)

print("\nstart generating visualizations")

heatmap_data = results_df.pivot_table(
    index='Manure_Label', 
    columns='Phosphorus_Label', 
    values='Predicted_Growth'
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu",
    linewidths=.5, 
    cbar_kws={'label': 'Predicted Growth (CM)'}
)
plt.title('Predicted Plant Growth by Fertilizer Conditions')
plt.xlabel('Phosphorus Level')
plt.ylabel('Manure Condition')
plt.show()

plt.figure(figsize=(10, 7))
sns.barplot(
    x='Phosphorus_Label', 
    y='Predicted_Growth', 
    hue='Manure_Label', 
    data=results_df, 
    palette='Set1'
)
plt.title('Predicted Plant Growth (CM) Comparison')
plt.xlabel('Phosphorus Level')
plt.ylabel('Predicted Growth (CM)')
plt.legend(title='Manure Condition')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 殘差圖實驗
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, bins=5, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', label='Zero Residual')
plt.title('Residual Distribution (Model Error)')
plt.xlabel('Residuals (Actual Growth - Predicted Growth)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 散佈圖實驗
plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Prediction (Y=X)')
plt.title(f'Model Accuracy: Predicted vs. Actual Growth (R2={r2:.4f})')
plt.xlabel('Actual Growth (CM) in Test Set')
plt.ylabel('Predicted Growth (CM) by Model')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show() 

print("\nVisualizations generated successfully.")