import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('default')

# Load data
df = pd.read_csv(r"D:\SoftNexis\Task_2\titanic.csv")

print("Data Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

print("\nSummary Stats:\n", df.describe(include='all'))

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

# -------------------------------
# Distribution
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df['Age'], bins=30, kde=True, ax=ax[0])
ax[0].set_title('Age Distribution')

sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax[1])
ax[1].set_title('Fare by Class')

plt.tight_layout()
plt.show()

# -------------------------------
# Categorical
# -------------------------------
plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

# -------------------------------
# Correlation
# -------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print("\nSurvival % by Class:\n")
print(pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100)

# -------------------------------
# Outliers
# -------------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title('Fare Outliers')
plt.show()

z_scores = np.abs(stats.zscore(df[['Fare']]))
outliers = df[(z_scores > 3).all(axis=1)]

print(f"\nOutliers Found: {len(outliers)}")

# -------------------------------
# Advanced visuals
# -------------------------------
g = sns.FacetGrid(df, col='Survived', row='Pclass', height=3)
g.map(sns.histplot, 'Age')
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Siblings/Spouses Aboard', 
                 'Parents/Children Aboard', 'Survived']], hue='Survived')
plt.show()

# -------------------------------
# Insights
# -------------------------------
print("\n--- KEY INSIGHTS ---")

print("\nSurvival Rate by Gender:")
print(pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100)

print("\nAverage Fare by Class:")
print(df.groupby('Pclass')['Fare'].mean())

print("\nAverage Age by Survival:")
print(df.groupby('Survived')['Age'].mean())

print("\n✅ EDA Completed Successfully!")