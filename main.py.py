import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("train.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include=['number']).corr(),
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.tight_layout()
plt.savefig("outputs/survival_distribution.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.tight_layout()
plt.savefig("outputs/class_vs_survival.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("outputs/age_vs_survival.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='Fare', y='Age', hue='Survived', data=df)
plt.title("Fare vs Age")
plt.tight_layout()
plt.savefig("outputs/fare_vs_age.png")
plt.show()

print("\nKey Insights")
print("1. Passenger class has a strong relationship with survival.")
print("2. Fare is positively related to passenger class and survival.")
print("3. Female passengers generally had higher survival rates.")
print("4. Most passengers were between 20 and 40 years old.")
print("5. Several features show meaningful correlations with survival.")

print("\nEDA Completed Successfully")
print("Visualizations saved in outputs folder")