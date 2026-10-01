import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from anywhere; paths below are relative to this folder
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
import pandas as pd

df = pd.read_csv("../data/raw/titanic.csv")

print("SHAPE:", df.shape)
print("\nDTYPES:\n", df.dtypes)
print("\nHEAD:\n", df.head())
print("\nDESCRIBE (numeric):\n", df.describe())
print("\nDESCRIBE (all):\n", df.describe(include='all'))
print("\nMISSING VALUES:\n", df.isnull().sum())
print("\nMISSING %:\n", (df.isnull().sum() / len(df) * 100).round(2))
print("\nDUPLICATE ROWS:", df.duplicated().sum())
print("\nUNIQUE Sex:", df['Sex'].unique())
print("UNIQUE Embarked:", df['Embarked'].unique())
print("UNIQUE Pclass:", df['Pclass'].unique())
print("\nNegative/zero Fare count:", (df['Fare'] <= 0).sum())
print("Age min/max:", df['Age'].min(), df['Age'].max())
print("Fare min/max:", df['Fare'].min(), df['Fare'].max())
