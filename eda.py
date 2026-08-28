import pandas as pd
import numpy as np

df = pd.read_csv("data/creditcard.csv")
print("Shape:", df.shape)
print("Nulls:", df.isnull().sum().sum())
print("\nClass balance:")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))

print("\nAmount by class:")
print(df.groupby("Class")["Amount"].describe())

print("\nTop correlations with Class:")
corr = df.corr()["Class"].drop("Class").sort_values()
print(pd.concat([corr.head(5), corr.tail(5)]))
