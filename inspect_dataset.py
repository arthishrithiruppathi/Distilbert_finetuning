import pandas as pd

df = pd.read_csv("dataset/customer_reviews.csv")

print("\nFIRST FIVE REVIEWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSENTIMENT DISTRIBUTION")
print(df["sentiment"].value_counts())

print("\nUNIQUE SENTIMENTS")
print(df["sentiment"].unique())

print("\nTOTAL REVIEWS")
print(len(df))
