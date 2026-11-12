import pandas as pd

df = pd.read_csv("dataset/customer_reviews.csv")

df["review"] = df["review"].str.replace( r"\s*\(review example [A-Za-z0-9]+\)$", "",regex=True)

label_counts = df.groupby("review")["sentiment"].nunique()
conflicting_reviews = label_counts[label_counts > 1].index

df = df[~df["review"].isin(conflicting_reviews)]

df = df.drop_duplicates(subset="review")

df.to_csv("dataset/customer_reviews_clean.csv", index=False)

print("Dataset cleaned successfully!")
print("\nCLEANED DATASET SHAPE")
print(df.shape)

print("\nSENTIMENT DISTRIBUTION")
print(df["sentiment"].value_counts())

print("\nDUPLICATE REVIEWS")
print(df["review"].duplicated().sum())

print("\nFIRST FIVE CLEANED REVIEWS")
print(df.head())