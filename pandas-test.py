import pandas as pd

df = pd.read_csv("train.csv")

print("Head")
print(df.head())
print("Shape")
print(df.shape)
print("Unique qtypes")
print(df["qtype"].unique())
print("Value counts")
print(df["qtype"].value_counts())

print("Type of df")
print(type(df))
print("Type of df['qtype']")
print(type(df["qtype"]))

# Take the first 10 entries and create a shortened CSV file
df_short = df.head(10)
df_short.to_csv("train-shortened.csv", index=False)
print(f"\nCreated train-shortened.csv with {len(df_short)} entries")
print("First few rows of shortened file:")
print(df_short.head())

print("Reading shortened file")
df = pd.read_csv("train-shortened.csv")

print("Head")
print(df.head(10))
print("Shape")
print(df.shape)
print("Unique qtypes")
print(df["qtype"].unique())
print("Value counts")
print(df["qtype"].value_counts())

print("Type of df")
print(type(df))
print("Type of df['qtype']")
print(type(df["qtype"]))
