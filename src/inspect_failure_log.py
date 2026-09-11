import pandas as pd


FILE = (
    "data/raw/"
    "Combined-Failure-Logbook-2016-2017.xlsx"
)


df = pd.read_excel(FILE)


print("=" * 80)
print("FAILURE LOG")
print("=" * 80)

print("\nShape:")
print(df.shape)


print("\nColumns:")

for i, col in enumerate(df.columns):

    print(
        f"{i:02d} : {col}"
    )


print("\nData types:")
print(df.dtypes)


print("\nFirst 20 rows:")
print(df.head(20).to_string())


print("\nMissing values:")
print(df.isnull().sum())