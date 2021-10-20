import pandas as pd

df = pd.read_csv(r"C:\Users\BioGuest\test\test_aix\Datab_test\avocado.csv")
print(df["AveragePrice"].head())

albany_df = df[ df['region'] == "Albany"] 
print(albany_df.index)

