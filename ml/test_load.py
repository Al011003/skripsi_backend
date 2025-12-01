from connect import load_financial_data

df = load_financial_data()
print(df.head())
print(df.info())

from connect import load_financial_data
from preprocessing import preprocess_dataframe

