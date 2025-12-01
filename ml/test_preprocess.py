from connect import load_financial_data
from preprocessing import preprocess_dataframe

df = load_financial_data()
processed_df, encoder, scaler = preprocess_dataframe(df)

print(processed_df.head())
print(processed_df.info())
