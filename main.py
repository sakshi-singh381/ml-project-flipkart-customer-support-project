from src.data_cleaning import load_data, clean_data
from src.eda import basic_info, complaint_analysis, category_analysis, response_analysis
from src.visualization import complaint_chart, csat_chart

df = load_data("data/Customer_support_data (1).csv")
df = clean_data(df)

basic_info(df)
complaint_analysis(df)
category_analysis(df)
response_analysis(df)

complaint_chart(df)
csat_chart(df)

print("\nPROJECT COMPLETED SUCCESSFULLY 🚀")