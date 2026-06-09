def basic_info(df):
    print("\nSHAPE:", df.shape)
    print("\nCOLUMNS:", df.columns.tolist())


def complaint_analysis(df):
    print("\nCATEGORY WISE:")
    print(df['category'].value_counts().head(10))


def category_analysis(df):
    print("\nPRODUCT CATEGORY:")
    print(df['product_category'].value_counts().head(10))


def response_analysis(df):
    print("\nCSAT AVERAGE:", df['csat_score'].mean())
    print("HANDLING TIME:", df['connected_handling_time'].mean())