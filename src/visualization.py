import matplotlib.pyplot as plt

def subcategory_chart(df):
    fig, ax = plt.subplots()
    df['sub-category'].value_counts().head(10).plot(kind='bar', ax=ax)
    plt.title("Top Sub Categories")
    return fig