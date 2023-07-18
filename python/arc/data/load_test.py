import pandas as pd

def load_dummy_data():
    df = pd.read_csv("/python/arc/data/arc_febrl1.csv")
    return df