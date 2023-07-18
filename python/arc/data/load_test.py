import pandas as pd

def load_dummy_data():
    df = pd.read_csv("/data/arc_febrl1.csv")
    return df