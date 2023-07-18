import pandas as pd
import importlib.resources

def load_dummy_data():
    #files = importlib.resources.path("arc.data", "arc_febrl1.csv")
    with importlib.resources.path("arc.data", "arc_febrl1.csv") as p:
                        csv_path = p.as_posix()
    df = pd.read_csv (csv_path)
    return df