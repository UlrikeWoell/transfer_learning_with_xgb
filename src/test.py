import pandas as pd

df = pd.read_csv("data/simulation/matrix_any_sign/237_08210934211692603261/tgt_train.csv")
print(1 in df["y"].values)