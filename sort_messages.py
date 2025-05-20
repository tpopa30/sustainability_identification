import pandas as pd

df = pd.read_csv("values/ground_truth.csv")
check = pd.read_csv("values/data.csv")

df_yes = df[df["relevant"] == "Yes"]
df_no = df[df["relevant"] == "No"]

# df_yes.to_csv("values/related.csv", index=False)
# df_no.to_csv("values/unrelated.csv", index=False)

df_filtered = check[check["Id"].isin(df_yes["Id"])]
df_filtered.to_csv("values/related.csv", index=False)

df_filtered = check[check["Id"].isin(df_no["Id"])]
df_filtered.to_csv("values/unrelated.csv", index=False)