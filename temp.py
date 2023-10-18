import os
import pandas as pd

folder = "csvs"
csv_files = os.listdir(folder)
queries = []
papers = []
summaries = []

for file in csv_files:
    df = pd.read_csv(os.path.join(folder, file))
    queries.append(df["query"].iloc[0])
    papers.append(df["paper"].iloc[0])
    summaries.append(df["summary"].iloc[0])

N = len(csv_files)

print(summaries)