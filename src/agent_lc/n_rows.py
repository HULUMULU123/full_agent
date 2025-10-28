import pandas as pd
import sys


input_file = "../../data/real_data.csv"     
output_file = "../../data/sample_transactions.csv"  
n = 50                 


df = pd.read_csv(input_file)

if n > len(df):
    print(f"В файле всего {len(df)} строк, выбираю все.")
    n = len(df)


sampled = df.sample(n=n, random_state=42)


sampled.to_csv(output_file, index=False)

print(f"Сохранено {n} случайных строк в {output_file}")
