import pandas as pd
tsv_file='1191_BNG_pbc.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('1191_BNG_pbc.csv', index=False)
