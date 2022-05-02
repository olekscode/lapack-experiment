import pandas as pd 
tsv_file='feynman_I_10_7.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('feynman_I_10_7.csv', index=False)