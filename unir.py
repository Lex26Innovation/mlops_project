import pandas as pd
import glob

print("Uniendo los 10 archivos")


archivos = glob.glob("data/raw/p*_extrac.csv")


df_total = pd.concat((pd.read_csv(f) for f in archivos), ignore_index=True)


df_total.to_csv("data/raw/Data_CU_venta.csv", index=False)

print(" El archivo Data_CU_venta.csv fue creado con éxito en data/raw/")