from fastapi import FastAPI
import pandas as pd

fake_app = FastAPI()

df = pd.read_excel("Pharmacy_Products_ALL_added.xlsx")

@fake_app.get("/drugs")
def get_drugs():
    return df.to_dict(orient="records")
