import pandas as pd

"""
def fetch_drugs():
    
    #Expected response format from your API:
    [
        {"name": "Panadol", "stock": 10},
        {"name": "Brufen", "stock": 0}
    ]

    response = requests.get(DB_API_URL)
    return response.json()
"""
def fetch_drugs():
    df = pd.read_excel("Pharmacy_Products_ALL_added.xlsx")
    return df.to_dict(orient="records")
