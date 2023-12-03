from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import uvicorn
from transforms import transform

app = FastAPI()

with open("./regr.pkl", 'rb') as f:
    regression = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int | None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def prepare_df(items: List[Item]):
    df = pd.DataFrame(columns=Item.model_json_schema()['required'])
    for i, item in enumerate(items):
        df.loc[i] = item.dict()
    
    df = transform(df, regression.feature_names_in_)

    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = prepare_df([item])
    pred = regression.predict(df)
    return pred[0]

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = prepare_df(items)
    pred = regression.predict(df)
    return list(pred)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug")
