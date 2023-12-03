import pandas as pd
import requests
import json

if __name__ == '__main__':
    n = int(input('Select mode:\n\t1. Enter object features in json format and get predictions\n\t2. Enter csv filename with object features, get predictions saved in csv\n\nChoice [1/2]: '))
    
    if n == 1:
        obj = input('Enter object features in json format: ')
        r = requests.post('http://localhost:8000/predict_item', json=json.loads(obj))
        print('Predicted price:', r.text)
    elif n == 2:
        filename = input('Enter csv file name: ')
        df = pd.read_csv(filename)
        payload = df.to_dict('records')
        _ = list(map(lambda x: x.pop('Unnamed: 0'), payload))

        r = requests.post('http://localhost:8000/predict_items', json=payload)
        df['predicted_price'] = r.json()

        df.to_csv(filename + '_predicted')
    else:
        print('No such option!')