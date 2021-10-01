import pandas as pd
import json
import os
from tqdm import tqdm

# set variables
exchange = 'Binance'
book_path = f'./Data/{exchange}/book'
trade_path = f'./Data/{exchange}/trade'
book_csv = 'binance_l2_cleaned.csv'
trade_csv = 'Binance_trades.csv'
aggregate = False
chunksize = 10 ** 6


def string_to_json(text: str):
    text = text.replace("'", '"')
    json_data = json.loads(text)
    return json_data


def level1_price_generator(text: str):
    try:
        json_data = string_to_json(text)
        prices = json_data.keys()
        return float(list(prices)[0])

    except:
        return None


def level1_volume_generator(text: str):
    try:
        json_data = string_to_json(text)
        volumes = json_data.values()
        return float(list(volumes)[0])

    except:
        return None


def total_depth(text: str):
    try:
        json_data = string_to_json(text)
        volume_list = list(map(float, json_data.values()))
        return sum(volume_list)

    except:
        return None

def generate_aggregated_df(path: str, data_type: str):
    df = pd.DataFrame()
    cwd = os.getcwd()
    os.chdir(path)
    for file in os.listdir():
        if file.endswith('.csv'):
            aux = pd.read_csv(file, error_bad_lines=False, engine="python")
            df = df.append(aux)
    try:
        if data_type == 'book':
            df = df.drop_duplicates(['best_bid_price','best_ask_price','best_bid_volume','best_ask_volume'])
            df = df.sort_values('receipt_timestamp')
            df = df[['receipt_timestamp', 'best_bid_price', 'best_ask_price',
                     'best_bid_volume', 'best_ask_volume', 'bid_total_depth', 'ask_total_depth']]

        else:
            df = df.sort_values('receipt_timestamp')
            df = df[['receipt_timestamp', 'side', 'amount', 'price']]

    except Exception as e:
        raise e

    df.to_csv(f"{exchange}_{data_type}_aggregated.csv")
    os.chdir(cwd)


# Processing Book Data
i = 1
print(f'Processing Book Data for {exchange}')
for chunk in pd.read_csv(f'{book_path}/{book_csv}', chunksize=chunksize, error_bad_lines=False, engine="python"):
    chunk = chunk[['receipt_timestamp', 'bid', 'ask']]

    print(f'processing book chunk {i}')
    # apply feature engineering
    chunk['best_bid_price'] = chunk.apply(lambda x: level1_price_generator(x['bid']), axis=1)
    chunk['best_ask_price'] = chunk.apply(lambda x: level1_price_generator(x['ask']), axis=1)
    chunk['best_bid_volume'] = chunk.apply(lambda x: level1_volume_generator(x['bid']), axis=1)
    chunk['best_ask_volume'] = chunk.apply(lambda x: level1_volume_generator(x['ask']), axis=1)
    chunk['bid_total_depth'] = chunk.apply(lambda x: total_depth(x['bid']), axis=1)
    chunk['ask_total_depth'] = chunk.apply(lambda x: total_depth(x['ask']), axis=1)


    # Cleaning and Sorting
    raw_l2 = chunk.dropna()
    raw_l2 = raw_l2.drop_duplicates(['best_bid_price','best_ask_price','best_bid_volume','best_ask_volume'])
    raw_l2 = raw_l2.sort_values('receipt_timestamp')
    raw_l2.to_csv(f'{book_path}/{exchange}_l2_cleaned{i}.csv')
    print(f'Finished processing book chunk {i}')
    i += 1

# Processing Trade data
j = 1
print(f'Processing trade Data for {exchange}')
for chunk in pd.read_csv(f'{trade_path}/{trade_csv}', chunksize=chunksize, error_bad_lines=False, engine="python"):
    chunk = chunk[['receipt_timestamp', 'side', 'amount', 'price']]
    chunk = chunk.sort_values('receipt_timestamp')

    print(f'processing trade chunk {j}')
    chunk.to_csv(f'{trade_path}/{exchange}_trades_cleaned{j}.csv')
    print(f'Finished processing trade chunk {j}')
    j += 1

# Aggregate cleaned files
if aggregate is True:
    generate_aggregated_df(book_path, data_type='book')
    generate_aggregated_df(trade_path, data_type='trade')


