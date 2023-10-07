# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import intrinsic_value_call, intrinsic_value_put, premium_discount

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta, datetime

polygon_api_key = "your polygon.io api key. Use 'QUANTGALORE' for 10% off."

# the code pulls snapshots, so if you're running it on a weekend set the date equal to the last Friday by moving the days = n value
# on weekdays, just use the date = datetime.today().strftime() line.

date = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
# date = datetime.today().strftime('%Y-%m-%d')

underlying_ticker = "ATVI"

underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
underlying_price = underlying["c"].iloc[0]

#

ticker_call_contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
ticker_call_contracts["date"] = pd.to_datetime(ticker_call_contracts["expiration_date"])
ticker_call_contracts["year"] = ticker_call_contracts["date"].dt.year
ticker_call_contracts["days_to_exp"] = (ticker_call_contracts["date"] - pd.to_datetime(date)).dt.days

all_expiration_dates = ticker_call_contracts["expiration_date"].drop_duplicates().values

first_available_expiration_date = ticker_call_contracts[ticker_call_contracts["expiration_date"] > date]["expiration_date"].iloc[0]
one_year_expiration_date = ticker_call_contracts[ticker_call_contracts["days_to_exp"] >= 300]["expiration_date"].iloc[0]

for expiration_date in all_expiration_dates:
    try:
        
        #
        
        calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/snapshot/options/{underlying_ticker}?expiration_date={expiration_date}&limit=250&contract_type=call&apiKey={polygon_api_key}").json()["results"])
        puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/snapshot/options/{underlying_ticker}?expiration_date={expiration_date}&limit=250&contract_type=put&apiKey={polygon_api_key}").json()["results"])
        
        #####
        
        call_chain = calls.copy()
        call_chain["underlying_asset.price"] = underlying_price
        call_chain["last_quote.last_updated"] = pd.to_datetime(call_chain["last_quote.last_updated"].values, unit = "ns", utc = True).tz_convert("America/New_York")
        call_chain["distance_from_price"] = abs(call_chain["details.strike_price"] - call_chain["underlying_asset.price"])
        call_chain["intrinsic_value"] = call_chain.apply(intrinsic_value_call, axis = 1)
        call_chain["premium"] = call_chain.apply(premium_discount, axis =1)
        
        useful_calls = call_chain[["underlying_asset.price", "last_quote.bid","details.strike_price","intrinsic_value","premium", "distance_from_price"]].copy()
        
        #####
        
        put_chain = puts.copy()
        put_chain["underlying_asset.price"] = underlying_price
        put_chain["last_quote.last_updated"] = pd.to_datetime(put_chain["last_quote.last_updated"].values, unit = "ns", utc = True).tz_convert("America/New_York")
        put_chain["distance_from_price"] = abs(put_chain["details.strike_price"] - put_chain["underlying_asset.price"])
        put_chain["intrinsic_value"] = put_chain.apply(intrinsic_value_put, axis = 1)
        put_chain["premium"] = put_chain.apply(premium_discount, axis =1)
        
        useful_puts = put_chain[["underlying_asset.price", "last_quote.bid","details.strike_price","intrinsic_value","premium", "distance_from_price"]].copy()
        
        ##
        
        otm_calls = useful_calls[useful_calls["details.strike_price"] > underlying_price].copy()
        otm_puts = useful_puts[useful_puts["details.strike_price"] < underlying_price].copy()
        
        otm_call_premium = otm_calls["premium"].sum()
        otm_put_premium = otm_puts["premium"].sum()
        imbalance = otm_call_premium - otm_put_premium
        
        otm_options = pd.concat([otm_calls, otm_puts], axis = 0).sort_values(by = "details.strike_price", ascending = True).fillna(0)
        otm_options = otm_options[otm_options["premium"] >= 0.05].copy()
        otm_options["ticker"] = underlying_ticker
        # otm_options = otm_options.nsmallest(n = 10, columns = "distance_from_price", keep = "all").sort_values(by="details.strike_price", ascending = True)
        least_likely_strike = otm_options[otm_options["premium"] == otm_options["premium"].min()]["details.strike_price"].iloc[0]
        otm_options["least_likely_strike"] = least_likely_strike
        
        premium_dataframe = pd.DataFrame([{"ticker": underlying_ticker, "otm_call_prem": otm_call_premium, "underlying_price": underlying_price,
                                          "otm_put_prem": otm_put_premium, "imbalance": imbalance}])
        
        #####
        
        plt.figure(dpi = 200)
        # the main error source, generally due to not enough variation in the bids --  ignore
        plt.hist(otm_options['details.strike_price'], weights=otm_options['premium'], density = True, color = "cornflowerblue")
        sns.kdeplot(x=otm_options['details.strike_price'], weights=otm_options['premium'], fill=True)
        plt.axvline(underlying_price, color='k', linestyle='dashed', linewidth=1, label='Underlying Price')
        plt.axvline(least_likely_strike, color='r', linestyle='dashed', linewidth=1, label='Implied Least Likely Strike')
        plt.xlabel('Strike Price')
        plt.ylabel('Probability Density')
        plt.title(f'Smoothed Probability Distribution of Strike Prices - {underlying_ticker}')
        plt.suptitle(f"Expiration Date: {expiration_date}")
        # plt.grid(axis='y')
        plt.legend()
        plt.show()
        
    except Exception as error:
        print(error)
        continue
        