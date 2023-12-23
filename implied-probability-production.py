# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:11:10 2023

@author: quant
"""

from feature_functions import intrinsic_value_call, intrinsic_value_put, premium_discount

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

dates = calendar.schedule(start_date = (datetime.today()-timedelta(days=4)), end_date = (datetime.today())).index.strftime("%Y-%m-%d").values

# The most recent trading date
date = dates[-1]

# Your desired ticker
underlying_ticker = "AAPL"

underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
underlying_price = underlying["c"].iloc[0]

#

ticker_call_contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
ticker_call_contracts["date"] = pd.to_datetime(ticker_call_contracts["expiration_date"])
ticker_call_contracts["year"] = ticker_call_contracts["date"].dt.year
ticker_call_contracts["days_to_exp"] = (ticker_call_contracts["date"] - pd.to_datetime(date)).dt.days
ticker_call_contracts["distance_from_price"] = abs(ticker_call_contracts["strike_price"] - underlying_price)

all_expiration_dates = ticker_call_contracts["expiration_date"].drop_duplicates().values

smallest_strikes = ticker_call_contracts.groupby('strike_price').size().copy()
smallest_strikes = smallest_strikes[smallest_strikes == len(all_expiration_dates)].reset_index().copy()
smallest_strikes["distance_from_price"] = abs(smallest_strikes["strike_price"] - underlying_price)

atm_strike = smallest_strikes.nsmallest(1,"distance_from_price")["strike_price"].iloc[0]

pricing_data_list = []

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
        call_chain["premium"] = call_chain["last_quote.midpoint"]
        
        useful_calls = call_chain[["underlying_asset.price", "last_quote.bid","details.strike_price","premium", "distance_from_price"]].copy()
        
        #####
        
        put_chain = puts.copy()
        put_chain["underlying_asset.price"] = underlying_price
        put_chain["last_quote.last_updated"] = pd.to_datetime(put_chain["last_quote.last_updated"].values, unit = "ns", utc = True).tz_convert("America/New_York")
        put_chain["distance_from_price"] = abs(put_chain["details.strike_price"] - put_chain["underlying_asset.price"])
        put_chain["premium"] = put_chain["last_quote.midpoint"]
        
        useful_puts = put_chain[["underlying_asset.price", "last_quote.bid","details.strike_price","premium", "distance_from_price"]].copy()
        
        ##
        
        otm_calls = useful_calls[useful_calls["details.strike_price"] > underlying_price].copy()
        otm_puts = useful_puts[useful_puts["details.strike_price"] < underlying_price].copy()
        
        otm_options = pd.concat([otm_calls, otm_puts], axis = 0).sort_values(by = "details.strike_price", ascending = True).fillna(0)        
        
        otm_options = otm_options[otm_options["premium"] >= 0.01].copy()
        otm_options = otm_options.nsmallest(10,"distance_from_price").sort_values(by = "details.strike_price", ascending = True)
        
        total_premium = otm_options['premium'].sum()
        
        otm_options["probability"] = otm_options['premium'] / total_premium
        otm_options["ticker"] = underlying_ticker
        
        prob_above = round(otm_options[otm_options["details.strike_price"] >= atm_strike]["probability"].sum()*100, 2)
        prob_below = round(otm_options[otm_options["details.strike_price"] < atm_strike]["probability"].sum()*100, 2)
        
        pricing_string = f"The options market implies a {prob_above}% chance of {underlying_ticker} being above or equal to ${atm_strike} by {expiration_date}"
        print(pricing_string)
        
        pricing_dataframe = pd.DataFrame([{"expiration_date": expiration_date, "atm_strike": atm_strike,
                                           "prob_above": prob_above, "prob_below": prob_below}])
        
        pricing_data_list.append(pricing_dataframe)
        
        #####
        
        plt.figure(dpi = 200)
        # the main error source, generally due to not enough variation in the bids --  ignore
        # plt.hist(otm_options['details.strike_price'], weights=otm_options['premium'], density = True, color = "cornflowerblue")
        sns.kdeplot(x=otm_options['details.strike_price'], weights=otm_options['premium'], fill=True)
        plt.axvline(underlying_price, color='k', linestyle='dashed', linewidth=1, label='Underlying Price')
        plt.xlabel('Strike Price')
        plt.ylabel('Probability Density')
        plt.title(f'Smoothed Probability Distribution of Strike Prices - {underlying_ticker}')
        plt.suptitle(f"Expiration Date: {expiration_date}")
        plt.legend()
        plt.show()
        
        ###
        
    except Exception as error:
        print(error)
        continue
    
pricing_term_structure = pd.concat(pricing_data_list)

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.plot(pricing_term_structure["expiration_date"], pricing_term_structure["prob_above"], marker='o', linestyle='-', color='lawngreen')
plt.plot(pricing_term_structure["expiration_date"], pricing_term_structure["prob_below"], marker='s', linestyle='--', color='r')
plt.xlabel("Expiration Date")
plt.ylabel("Probability")
plt.legend([f"Probability of price being above or equal to {atm_strike}", f"Probability of price being below {atm_strike}"])
plt.grid(True, linestyle='--', alpha=0.7)
plt.title("Probability Distribution over Expiration Dates")
plt.show()