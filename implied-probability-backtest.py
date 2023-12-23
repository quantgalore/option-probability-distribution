# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import sqlalchemy
import mysql.connector

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar


def set_prediction(row):
    
    if row["prob_above"] > row["prob_below"]:
        return 1
    elif row["prob_below"] > row["prob_above"]:
        return 0


def open_actual(row):
    
    if row["next_open_price"] >= row["atm_strike"]:
        return 1
    elif row["next_open_price"] < row["atm_strike"]:
        return 0
    
def close_actual(row):
    
    if row["next_closing_price"] >= row["atm_strike"]:
        return 1
    elif row["next_closing_price"] < row["atm_strike"]:
        return 0

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

tz = pytz.timezone("GMT")

start_date = "2023-08-01"
end_date = (datetime.today()).strftime("%Y-%m-%d")

trade_dates = calendar.schedule(start_date = start_date, end_date = end_date)
trade_dates["day_of_week"] = trade_dates.index.strftime("%A")

fridays = trade_dates[trade_dates["day_of_week"] == "Friday"].index.strftime("%Y-%m-%d").values
thursdays = trade_dates[trade_dates["day_of_week"] == "Thursday"].index.strftime("%Y-%m-%d").values

weekly_expiration_tickers = ["SOFI", "AMD", "PLTR", "AMC", "GME", "AFRM", "RBLX","UBER", "AAPL","SNAP", "TSLA", "COIN", "JPM", "NFLX"]

trade_list = []

for underlying_ticker in weekly_expiration_tickers:
    
    try:
        
        pricing_data_list = []
        times = []
        strikes_in_calculation = 10
        
        for date in thursdays:
            
            start_time = datetime.now()
        
            underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{date}/{date}?adjusted=false&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
            underlying_price = underlying["c"].iloc[0]
        
            #
        
            expiration_date = (pd.to_datetime(date) + timedelta(days=1)).strftime("%Y-%m-%d")
                
            try:
                
                calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&expiration_date={expiration_date}&contract_type=call&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
                calls["distance_from_price"] = abs(calls["strike_price"] - underlying_price)
                calls = calls.nsmallest(n=strikes_in_calculation, columns="distance_from_price").sort_values(by="strike_price", ascending=True)
                atm_strike = calls[calls["distance_from_price"] == calls["distance_from_price"].min()]["strike_price"].iloc[0]
                
                puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&expiration_date={expiration_date}&contract_type=put&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
                puts["distance_from_price"] = abs(puts["strike_price"] - underlying_price)
                puts = puts.nsmallest(n=strikes_in_calculation, columns="distance_from_price").sort_values(by="strike_price", ascending=True)
                
                #
                
                pre_close_timestamp = (pd.to_datetime(date) + timedelta(hours = 15, minutes = 55)).tz_localize("America/New_York").tz_convert(tz).value
                close_timestamp = (pd.to_datetime(date) + timedelta(hours = 16, minutes = 00)).tz_localize("America/New_York").tz_convert(tz).value
                
                call_data_list = []
                put_data_list = []
                
                for call in calls["ticker"]:
                    
                    try: 
                        
                        call_quote = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{call}?timestamp.gte={pre_close_timestamp}&timestamp.lt={close_timestamp}&limit=50000&sort=timestamp&order=desc&apiKey=KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3").json()["results"]).set_index("sip_timestamp")
                        call_quote.index = pd.to_datetime(call_quote.index, origin = "unix").tz_localize(tz).tz_convert("America/New_York")
                        call_quote = call_quote.head(1)
                        
                        call_info = calls[calls["ticker"] == call].copy()
                        
                        call_data = pd.DataFrame([{"underlying_price": underlying_price,"strike_price": call_info["strike_price"].iloc[0], "bid": call_quote["bid_price"].iloc[0], "ask": call_quote["ask_price"].iloc[0],
                                                   "mid_price":(call_quote["bid_price"].iloc[0] + call_quote["ask_price"].iloc[0])/2,
                                                  "quote_time": call_quote.index[0]}])
                        
                        call_data_list.append(call_data)
                        
                    except Exception as call_error:
                        continue
                
                for put in puts["ticker"]:
                    
                    try:
                    
                        put_quote = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{put}?timestamp.gte={pre_close_timestamp}&timestamp.lt={close_timestamp}&limit=50000&sort=timestamp&order=desc&apiKey=KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3").json()["results"]).set_index("sip_timestamp")
                        put_quote.index = pd.to_datetime(put_quote.index, origin = "unix").tz_localize(tz).tz_convert("America/New_York")
                        put_quote = put_quote.head(1)
                        
                        put_info = puts[puts["ticker"] == put].copy()
                        
                        put_data = pd.DataFrame([{"underlying_price": underlying_price,"strike_price": put_info["strike_price"].iloc[0], "bid": put_quote["bid_price"].iloc[0], "ask": put_quote["ask_price"].iloc[0],
                                                   "mid_price":(put_quote["bid_price"].iloc[0] + put_quote["ask_price"].iloc[0])/2,
                                                   "quote_time": put_quote.index[0]}])
                        
                        put_data_list.append(put_data)
                                    
                    except Exception as put_error:
                        continue
                    
      
                if (len(call_data_list) < 1) or (len(put_data_list) < 1):  
                    continue
                
                call_chain = pd.concat(call_data_list)
                put_chain = pd.concat(put_data_list)
                
                full_options_chain = pd.merge(call_chain, put_chain, on = "strike_price")
                
                full_options_chain["straddle_cost"] = full_options_chain["mid_price_x"] + full_options_chain["mid_price_y"]
                full_options_chain["distance_from_price"] = abs(full_options_chain["strike_price"] - full_options_chain["underlying_price_x"])
                
                otm_calls = full_options_chain[full_options_chain["strike_price"] > full_options_chain["underlying_price_x"]].copy()[["underlying_price_x","strike_price", "mid_price_x"]].rename(columns={"mid_price_x": "mid_price", "underlying_price_x":"underlying_price"})
                otm_puts = full_options_chain[full_options_chain["strike_price"] < full_options_chain["underlying_price_x"]].copy()[["underlying_price_x","strike_price", "mid_price_y"]].rename(columns={"mid_price_y": "mid_price", "underlying_price_x":"underlying_price"})
                
                otm_options = pd.concat([otm_calls, otm_puts]).sort_values(by="strike_price",ascending=True)
                otm_options["distance_from_price"] = abs(otm_options["strike_price"] - otm_options["underlying_price"])
                otm_options = otm_options.nsmallest(10,"distance_from_price").sort_values(by = "strike_price", ascending = True)
                
                otm_options["probability"] = otm_options["mid_price"] / otm_options["mid_price"].sum()
                
                prob_above = round(otm_options[otm_options["strike_price"] >= atm_strike]["probability"].sum()*100, 2)
                prob_below = round(otm_options[otm_options["strike_price"] < atm_strike]["probability"].sum()*100, 2)
                
                next_day = (pd.to_datetime(date) + timedelta(days=1)).strftime("%Y-%m-%d")
                
                underlying_next_day = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{next_day}/{next_day}?adjusted=false&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                underlying_next_day.index = pd.to_datetime(underlying_next_day.index, unit = "ms", utc = True).tz_convert("America/New_York")
                
                pricing_dataframe = pd.DataFrame([{"expiration_date": expiration_date,"underlying_price":underlying_price,
                                                   "atm_strike": atm_strike,
                                                   "prob_above": prob_above, "prob_below": prob_below,
                                                   "next_open_price": underlying_next_day["o"].iloc[0],
                                                   "next_closing_price": underlying_next_day["c"].iloc[0]}])
                
                pricing_data_list.append(pricing_dataframe)
                
                end_time = datetime.now()
                seconds_to_complete = (end_time - start_time).total_seconds()
                times.append(seconds_to_complete)
                iteration = round((np.where(thursdays==date)[0][0]/len(thursdays))*100,2)
                iterations_remaining = len(thursdays) - np.where(thursdays==date)[0][0]
                average_time_to_complete = np.mean(times)
                estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
                time_remaining = estimated_completion_time - datetime.now()
                
                print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
                
            except Exception as error:
                print(error)
                continue
        
    
        full_pricing_data = pd.concat(pricing_data_list).set_index("expiration_date")
        full_pricing_data.index = pd.to_datetime(full_pricing_data.index)
        
        full_pricing_data["prediction"] = full_pricing_data.apply(set_prediction, axis = 1)
        full_pricing_data["open_actual"] = full_pricing_data.apply(open_actual, axis = 1)
        full_pricing_data["closing_actual"] = full_pricing_data.apply(close_actual, axis = 1)
        
        full_pricing_data["ticker"] = underlying_ticker
    
        #################
        
        higher = full_pricing_data[(full_pricing_data["prediction"] == 1)].copy()
        higher_close_win_rate = len(higher[higher["prediction"] == higher["closing_actual"]]) / len(higher)
        
        ###
        
        lower = full_pricing_data[(full_pricing_data["prediction"] == 0)].copy()
        if len(lower) < 1:
            lower_closing_win_rate = np.nan
        else:
            lower_closing_win_rate = len(lower[lower["prediction"] == lower["closing_actual"]]) / len(lower)
            
        
        ########
        
        both = pd.concat([higher, lower], axis = 0).sort_index(ascending = True)
        overall_win_rate = len(both[both["prediction"] == both["closing_actual"]]) / len(both)
        print(f"Overall W/R: {round(overall_win_rate*100,2)}%")
        
        win_rate_data = pd.DataFrame([{"ticker": underlying_ticker, "win_rate": overall_win_rate}])
        
        ########
        trade_list.append(win_rate_data)
        ########

    except Exception as macro_error:
        print(macro_error, underlying_ticker)
        continue
    
ticker_performances = pd.concat(trade_list)
ticker_performances["win_rate"] = round(ticker_performances["win_rate"] * 100,2)

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.xlabel("Tickers")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of Probability Distribution by Ticker")
plt.scatter(x=ticker_performances["ticker"], y=ticker_performances["win_rate"])
text = ticker_performances["ticker"].values
for i in range(len(ticker_performances)): 
    plt.annotate(text[i], (ticker_performances["ticker"].values[i], ticker_performances["win_rate"].values[i] +.1)) 
plt.show()