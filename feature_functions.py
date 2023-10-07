# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:55:53 2023

@author: Local User
"""
import numpy as np
import math
from scipy.stats import norm

def binomial_option_price(S, K, T, r, sigma, n, option_type):
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    
    option_tree = [[0 for j in range(n+1)] for i in range(n+1)]
    
    # Calculate option values at expiration (n periods)
    for j in range(n+1):
        if option_type == 'call':
            option_tree[n][j] = max(0, S * (u ** (n-j)) * (d ** j) - K)
        elif option_type == 'put':
            option_tree[n][j] = max(0, K - S * (u ** (n-j)) * (d ** j))
    
    # Backward induction to calculate option values at earlier nodes
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            if option_type == 'call':
                option_tree[i][j] = max(0, math.exp(-r * dt) * (p * option_tree[i+1][j] + (1-p) * option_tree[i+1][j+1]))
            elif option_type == 'put':
                option_tree[i][j] = max(0, math.exp(-r * dt) * (p * option_tree[i+1][j] + (1-p) * option_tree[i+1][j+1]))
    
    return option_tree[0][0]

def bjerksund_stensland_greeks(S, K, T, r, sigma, option_type):
    
    if option_type == "call":
        option_type = 0
    elif option_type == "put":
        option_type = 1
    
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    alpha = (r * (1 - option_type) - 0.5 * (sigma**2)) / (sigma**2)
    beta = (r * (1 - option_type) + 0.5 * (sigma**2)) / (sigma**2)
    
    if option_type == 0:
        # Calculate Delta for a call option
        delta = norm.cdf(d1)
        
        # Calculate Gamma for a call option
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Calculate Theta for a call option
        theta = (r * K * np.exp(-r * T) * norm.cdf(d2) -
                 (r - beta * sigma**2) * S * norm.cdf(d1) -
                 (1 - option_type) * (r - beta * sigma**2) * S * norm.pdf(d1) / (2 * np.sqrt(T)))
        
        # Calculate Vega for a call option
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        return delta, gamma, theta, vega

    elif option_type == 1:
        # Calculate Delta for a put option
        delta = -norm.cdf(-d1)
        
        # Calculate Gamma for a put option
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Calculate Theta for a put option
        theta = (r * K * np.exp(-r * T) * norm.cdf(-d2) -
                 (r - beta * sigma**2) * S * norm.cdf(-d1) +
                 (1 - option_type) * (r - beta * sigma**2) * S * norm.pdf(-d1) / (2 * np.sqrt(T)))
        
        # Calculate Vega for a put option
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        return delta, gamma, theta, vega

def Binarizer(number):
    
    if number <= 0:
        return 0
    elif number > 0:
        return 1
    
def return_proba(prediction_dataset):
    probabilities = []
    
    for row in prediction_dataset.index:
        
        prediction_data = prediction_dataset[prediction_dataset.index == row]
        prediction = prediction_data["prediction"].iloc[0]
        if prediction == 0:
            probabilities.append(prediction_data["probability_0"].iloc[0])
        elif prediction == 1:
            probabilities.append(prediction_data["probability_1"].iloc[0])
            
    return probabilities

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def butterfly_cost(x):
    return (x.iloc[1] * 2) - (x.iloc[0] + x.iloc[2])

def get_non_one(row):
    if row['price_below_strike_prob'] != 1:
        return row['price_below_strike_prob']
    elif row['price_above_strike_prob'] != 1:
        return row['price_above_strike_prob']
    else:
        return None
    
def premium_discount(row):
    if row["last_quote.bid"] - row["intrinsic_value"] < 0:
        return 0
    else:
        return row["last_quote.bid"] - row["intrinsic_value"]

def intrinsic_value_call(row):
    if row["underlying_asset.price"] - row["details.strike_price"] < 0:
        return 0
    else:
        return row["underlying_asset.price"] - row["details.strike_price"]

def intrinsic_value_put(row):
    if row["details.strike_price"] - row["underlying_asset.price"] < 0:
        return 0
    else:
        return row["details.strike_price"] - row["underlying_asset.price"] 