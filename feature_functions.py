# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:55:53 2023

@author: Local User
"""
import numpy as np
import math
import pandas as pd
import scipy.optimize as optimize

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
    
def black_scholes(option_type, S, K, t, r, q, sigma):
    """
    Calculate the Black-Scholes option price.
    
    :param option_type: 'call' for call option, 'put' for put option.
    :param S: Current stock price.
    :param K: Strike price.
    :param t: Time to expiration (in years).
    :param r: Risk-free interest rate (annualized).
    :param q: Dividend yield (annualized).
    :param sigma: Stock price volatility (annualized).
    
    :return: Option price.
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    
    if option_type == 'call':
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")
        
def black_scholes_greeks(S, K, T, r, sigma, option_type):
    """
    Computes the Black-Scholes Greeks: Delta, Gamma, Theta, Vega, Rho
    
    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (str): 'call' for call option, 'put' for put option
    
    Returns:
    dict: A dictionary containing the Black-Scholes Greeks
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = - (S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(d2))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = - (S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # Dividing by 100 to scale vega to 1% change in volatility
    
    greeks = {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }
    
    return pd.Series(greeks)

def seconds_to_days(seconds):
    """
    Converts seconds to days.
    
    Parameters:
    seconds (int or float): The number of seconds.
    
    Returns:
    float: The number of days.
    """
    seconds_per_day = 24 * 60 * 60  # Number of seconds in a day
    days = seconds / seconds_per_day
    return days

def call_implied_vol(row):
    S = row["underlying_price"]
    K = row["strike_price"]
    t = row["time_to_exp"]
    r = .05
    q = 0.015
    option_type = "call"
    
    def f_call(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - row["call_c"]

    try:        
        call_newton_vol = optimize.newton(f_call, x0=0.15, tol=0.05, maxiter=50)
    except:
        call_newton_vol = np.nan
    
    return call_newton_vol

def put_implied_vol(row):
    S = row["underlying_price"]
    K = row["strike_price"]
    t = row["time_to_exp"]
    r = .05
    q = 0.015
    option_type = "put"
    
    def f_put(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - row["put_c"]

    try:        
        put_newton_vol = optimize.newton(f_put, x0=0.15, tol=0.05, maxiter=50)
    except:
        put_newton_vol = np.nan
    
    return put_newton_vol

def call_fair_value(row):
    
    S = row["underlying_price"]
    K = row["strike_price"]
    t = row["time_to_exp"]
    sigma = row["call_implied_vol"]
    r = .05
    q = 0.015
    option_type = "call"
    
    if np.isnan(sigma):
        return np.nan
    else:
        return black_scholes(option_type, S, K, t, r, q, sigma)
    
def put_fair_value(row):
    
    S = row["underlying_price"]
    K = row["strike_price"]
    t = row["time_to_exp"]
    sigma = row["put_implied_vol"]
    r = .05
    q = 0.015
    option_type = "put"
    
    if np.isnan(sigma):
        return np.nan
    else:
        return black_scholes(option_type, S, K, t, r, q, sigma)        
    
def call_greeks(row):
    
   S = row["underlying_price"]
   K = row["strike_price"]
   T = row["time_to_exp"]
   sigma = row["call_implied_vol"]
   r = .05
   q = 0.015
   option_type = "call"
   
   if np.isnan(sigma):
       return np.nan
   else:
       return black_scholes_greeks(S, K, T, r, sigma, option_type)
   
def put_greeks(row):
    
   S = row["underlying_price"]
   K = row["strike_price"]
   T = row["time_to_exp"]
   sigma = row["put_implied_vol"]
   r = .05
   q = 0.015
   option_type = "put"
   
   if np.isnan(sigma):
       return np.nan
   else:
       return black_scholes_greeks(S, K, T, r, sigma, option_type)  