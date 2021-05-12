# Charles Thomas
# Data Analysis
# MTG Data Analysis
# 2020/05/11

'''
This is a notebook to do an analysis on Magic the Gathering cards.
The json file containing the cards can be found at:
https://www.kaggle.com/mylesoneill/magic-the-gathering-cards

The main goal of the program is to plot a cards cost versus its mana cost.
This can be done under 'Main'. It does this by scraping a website for a card's
current cost and storing that price in a DataFrame.

Improvements:
Scraping is taking a long time. Improvements need to be made in order for this
to run efficiently. Right now, 10 cards are taken at rondom to plot, since the
program is taking long.
'''

# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import urllib.request

# %% import files
with open('magic-the-gathering-cards/AllCards.json', encoding="utf8") as f:
  cards = json.load(f)
with open('magic-the-gathering-cards/AllPrices.json') as f:
  prices = json.load(f)

# %% make card DataFrame
cards_df = pd.DataFrame.from_dict(cards)
cards_df = cards_df.transpose()
#cards_df.columns

# %% look up Black Lotus
cond = cards_df["name"] == "Black Lotus"
cards_df[cond]["purchaseUrls"][0]["cardmarket"]
cards_df[cond]

# %% look up Black Lotus
cond = cards_df["name"] == "Alpha Tyrranax"
cards_df[cond]
cards_df[cond]["edhrecRank"][0]

# %% *****************Main*********************
# %%
# Plot random sample
random_sample = random_sample_prices()
x = random_sample["price"]
y = random_sample["convertedManaCost"]
plt.scatter(x, y, marker=".")
plt.title("Price versus Mana Cost")
plt.xlabel("Current cost of card in €")
plt.ylabel("Mana cost")
plt.show()

# %% *************************************************************
#    ****************Helper methods*******************************
#    *************************************************************
# Returns a random sample of 10 card prices as a DataFrame
def random_sample_prices():
    cond = cards_df["purchaseUrls"].isnull()
    cards_with_prices = pd.DataFrame(cards_df[~cond])
    cards_with_prices["price"] = np.nan # Create extra column for prices in DataFrame
    random_sample = pd.DataFrame(cards_with_prices.sample(frac=1)) # Randomizes indices

    for i in range(0, 10):  # cards_df.size takes too long
        try:
            try:
                random_sample["purchaseUrls"][i]["cardmarket"]
                url = random_sample["purchaseUrls"][i]["cardmarket"]
                random_sample.loc[random_sample.iloc[i]["name"], "price"] = scrape_price(url)
            except:
                url = random_sample["purchaseUrls"][i]["tcgplayer"]
                random_sample.loc[random_sample.iloc[i]["name"], "price"] = scrape_price(url)
        except:
            print("error")

    return random_sample.head(10)

# Scrapes html for 30-days average price of magic card
def scrape_price(url):
    import re

    raw_html = simple_get(url)
    html = BeautifulSoup(raw_html, 'html.parser')
    sections = html.findAll("section")

    for i in sections:
        if i['id'] == 'tabs':
            s = re.search('30-days average price.+\d*\.*\d+,\d\d €', str(i))
            try:
                price = re.search('\d*\.*\d+,\d\d €', str(s.group(0)))
            except:
                print("error in parsing")
    return to_float(price.group(0))
    #return str(price.group(0))

# Takes Euros to float number
def to_float(euro):
    e = euro.replace(".", "")
    e = e.replace(",", ".")
    e = e.replace("€", "")
    e = e.replace(" ", "")
    return float(e)

# Below are helpers for get request
def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)

def log_error(e):
    print(e)
