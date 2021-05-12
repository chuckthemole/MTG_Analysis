# Charles Thomas
# Data Analysis
# Final Project
# 2020/05/09

# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.datasets
import json

# %% Takes Euros to float number **********Need to to this, **********************
def to_float(euro):
    e = euro.replace(".", "")
    e = e.replace(",", ".")
    e = e.replace("€", "")
    e = e.replace(" ", "")
    return float(e)

# %% import files
with open('magic-the-gathering-cards/AllCards.json', encoding="utf8") as f:
  cards = json.load(f)
with open('magic-the-gathering-cards/AllPrices.json') as f:
  prices = json.load(f)

# %% make card DataFrame
cards_df = pd.DataFrame.from_dict(cards)
cards_df = cards_df.transpose()
cards_df.columns
print()
#print(cards_df.head(3))
#print(cards_df.columns)


# %%
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
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
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


# %% returns a random sample of 10 card prices as a DataFrame
import urllib.request

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

### test test test
j = random_sample_prices()["price"]
j
to_float(j.iloc[6])

# %% trying to pull prices from web pages
import urllib.request
noData = 0
loopCount = 0

# Create extra column for prices in DataFrame
cards_df["price"] = np.nan
'''
Trying to set cells to values, use the top two.
cards_df.loc[cards_df.iloc[0]["name"], "price"]
cards_df.loc[cards_df.iloc[0]["name"], "price"] = 9

cards_df["price"][cards_df.iloc[0]["name"]]
cards_df.iloc[0]["price"] = 1
cards_df["price"][cards_df.iloc[0]["name"]] = 10
'''

# Get rid of cards with no price info
cond = cards_df["purchaseUrls"].isnull()
cards_with_prices = pd.DataFrame(cards_df[~cond])
random_sample = pd.DataFrame(cards_with_prices.sample(frac=1)) # Obtain a random sample
#random_sample["purchaseUrls"][0]

# Pull urls for cards and feed into function to determine current price
for i in range(0, 10):   # cards_df.size takes too long
    loopCount += 1
    try:
        try:
            random_sample["purchaseUrls"][i]["cardmarket"]
            url = random_sample["purchaseUrls"][i]["cardmarket"]
            random_sample.loc[random_sample.iloc[i]["name"], "price"] = scrape_price(url)
        except:
            url = random_sample["purchaseUrls"][i]["tcgplayer"]
            random_sample.loc[random_sample.iloc[i]["name"], "price"] = scrape_price(url)
    except:
        noData += 1

print("The loop went", loopCount, "times.")
print("There is no data for", noData, "entries.")
print("Sample size of price data is", (loopCount - noData), "in size.")

random_sample["price"]
cond = random_sample["price"].isnull()
type(random_sample["price"].head(10))
random_sample["convertedManaCost"].head(10)
#random_sample = pd.DataFrame(cards_with_prices[~cond])

# %% Scraping html
import re

def scrape_price(url):
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
    return str(price.group(0))


# %% Scraping html
#raw_html = simple_get(url)
raw_html = simple_get(cards_df[cond]["purchaseUrls"][0]["cardmarket"]) # Black Lotus test
len(raw_html)
#print(raw_html)
html = BeautifulSoup(raw_html, 'html.parser')
x = html.findAll("section")

# Using regular expression to parse the html
import re
for i in x:
    if i['id'] == 'tabs':
        s = re.search(
            '30-days average price.+\d*\.*\d+,\d\d €', str(i))
        try:
            c = re.search('\d*\.*\d+,\d\d €', str(s.group(0)))
        except:
            print("error in parsing")

print(s.group(0))
print(str(c))

# %% edhrecRank
cond = cards_df["edhrecRank"].isnull()
rankedCards = cards_df["edhrecRank"][~cond]
rankedCards.size
cards_df.size

# %% edhrecRank
cond = cards_df["edhrecRank"].isnull()
rankedCards = pd.DataFrame(cards_df[~cond])
rankedCards["edhrecRank"].iloc[0]
rankedCards["edhrecRank"].sort_values(ascending=False)


# %% Frequency for convertedManaCost
count = cards_df.groupby("convertedManaCost").size()
count.to_frame()
count.colums = ["convertedManaCost", "numberOfCards"]
count.plot.bar(x="convertedManaCost", y="numberOfCards")
plt.show()

# %% Frequency Top 10 Types
#maybe see if different types have higher mana cost
types = cards_df.groupby("type").size()
types.to_frame()
types.colums = ["type", "numberOfCards"]
top20 = types.sort_values(ascending=False).head(20)
top20.head(10)
top20.plot.bar(x="type", y="numberOfCards", rot=80)
plt.show()

#cond = cards_df["type"].isin(["Artifact"])
cond = cards_df["type"].str.contains("Creature")
types = cards_df[cond]["type"]
types


# %% ******************************************************
# *********Plot convertedManaCost versus power*************
# *********Plot convertedManaCost versus toughness*********
# *********************************************************

# Create Creature DataFrame
cond = cards_df["type"].str.contains("Creature")
creatures = cards_df[cond]

#Power
# Clean data; some power elements are not numbers
nan_elems1 = creatures["power"].isin(["*"])
nan_elems2 = creatures["power"].isnull()
nan_elems3 = creatures["power"].isin(['1+*'])
nan_elems4 = creatures["power"].isin(['2+*'])
nan_elems5 = creatures["power"].isin(['∞'])
nan_elems6 = creatures["power"].isin(['?'])
nan_elems7 = creatures["power"].isin(['*²'])

y = creatures["power"][~nan_elems1]
y = y[~nan_elems2]
y = y[~nan_elems3]
y = y[~nan_elems4]
y = y[~nan_elems5]
y = y[~nan_elems6]
y = y[~nan_elems7]

x = creatures["convertedManaCost"][~nan_elems1]
x = x[~nan_elems2]
x = x[~nan_elems3]
x = x[~nan_elems4]
x = x[~nan_elems5]
x = x[~nan_elems6]
x = x[~nan_elems7]

x = x.astype(float)
y = y.astype(float)

plt.scatter(x, y, marker="^")
plt.ylim(-2, 20)
plt.xlabel("Mana Cost")
plt.ylabel("Power")
#plt.show()

# Toughness
# Clean data; some toughness elements are not numbers
nan_elems1 = creatures["toughness"].isin(["*"])
nan_elems2 = creatures["toughness"].isnull()
nan_elems3 = creatures["toughness"].isin(['1+*'])
nan_elems4 = creatures["toughness"].isin(['2+*'])
nan_elems5 = creatures["toughness"].isin(['∞'])
nan_elems6 = creatures["toughness"].isin(['?'])
nan_elems7 = creatures["toughness"].isin(['*²'])
nan_elems8 = creatures["toughness"].isin(['*+1'])
nan_elems9 = creatures["toughness"].isin(['7-*'])

# Clean data; some power elements are not numbers
y = creatures["toughness"][~nan_elems1]
y = y[~nan_elems2]
y = y[~nan_elems3]
y = y[~nan_elems4]
y = y[~nan_elems5]
y = y[~nan_elems6]
y = y[~nan_elems7]
y = y[~nan_elems8]
y = y[~nan_elems9]

x = creatures["convertedManaCost"][~nan_elems1]
x = x[~nan_elems2]
x = x[~nan_elems3]
x = x[~nan_elems4]
x = x[~nan_elems5]
x = x[~nan_elems6]
x = x[~nan_elems7]
x = x[~nan_elems8]
x = x[~nan_elems9]

x = x.astype(float)
y = y.astype(float)
#x.size
#y.size
plt.scatter(x, y, marker="v")
plt.ylim(-2, 20)
plt.xlabel("Mana Cost")
plt.ylabel("Toughness")
plt.show()


# %% Not working out
for v in cards_df["colors"].unique():
    cond = cards_df["colors"] == v
    plt.scatter(cards_df[cond]["convertedManaCost"], cards_df[cond]["power"])

nan_elems1 = cards_df["power"].isin(["*"])
nan_elems2 = cards_df["power"].isnull()
nan_elems3 = cards_df["power"].isin(['1+*'])
nan_elems4 = cards_df["power"].isin(['2+*'])

#nan_elems.append(cards_df["power"].isnull())
y = cards_df["power"][~nan_elems1]
y = y[~nan_elems2]
y = y[~nan_elems3]
y = y[~nan_elems4]

x = cards_df["convertedManaCost"][~nan_elems1]
x = x[~nan_elems2]
x = x[~nan_elems3]
x = x[~nan_elems4]

try:
    plt.scatter(x.astype(int), y.astype(int))
except:
  print("An exception occurred")
plt.show()

# %% codecell
#prices_df = pd.DataFrame.from_dict(prices)
#prices_df = prices_df.transpose()
#print(prices_df.head(3))
#print(prices_df.columns)
#df = prices_df["prices"]
#df = df.to_frame()
#print(type(df))
#print(df.columns)
#print(prices_df["purchaseUrls"].head(3))
