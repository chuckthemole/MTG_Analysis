# Charles Thomas
# Data Analysis
# MTG Data Analysis2
# 2020/05/11

'''
This is a notebook to do an analysis on Magic the Gathering cards.
The json file containing the cards can be found at:
https://www.kaggle.com/mylesoneill/magic-the-gathering-cards

The main goal of the program is to plot a cards power and toughness
versus its mana cost.

Questions to be answered:
Is there a relationship between a cards mana cost and power?
Is there a relationship between a cards mana cost and toughness?
Is there a relationship between a cards power and toughness?
'''

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
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import urllib.request
from sklearn.linear_model import LinearRegression

# %% import files
with open('magic-the-gathering-cards/AllCards.json', encoding="utf8") as f:
  cards = json.load(f)
with open('magic-the-gathering-cards/AllPrices.json') as f:
  prices = json.load(f)

# %% make card DataFrame
cards_df = pd.DataFrame.from_dict(cards)
cards_df = cards_df.transpose()
cards_df

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

# %% Test Methods
plot_mana_pt()
plot_rank_pt()
plot_pt()

# %% LinearRegression for power versus toughness, Actual v Predicted
z = plot_pt()
lin_reg(z[0], z[1])


# %%*******************************************************
# *******************Helper Methods************************
# ********************************************************

# *********************************************************
# *********Plot convertedManaCost versus power*************
# *********Plot convertedManaCost versus toughness*********
# *********************************************************
def plot_mana_pt():
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

    plt.scatter(x, y, marker="^", label="Power")
    plt.legend()
    plt.ylim(-2, 20)
    plt.xlabel("Mana Cost")
    #plt.ylabel("Power")

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

    plt.scatter(x, y, marker="v", label="Toughness")
    plt.legend()
    plt.ylim(-2, 20)
    plt.xlabel("Mana Cost")
    plt.ylabel("Score")
    plt.show()

# *********************************************************
# *********Plot Rank versus power**************************
# *********Plot Rank versus toughness**********************
# *********************************************************
def plot_rank_pt():
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

    x = creatures["edhrecRank"][~nan_elems1]
    x = x[~nan_elems2]
    x = x[~nan_elems3]
    x = x[~nan_elems4]
    x = x[~nan_elems5]
    x = x[~nan_elems6]
    x = x[~nan_elems7]

    y = y.astype(float)

    plt.scatter(x, y, marker=".", label="Power")
    plt.legend()
    plt.ylim(-2, 20)
    plt.xlabel("Rank")
    plt.ylabel("Score (Power or Toughness)")

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

    x = creatures["edhrecRank"][~nan_elems1]
    x = x[~nan_elems2]
    x = x[~nan_elems3]
    x = x[~nan_elems4]
    x = x[~nan_elems5]
    x = x[~nan_elems6]
    x = x[~nan_elems7]
    x = x[~nan_elems8]
    x = x[~nan_elems9]

    #x = x.astype(float)
    y = y.astype(float)

    plt.scatter(x, y, marker=".", label="Toughness")
    plt.legend()
    plt.ylim(-2, 20)
    plt.show()

# *********************************************************
# *********Plot power versus toughness*********************
# *********************************************************
def plot_pt():
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

    # Toughness
    # Clean data; some toughness elements are not numbers
    nan_elems8 = creatures["toughness"].isin(["*"])
    nan_elems9 = creatures["toughness"].isnull()
    nan_elems10 = creatures["toughness"].isin(['1+*'])
    nan_elems11 = creatures["toughness"].isin(['2+*'])
    nan_elems12 = creatures["toughness"].isin(['∞'])
    nan_elems13 = creatures["toughness"].isin(['?'])
    nan_elems14 = creatures["toughness"].isin(['*²'])
    nan_elems15 = creatures["toughness"].isin(['*+1'])
    nan_elems16 = creatures["toughness"].isin(['7-*'])

    y = creatures["power"][~nan_elems1]
    y = y[~nan_elems2]
    y = y[~nan_elems3]
    y = y[~nan_elems4]
    y = y[~nan_elems5]
    y = y[~nan_elems6]
    y = y[~nan_elems7]
    y = y[~nan_elems8]
    y = y[~nan_elems9]
    y = y[~nan_elems10]
    y = y[~nan_elems11]
    y = y[~nan_elems12]
    y = y[~nan_elems13]
    y = y[~nan_elems14]
    y = y[~nan_elems15]
    y = y[~nan_elems16]

    x = creatures["toughness"][~nan_elems1]
    x = x[~nan_elems2]
    x = x[~nan_elems3]
    x = x[~nan_elems4]
    x = x[~nan_elems5]
    x = x[~nan_elems6]
    x = x[~nan_elems7]
    x = x[~nan_elems8]
    x = x[~nan_elems9]
    x = x[~nan_elems10]
    x = x[~nan_elems11]
    x = x[~nan_elems12]
    x = x[~nan_elems13]
    x = x[~nan_elems14]
    x = x[~nan_elems15]
    x = x[~nan_elems16]

    x = x.astype(float)
    y = y.astype(float)

    plt.scatter(x, y, marker=".")
    plt.ylim(-2, 20)
    plt.xlim(-2, 20)
    plt.xlabel("Toughness")
    plt.ylabel("Power")
    plt.show()

    return x, y, creatures

def lin_reg(x, y):
    x1 = x.values.reshape(-1, 1)
    y1 = y.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=0)

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X_train, y_train)  # perform linear regression
    y_pred = linear_regressor.predict(X_test)
    plt.scatter(X_test, y_test, color="gray")
    plt.plot(X_test, y_pred, color='red', linewidth=4)
    plt.xlabel("Toughness")
    plt.ylabel("Power")
    plt.show()

    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

    # Bar graph showing first 25 cards actual versus predicted
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    return df
