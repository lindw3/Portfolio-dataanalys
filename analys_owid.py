import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import requests




  # LADDA DATASET
    # CO2-utsläpp per capita
co2 = pd.read_csv("https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


    # Utbildning som andel av GDP
utbildning = pd.read_csv("https://ourworldindata.org/grapher/total-government-expenditure-on-education-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


    # Hälsa- och sjukvård som andel av GDP
sjukvård = pd.read_csv("https://ourworldindata.org/grapher/public-health-expenditure-share-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


    # Lönegap mellan könen
lön = pd.read_csv("https://ourworldindata.org/grapher/gender-gap-in-average-wages-ilo.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


    # Arbetstimmar per person i arbete
arbetstimmar = pd.read_csv("https://ourworldindata.org/grapher/annual-working-hours-per-person-employed.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


    # Bistånd som andel av BNI
bistånd = pd.read_csv("https://ourworldindata.org/grapher/foreign-aid-given-as-a-share-of-national-income.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})


    # Skatt som andel av BNP
skatt = pd.read_csv("https://ourworldindata.org/grapher/tax-revenues-as-a-share-of-gdp-unu-wider.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Statliga utgifter som andel av BNP
statligautgifter = pd.read_csv("https://ourworldindata.org/grapher/historical-gov-spending-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Gini-koefficient
gini = pd.read_excel('data/gini.xlsx')

    # GDP per capita
gdp = pd.read_csv("https://ourworldindata.org/grapher/gdp-per-capita-maddison-project-database.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Livstillfredsställelse
livstillfredsställelse = pd.read_csv("https://ourworldindata.org/grapher/happiness-cantril-ladder.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Enpersonshushåll
hushåll = pd.read_csv("https://ourworldindata.org/grapher/one-person-households.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Barn per kvinna
barn = pd.read_csv("https://ourworldindata.org/grapher/children-born-per-woman.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

