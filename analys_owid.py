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

    # Livslängd
livslängd = pd.read_csv("https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Barnadödlighet (innan 5 år)
barnadödlighet = pd.read_csv("https://ourworldindata.org/grapher/child-mortality-igme.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Suicid (per 100 000 invånare)
suicid = pd.read_csv("https://ourworldindata.org/grapher/death-rate-from-suicides-gho.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Alkoholkonsumption (liter)
alkohol = pd.read_csv("https://ourworldindata.org/grapher/total-alcohol-consumption-per-capita-litres-of-pure-alcohol.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Andel med fetma
fetma = pd.read_csv("https://ourworldindata.org/grapher/share-of-adults-defined-as-obese.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # HDI
hdi = pd.read_csv("https://ourworldindata.org/grapher/human-development-index.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Human Rights Index
hri = pd.read_csv("https://ourworldindata.org/grapher/human-rights-index-vdem.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Demokratiindex
demokrati = pd.read_csv('data/electoral-democracy-index.csv')

    # CO2-utsläpp per capita
co2 = pd.read_csv("https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Energikonsumtion per capita
energi = pd.read_csv('data/energy-use-per-person.csv')

    # Utbildning som andel av GDP
utbildning = pd.read_csv("https://ourworldindata.org/grapher/total-government-expenditure-on-education-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Hälsa- och sjukvård som andel av GDP
sjukvård = pd.read_csv("https://ourworldindata.org/grapher/public-health-expenditure-share-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Womens Political Empowerment Index
wpe = pd.read_csv("https://ourworldindata.org/grapher/women-political-empowerment-index.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Lönegap mellan könen
lön = pd.read_csv("https://ourworldindata.org/grapher/gender-gap-in-average-wages-ilo.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Andel kvinnor i arbete jämfört med män
andel_kvinnor_arbete = pd.read_csv("https://ourworldindata.org/grapher/ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Arbetstimmar per person i arbete
arbetstimmar = pd.read_csv("https://ourworldindata.org/grapher/annual-working-hours-per-person-employed.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Genomsnittligt antal skolår
skolår = pd.read_csv("https://ourworldindata.org/grapher/average-years-of-schooling.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

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

    # Handel som andel av GDP
handel = pd.read_csv("https://ourworldindata.org/grapher/trade-as-share-of-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Livstillfredsställelse
livstillfredsställelse = pd.read_csv("https://ourworldindata.org/grapher/happiness-cantril-ladder.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Enpersonshushåll
hushåll = pd.read_csv("https://ourworldindata.org/grapher/one-person-households.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Barn per kvinna
barn = pd.read_csv("https://ourworldindata.org/grapher/children-born-per-woman.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Korruptionsindex
korruption = pd.read_csv("https://ourworldindata.org/grapher/political-corruption-index.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Mord per capita
mord = pd.read_csv("https://ourworldindata.org/grapher/homicide-rate-ghe.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})

    # Död i väpnad konflikt
död_konflikt = pd.read_csv("https://ourworldindata.org/grapher/homicide-rate-ghe.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})