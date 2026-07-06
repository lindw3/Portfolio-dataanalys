import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import openpyxl
import statsmodels.formula.api as smf
from functools import reduce
from turtle import write


  # LADDA OCH PREPPA DATASET

    # Livslängd
livslängd = pd.read_csv("https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
livslängd.columns = ['land', 'kod', 'år', 'livslängd']
livslängd = livslängd[['land', 'år', 'livslängd']]

    # Barnadödlighet (innan 5 år)
barnadödlighet = pd.read_csv("https://ourworldindata.org/grapher/child-mortality-igme.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
barnadödlighet.columns = ['land', 'kod', 'år', 'barnadödlighet']
barnadödlighet = barnadödlighet[['land', 'år', 'barnadödlighet']]

    # Suicid (per 100 000 invånare)
suicid = pd.read_csv("https://ourworldindata.org/grapher/death-rate-from-suicides-gho.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
suicid.columns = ['land', 'kod', 'år', 'suicid/100k']
suicid = suicid[['land', 'år', 'suicid/100k']]

    # Alkoholkonsumption (liter)
alkohol = pd.read_csv("https://ourworldindata.org/grapher/total-alcohol-consumption-per-capita-litres-of-pure-alcohol.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
alkohol.columns = ['land', 'kod', 'år', 'alkohol_percapita']
alkohol = alkohol[['land', 'år', 'alkohol_percapita']]

    # Andel med fetma
fetma = pd.read_csv("https://ourworldindata.org/grapher/share-of-adults-defined-as-obese.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
fetma.columns = ['land', 'kod', 'år', 'fetma_andel', 'region']
fetma = fetma[['land', 'år', 'fetma_andel']]

    # HDI
hdi = pd.read_csv("https://ourworldindata.org/grapher/human-development-index.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
hdi.columns = ['land', 'kod', 'år', 'hdi', 'region']
hdi = hdi[['land', 'år', 'hdi']]

    # Human Rights Index
hri = pd.read_csv("https://ourworldindata.org/grapher/human-rights-index-vdem.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
hri.columns = ['land', 'kod', 'år', 'hri', 'region']
hri = hri[['land', 'år', 'hri']]

    # Demokratiindex
demokrati = pd.read_csv('data/electoral-democracy-index.csv')
demokrati.columns = ['land', 'år', 'demokratiindex', 'region']
demokrati = demokrati[['land', 'år', 'demokratiindex']]

    # CO2-utsläpp per capita
co2 = pd.read_csv("https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
co2.columns = ['land', 'kod', 'år', 'co2_percapita']
co2 = co2[['land', 'år', 'co2_percapita']]

    # Energikonsumtion per capita
energi = pd.read_csv('data/energy-use-per-person.csv')
energi.columns = ['land', 'år', 'energi_percapita']
energi = energi[['land', 'år', 'energi_percapita']]

    # Utbildning som andel av GDP
utbildning = pd.read_csv("https://ourworldindata.org/grapher/total-government-expenditure-on-education-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
utbildning.columns = ['land', 'kod', 'år', 'utbildning_andel_gdp']
utbildning = utbildning[['land', 'år', 'utbildning_andel_gdp']]

    # Hälsa- och sjukvård som andel av GDP
sjukvård = pd.read_csv("https://ourworldindata.org/grapher/public-health-expenditure-share-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
sjukvård.columns = ['land', 'kod', 'år', 'sjukvård_andel_gdp']
sjukvård = sjukvård[['land', 'år', 'sjukvård_andel_gdp']]

    # Womens Political Empowerment Index
wpe = pd.read_csv("https://ourworldindata.org/grapher/women-political-empowerment-index.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
wpe.columns = ['land', 'kod', 'år', 'wpe_index', 'region']
wpe = wpe[['land', 'år', 'wpe_index']]

    # Lönegap mellan könen
lönegap = pd.read_csv("https://ourworldindata.org/grapher/gender-gap-in-average-wages-ilo.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
lönegap.columns = ['land', 'kod', 'år', 'lönegap']
lönegap = lönegap[['land', 'år', 'lönegap']]

    # Andel kvinnor i arbete jämfört med män
andel_kvinnor_arbete = pd.read_csv("https://ourworldindata.org/grapher/ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
andel_kvinnor_arbete.columns = ['land', 'kod', 'år', 'andel_kvinnor_arbete', 'region']
andel_kvinnor_arbete = andel_kvinnor_arbete[['land', 'år', 'andel_kvinnor_arbete']]

    # Arbetstimmar per person i arbete
arbetstimmar = pd.read_csv("https://ourworldindata.org/grapher/annual-working-hours-per-person-employed.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
arbetstimmar.columns = ['land', 'kod', 'år', 'arbetstimmar']
arbetstimmar = arbetstimmar[['land', 'år', 'arbetstimmar']]

    # Genomsnittligt antal skolår
skolår = pd.read_csv("https://ourworldindata.org/grapher/average-years-of-schooling.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
skolår.columns = ['land', 'kod', 'år', 'skolår']
skolår = skolår[['land', 'år', 'skolår']]

    # Bistånd som andel av BNI
bistånd = pd.read_csv("https://ourworldindata.org/grapher/foreign-aid-given-as-a-share-of-national-income.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
bistånd.columns = ['land', 'kod', 'år', 'bistånd_andel_bni', 'annotation']
bistånd = bistånd[['land', 'år', 'bistånd_andel_bni']]

    # Skatt som andel av BNP
skatt = pd.read_csv("https://ourworldindata.org/grapher/tax-revenues-as-a-share-of-gdp-unu-wider.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
skatt.columns = ['land', 'kod', 'år', 'skatt_andel_bnp', 'region']
skatt = skatt[['land', 'år', 'skatt_andel_bnp']]

    # Statliga utgifter som andel av BNP
statligautgifter = pd.read_csv("https://ourworldindata.org/grapher/historical-gov-spending-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
statligautgifter.columns = ['land', 'kod', 'år', 'statligautgifter_andel_bnp', 'annotation']
statligautgifter = statligautgifter[['land', 'år', 'statligautgifter_andel_bnp']]

    # Gini-koefficient
gini = pd.read_excel('data/gini.xlsx')
gini.columns = ['land', 'år', 'gini']

    # GDP per capita
gdp = pd.read_csv("https://ourworldindata.org/grapher/gdp-per-capita-maddison-project-database.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
gdp.columns = ['land', 'kod',  'år', 'gdp_per_capita', 'annotation']
gdp = gdp[['land', 'år', 'gdp_per_capita']]

    # Handel som andel av GDP
handel = pd.read_csv("https://ourworldindata.org/grapher/trade-as-share-of-gdp.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
handel.columns = ['land', 'kod', 'år', 'handel_andel_gdp']
handel = handel[['land', 'år', 'handel_andel_gdp']]

    # Livstillfredsställelse
livstillfredsställelse = pd.read_csv("https://ourworldindata.org/grapher/happiness-cantril-ladder.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
livstillfredsställelse.columns = ['land', 'kod', 'år', 'livstillfredsställelse']
livstillfredsställelse = livstillfredsställelse[['land', 'år', 'livstillfredsställelse']]

    # Enpersonshushåll
hushåll = pd.read_csv("https://ourworldindata.org/grapher/one-person-households.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
hushåll.columns = ['land', 'kod', 'år', 'hushåll_andel']
hushåll = hushåll[['land', 'år', 'hushåll_andel']]

    # Barn per kvinna
barn = pd.read_csv("https://ourworldindata.org/grapher/children-born-per-woman.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
barn.columns = ['land', 'kod', 'år', 'barn_per_kvinna']
barn = barn[['land', 'år', 'barn_per_kvinna']]

    # Korruptionsindex
korruption = pd.read_csv("https://ourworldindata.org/grapher/political-corruption-index.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
korruption.columns = ['land', 'kod', 'år', 'korruption_index', 'region']
korruption = korruption[['land', 'år', 'korruption_index']]

    # Mord per capita
mord = pd.read_csv("https://ourworldindata.org/grapher/homicide-rate-ghe.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
mord.columns = ['land', 'kod', 'år', 'mord_percapita', 'region']
mord = mord[['land', 'år', 'mord_percapita']]

    # Död i väpnad konflikt
död_konflikt = pd.read_csv("https://ourworldindata.org/grapher/homicide-rate-ghe.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
död_konflikt.columns = ['land', 'kod', 'år', 'död_i_konflikt_percapita', 'region']
död_konflikt = död_konflikt[['land', 'år', 'död_i_konflikt_percapita']]




    # VILKET SPANN AV ÅR INRYMMER SAMTLIGA (ELLER SÅ MÅNGA SOM MÖJLIGT) AV DATAFRAMES?
    # Min = 2011, men de flesta har från 1990
    # Max = 2017, men de flesta har till 2023, något fler har 2021

    # SLÅ IHOP DATAFRAMES UTIFRÅN LAND OCH ÅR
dataramar = [
    livslängd,
    barnadödlighet,
    suicid,
    alkohol,
    fetma,
    hdi,
    hri,
    demokrati,
    co2,
    energi,
    utbildning,
    sjukvård,
    wpe,
    lönegap,
    andel_kvinnor_arbete,
    arbetstimmar,
    skolår,
    bistånd,
    skatt,
    statligautgifter,
    gini,
    gdp,
    handel,
    livstillfredsställelse,
    hushåll,
    barn,
    korruption,
    mord,
    död_konflikt
]

data = reduce(
    lambda left, right: pd.merge(
        left,
        right,
        on=['land', 'år'],
        how='outer'
    ),
    dataramar
)


    # Exkludera år och land, samt inkludera endast siffror från 2023
model_data = data[data['år'] == 2021]
model_data = model_data.drop(columns=['land', 'år'])
model_data = model_data.dropna(axis=1, how='all')


    # Corrplot
corrplt = model_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corrplt,
            annot=False,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True)

plt.title("Correlation Matrix")
plt.show()


    # Multipel regression för att se vilka faktorer som påverkar Demokratiindex
model_demokratiindex = smf.ols(
    "demokratiindex ~ age + education + experience",
    data=model_data
).fit()

print(model_demokratiindex.summary())



    # -..- Gini-koefficient

    # Fynd från corrplot: Ökad gini = minskat lönegap. Vidare undersökning visar att gini är som lägst i medelinkomstländer, vilket man tror beror på att endast kvinnor med goda kvalifikationer är i arbete i många av dessa länder.
    # Visualisering av detta i form av lönegap vs gdp per capita, logaritmisk x-axelskala

    # Multipel regression
model_gini = smf.ols(
    "gini ~ age + education + experience",
    data=model_data
).fit()

print(model_gini.summary())



    # "Tidsserieanalys" (det jag lärde mig på statistikkursen) för att estimera Sveriges GDP 2030?


    # Någon typ av maskininlärning för att predicera om ett land är en demokrati eller inte utifrån de andra faktorerna (exkl. demokratiindex)?
