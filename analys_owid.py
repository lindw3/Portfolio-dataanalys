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



   # HAR ALLA DATAFRAMES SAMMA UPPDELNING AV LÄNDER, ELLER NÅGON SOM HAR VÄRLDSDELAR ELLER ANDRA NIVÅER?
   # VILKET SPANN AV ÅR INRYMMER SAMTLIGA (ELLER SÅ MÅNGA SOM MÖJLIGT) AV DATAFRAMES?

print(livslängd.head)
print(livslängd.Year.min(), livslängd.Year.max())

print(barnadödlighet.head)
print(barnadödlighet.Year.min(), barnadödlighet.Year.max())

print(suicid.head)
print(suicid.Year.min(), suicid.Year.max())

print(alkohol.head)
print(alkohol.Year.min(), alkohol.Year.max())

print(fetma.head)
print(fetma.Year.min(), fetma.Year.max())

print(hdi.head)
print(hdi.Year.min(), hdi.Year.max())

print(hri.head)
print(hri.Year.min(), hri.Year.max())

print(demokrati.head)
print(demokrati.Year.min(), demokrati.Year.max())

print(co2.head)
print(co2.Year.min(), co2.Year.max())

print(energi.head)
print(energi.Year.min(), energi.Year.max())

print(utbildning.head)
print(utbildning.Year.min(), utbildning.Year.max())

print(sjukvård.head)
print(sjukvård.Year.min(), sjukvård.Year.max())

print(wpe.head)
print(wpe.Year.min(), wpe.Year.max())

print(lön.head)
print(lön.Year.min(), lön.Year.max())

print(andel_kvinnor_arbete.head)
print(andel_kvinnor_arbete.Year.min(), andel_kvinnor_arbete.Year.max())

print(arbetstimmar.head)
print(arbetstimmar.Year.min(), arbetstimmar.Year.max())

print(skolår.head)
print(skolår.Year.min(), skolår.Year.max())

print(bistånd.head)
print(bistånd.Year.min(), bistånd.Year.max())

print(skatt.head)
print(skatt.Year.min(), skatt.Year.max())

print(statligautgifter.head)
print(statligautgifter.Year.min(), statligautgifter.Year.max())

print(gini.head)
print(gini.Year.min(), gini.Year.max())

print(gdp.head)
print(gdp.Year.min(), gdp.Year.max())

print(handel.head)
print(handel.Year.min(), handel.Year.max())

print(livstillfredsställelse.head)
print(livstillfredsställelse.Year.min(), livstillfredsställelse.Year.max())

print(hushåll.head)
print(hushåll.Year.min(), hushåll.Year.max())

print(barn.head)
print(barn.Year.min(), barn.Year.max())

print(korruption.head)
print(korruption.Year.min(), korruption.Year.max())

print(mord.head)
print(mord.Year.min(), mord.Year.max())

print(död_konflikt.head)
print(död_konflikt.Year.min(), död_konflikt.Year.max())


    # Multipel regression för att se vilka faktorer som påverkar Demokratiindex

    # SLÅ IHOP DATAFRAMES UTIFRÅN LAND OCH ÅR
        # Ta responsmåttet och left join med df som har samma land och år


    # "Tidsserieanalys" (det jag lärde mig på statistikkursen) för att estimera Sveriges GDP 2030?
