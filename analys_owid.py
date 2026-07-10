import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import openpyxl
import statsmodels.api as sm
from functools import reduce
from turtle import write


  # LADDA OCH PREPPA DATASET

    # Livslängd
livslängd = pd.read_csv("https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
livslängd.columns = ['land', 'kod', 'år', 'livslängd']
livslängd = livslängd[['land', 'år', 'livslängd']]

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
död_konflikt = pd.read_csv("https://ourworldindata.org/grapher/deaths-in-armed-conflicts.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
död_konflikt.columns = ['land', 'kod', 'år', 'död_i_konflikt_percapita_hög', 'död_i_konflikt_percapita', 'död_i_konflikt_percapita_låg']
död_konflikt = död_konflikt[['land', 'år', 'död_i_konflikt_percapita']]




    # VILKET SPANN AV ÅR INRYMMER SAMTLIGA (ELLER SÅ MÅNGA SOM MÖJLIGT) AV DATAFRAMES?
    # Min = 2011, men de flesta har från 1990
    # Max = 2017, men de flesta har till 2023, något fler har 2021

    # SLÅ IHOP DATAFRAMES UTIFRÅN LAND OCH ÅR
dataramar = [
    livslängd,
    suicid,
    alkohol,
    fetma,
    hdi,
    demokrati,
    co2,
    energi,
    utbildning,
    sjukvård,
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


    # Exkludera år och land, samt inkludera endast siffror från 2021
model_data = data[data['år'] == 2021]
model_data = model_data.drop(columns=['land', 'år'])
model_data = model_data.dropna(axis=1, how='all')

    # Undersök hur många värden som är NaN per land
missing_share = model_data.isna().mean(axis=1)

    # Noterat att det är en hög andel värden som är NaN

    # Exkludera länder med mer än 25% NaN-värden
model_data = model_data[missing_share <= 0.25]

    # Fyra variabler med hög andel NaN-värden som exkluderas
model_data = model_data.drop(columns=['sjukvård_andel_gdp', 'gini', 'lönegap', 'bistånd_andel_bni'])

    # Imputering av resterande NaN-värden med medianen för respektive variabel
imputer = SimpleImputer(strategy='median')
model_data[model_data.columns] = imputer.fit_transform(model_data)

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



    # Fynd från corrplot: Ökad gini = minskat lönegap.
    # Fynd från corrplot: Ökad GDP per capita = lägre gini
    # Undersök dessa samband!

    # Gini vs lönegap - innebär ökad omjämlikhet totalt sett även omjämlikhet mellan könen?

lönegap_gini = data[['land', 'år', 'lönegap', 'gini']].dropna()
lönegap_gini = lönegap_gini[lönegap_gini['år'] == 2021]

# Visualisering: lönegap vs Gini-koefficient

points = (
    alt.Chart(lönegap_gini)
    .mark_circle(size=70)
    .encode(
        x=alt.X(
            "gini:Q",
            title="Gini-koefficient",
            scale=alt.Scale(
                domain=[0.2, 0.6]
            ),
            axis=alt.Axis(
                format=".2f"
            )
        ),
        y=alt.Y(
            "lönegap:Q",
            title="Ojusterat lönegap",
            scale=alt.Scale(
                domain=[-30, 40]
            ),
            axis=alt.Axis(
                labelExpr="datum.value + '%'"
            )
        ),
        tooltip=[
            alt.Tooltip(
                "land:N",
                title="Land"
            ),
            alt.Tooltip(
                "gini:Q",
                title="Gini-koefficient",
                format=".1f"
            ),
            alt.Tooltip(
                "lönegap:Q",
                title="Lönegap, %",
                format=".1f"
            )
        ]
    )
)

# Horisontell referenslinje vid y = 0
zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(strokeWidth=2, color="black")
    .encode(y="y:Q")
)

fig = (
    (zero_line + points)
    .properties(
            width="container",
            height=350
            )
    .configure_view(
        stroke=None
    )
    .configure_axis(
        gridColor="#e6e6e6",
        gridWidth=0.8,
        domain=False,
        tickColor="#999",
        labelFontSize=12,
        titleFontSize=15
    )
    .interactive()
)

fig

    # Tydlig trend där högre gini = lägre lönegap, eller snarare övervikt mot kvinnor
    # Länderna som är på eller under linjen är låg- medelinkomstländer

    # Har noterat i corrplot att gini har en negativ korrelation med gdp per capita
    # Gini vs GDP per capita

gdp_gini = data[['land', 'år', 'gdp_per_capita', 'gini']].dropna()
gdp_gini = gdp_gini[gdp_gini['år'] == 2021]

# Huvuddiagram
points = (
    alt.Chart(gdp_gini)
    .mark_circle(size=70)
    .encode(
        x=alt.X(
            "gdp_per_capita:Q",
            title="GDP per capita ($), logaritmisk skala",
            scale=alt.Scale(type="log"),
            axis=alt.Axis(
                values=[1000, 2500, 5000, 10000, 25000, 50000],
                format=",.0f"
            )
        ),
        y=alt.Y(
            "gini:Q",
            title="Gini-koefficient",
            scale=alt.Scale(domain=[0.2, 0.6]),
            axis=alt.Axis(
                format=".2f"
            )
        ),
        tooltip=[
    alt.Tooltip("land:N", title="Land"),
    alt.Tooltip("gdp_per_capita:Q", title="GDP per capita, $", format=",.0f"),
    alt.Tooltip("gini:Q", title="Gini-koefficient", format=".2f")
]
    )
)


fig = (
    points
    .properties(
            width="container",
            height=350
            )
    .configure_view(
        stroke=None
    )
    .configure_axis(
        gridColor="#e6e6e6",
        gridWidth=0.8,
        domain=False,
        tickColor="#999",
        labelFontSize=12,
        titleFontSize=15
    )
    .interactive()
)

fig

    # Figuren visar att gini är högre särskilt i medelinkomstländer, men lägst i höginkomstländer.


    # Visualisering av lönegap vs gdp per capita, logaritmisk x-axelskala

    # Hämta ut data för 2022 gällande lönegap och gdp per capita per land
lönegap_gdp = data[['land', 'år', 'lönegap', 'gdp_per_capita']].dropna()
lönegap_gdp = lönegap_gdp[lönegap_gdp['år'] == 2022]

    # Visualisering med cirkeldiagram

# Huvuddiagram
points = (
    alt.Chart(lönegap_gdp)
    .mark_circle(size=70)
    .encode(
        x=alt.X(
            "gdp_per_capita:Q",
            title="GDP per capita ($), logaritmisk skala",
            scale=alt.Scale(type="log"),
            axis=alt.Axis(
                values=[1000, 2500, 5000, 10000, 25000, 50000],
                format=",.0f"
            )
        ),
        y=alt.Y(
            "lönegap:Q",
            title="Ojusterat lönegap",
            scale=alt.Scale(domain=[-30, 40]),
            axis=alt.Axis(
            labelExpr="datum.value + '%'"
            )
        ),
        tooltip=[
    alt.Tooltip("land:N", title="Land"),
    alt.Tooltip("gdp_per_capita:Q", title="GDP per capita, $", format=",.0f"),
    alt.Tooltip("lönegap:Q", title="Lönegap, %", format=".1f")
]
    )
)

# Horisontell referenslinje vid y = 0
zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(strokeWidth=2, color="black")
    .encode(y="y:Q")
)

fig = (
    (zero_line + points)
    .properties(
            width="container",
            height=350
            )
    .configure_view(
        stroke=None
    )
    .configure_axis(
        gridColor="#e6e6e6",
        gridWidth=0.8,
        domain=False,
        tickColor="#999",
        labelFontSize=12,
        titleFontSize=15
    )
    .interactive()
)

fig

    # Vidare undersökning visar att lönegapet är som lägst i medelinkomstländer, vilket man tror beror på att endast kvinnor med goda kvalifikationer är i arbete i många av dessa länder.




    # Multipel regression
# Demokratiindex som beroende variabel
y = model_data["demokratiindex"]

# X-variabler
X = model_data.drop(columns=["demokratiindex"])

# Spara kolumnnamn och index innan transformation
X_columns = X.columns
X_index = X.index

# Standardisera
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Återskapa DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X_columns,
    index=X_index
)

# Lägg till intercept
X_scaled = sm.add_constant(X_scaled)

# OLS med robusta standardfel
model = sm.OLS(y, X_scaled).fit(cov_type="HC3")

print(model.summary())

    # För hög multikollinearitet, genomför VIF-analys
vif_df = pd.DataFrame()
vif_df["variable"] = X.columns
vif_df["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_df = vif_df[vif_df["variable"] != "const"]
print(vif_df.sort_values("VIF", ascending=False))

    # Ta bort hdi pga hög VIF (42). HDI är en sammansatt variabel som påverkas av utbildning, hälsa och inkomst, vilket förklarar den höga VIF:en.
X = X.drop(columns=['hdi'])

    # Gör om VIF-analysen efter borttagning av HDI

vif_df = pd.DataFrame()
vif_df["variable"] = X.columns
vif_df["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_df = vif_df[vif_df["variable"] != "const"]
print(vif_df.sort_values("VIF", ascending=False))

    # OK

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

    # Färdig summering av vilka variabler som hänger ihop med demokratiindex, alltså - hur tenderar länder med högre demokratiindex se ut? Mer av X, Y, Z men mindre av A, B, C.


    # Nu - skapa en modell som så bra som möjligt predicerar demokratiindex OBEROENDE AV ÖVRIGA VARIABLER.
    # Genom att exkludera variabler som är icke statistiskt signifikanta

X = X.drop(columns=['mord_percapita', 'död_i_konflikt_percapita', 'skolår'])

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

    # Ytterligare justeringar
X = X.drop(columns=['livslängd', 'utbildning_andel_gdp'])

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

    # Ytterligare justeringar
X = X.drop(columns=['suicid/100k', 'skatt_andel_bnp'])

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

    # Ytterligare justeringar
X = X.drop(columns=['gdp_per_capita'])

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

    # Ytterligare justeringar
X = X.drop(columns=['andel_kvinnor_arbete', 'handel_andel_gdp', 'fetma_andel'])

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

  # Ytterligare justeringar
X = X.drop(columns=['energi_percapita'])

X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

    # Färdig modell där samtliga är statistiskt signifikanta!
    # Sista VIF-analys

vif_df = pd.DataFrame()
vif_df["variable"] = X.columns
vif_df["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_df = vif_df[vif_df["variable"] != "const"]
print(vif_df.sort_values("VIF", ascending=False))

    # Betydligt bättre VIF-värden, vilket indikerar att variablerna är relativt oberoende av varandra.
  
    # "En initial modell inkluderade ett större antal samhällsindikatorer och användes deskriptivt för att identifiera potentiella samband. Därefter reducerades modellen genom att exkludera variabler utan statistiskt stöd samt indikatorer med stark konceptuell överlappning. Den reducerade modellen används för att identifiera variabler som uppvisar samband med demokratiindex när övriga inkluderade faktorer hålls konstanta."

    # Starkast association med demokratiindex, i ordning: Korruptionsindex (negativ korrelation), livstillfredsställelse, barn per kvinna, co2-utsläpp per capita, statliga utgifter som andel av BNP.




    # Någon typ av maskininlärning för att predicera om ett land är en demokrati eller inte utifrån de andra faktorerna (exkl. demokratiindex)?





    # "Tidsserieanalys" (det jag lärde mig på statistikkursen) för att estimera Sveriges GDP 2030?
