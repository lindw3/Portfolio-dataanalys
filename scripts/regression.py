import pandas as pd

from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# --------------------------------------------------
# LADDA DATA
# --------------------------------------------------

data = pd.read_parquet(
    "data/combined_dataset.parquet"
)



# --------------------------------------------------
# DATAFÖRBEREDELSE
# --------------------------------------------------

# Begränsar analysen till 2021 eftersom de flesta
# indikatorer har relativt hög täckning detta år.
model_data = data[
    data["år"] == 2021
]


# Land och år används inte som prediktorer.
model_data = model_data.drop(
    columns=[
        "land",
        "år"
    ]
)


# Tar bort variabler som helt saknar data.
model_data = model_data.dropna(
    axis=1,
    how="all"
)


# Undersöker andelen saknade värden per observation.
# Länder med mycket begränsad datatäckning exkluderas.
missing_share = model_data.isna().mean(axis=1)


model_data = model_data[
    missing_share <= 0.25
]


# Dessa indikatorer exkluderas manuellt.
#
# Lönegap och Gini analyseras separat i den explorativa delen.
# Sjukvård och bistånd har för låg täckning.
model_data = model_data.drop(
    columns=[
        "sjukvård_andel_gdp",
        "gini",
        "lönegap",
        "bistånd_andel_bni"
    ]
)


# Ersätter resterande saknade värden med medianen.
# Detta gör att samtliga observationer kan användas
# i regressionsmodellen.
imputer = SimpleImputer(
    strategy="median"
)

model_data[
    model_data.columns
] = imputer.fit_transform(model_data)



# --------------------------------------------------
# FUNKTION FÖR VIF-ANALYS
# --------------------------------------------------

def calculate_vif(X):

    vif_df = pd.DataFrame()

    vif_df["variable"] = X.columns

    vif_df["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    vif_df = vif_df[
        vif_df["variable"] != "const"
    ]

    return (
        vif_df
        .sort_values(
            "VIF",
            ascending=False
        )
    )



# --------------------------------------------------
# INITIAL MODELL
# Deskriptiv modell:
# Vilka samhällsindikatorer samvarierar med demokrati?
# --------------------------------------------------

y = model_data[
    "demokratiindex"
]


X = model_data.drop(
    columns=[
        "demokratiindex"
    ]
)


X = sm.add_constant(X)


model_initial = sm.OLS(
    y,
    X
).fit(
    cov_type="HC3"
)


print(
    model_initial.summary()
)



# --------------------------------------------------
# MULTIKOLLINEARITET
# --------------------------------------------------

vif_initial = calculate_vif(
    X
)

print(vif_initial)



# HDI tas bort eftersom det är ett sammansatt index
# av bland annat utbildning, hälsa och inkomst.
# Detta skapar konceptuell överlappning med flera
# andra prediktorer.
X = X.drop(
    columns=[
        "hdi"
    ]
)


vif_after_hdi = calculate_vif(
    X
)

print(vif_after_hdi)



# --------------------------------------------------
# REDUCERING AV MODELL
#
# Variabler utan statistiskt stöd tas bort stegvis.
# Syftet är att skapa en mer parsimonisk modell.
# --------------------------------------------------

variables_to_remove = [
    "mord_percapita",
    "död_i_konflikt_percapita",
    "skolår",
    "livslängd",
    "utbildning_andel_gdp",
    "suicid/100k",
    "skatt_andel_bnp",
    "gdp_per_capita",
    "andel_kvinnor_arbete",
    "handel_andel_gdp",
    "fetma_andel",
    "energi_percapita"
]


X_reduced = X.drop(
    columns=variables_to_remove
)


model_final = sm.OLS(
    y,
    X_reduced
).fit(
    cov_type="HC3"
)


print(
    model_final.summary()
)



# --------------------------------------------------
# SLUTLIG VIF-KONTROLL
# --------------------------------------------------

vif_final = calculate_vif(
    X_reduced
)

print(vif_final)

    # Betydligt bättre VIF-värden, vilket indikerar att variablerna är relativt oberoende av varandra.



# --------------------------------------------------
# SPARA RESULTAT
# --------------------------------------------------

regression_results = {
    "initial_model": model_initial,
    "final_model": model_final,
    "vif_initial": vif_initial,
    "vif_final": vif_final
}

    # "En initial modell inkluderade ett större antal samhällsindikatorer och användes deskriptivt för att identifiera potentiella samband. Därefter reducerades modellen genom att exkludera variabler utan statistiskt stöd samt indikatorer med stark konceptuell överlappning. Den reducerade modellen används för att identifiera variabler som uppvisar samband med demokratiindex när övriga inkluderade faktorer hålls konstanta."

    # Starkast association med demokratiindex, i ordning: Korruptionsindex (negativ korrelation), livstillfredsställelse, barn per kvinna, co2-utsläpp per capita, statliga utgifter som andel av BNP.



