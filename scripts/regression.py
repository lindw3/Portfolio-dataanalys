import pandas as pd
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


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
missing_share_rows = model_data.isna().mean(axis=1)

model_data = model_data[
    missing_share_rows <= 0.25
]

# Tar bort variabler med 25 % eller mer saknade värden.
model_data = model_data.dropna(
    axis=1,
    thresh=len(model_data) * 0.75
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

    # Standardisera X-variablerna
scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

X = X_scaled


X = sm.add_constant(X)


model_initial = sm.OLS(
    y,
    X
).fit(
    cov_type="HC3"
)


  # Skapa egna tabeller med relevanta variabler
model_info_initial = pd.DataFrame({
    "Oberoende variabel": [model_initial.model.endog_names],
    "Antal observationer": [int(model_initial.nobs)],
    "R-squared": [model_initial.rsquared],
    "Adj. R-squared": [model_initial.rsquared_adj],
    "F-statistic": [model_initial.fvalue],
    "Prob (F-statistic)": [model_initial.f_pvalue],
    "AIC": [model_initial.aic],
    "BIC": [model_initial.bic]
})

model_info_initial = model_info_initial.round(3)


coef_table_initial = pd.DataFrame({
    "Koefficient": model_initial.params,
    "Standardfel": model_initial.bse,
    "z": model_initial.tvalues,
    "P>|z|": model_initial.pvalues,
    "[0.025": model_initial.conf_int()[0],
    "0.975]": model_initial.conf_int()[1]
})

coef_table_initial = coef_table_initial.round(3)
coef_table_initial = coef_table_initial.drop(
    index="const"
)




# --------------------------------------------------
# MULTIKOLLINEARITET
# --------------------------------------------------

vif_initial = calculate_vif(
    X
)




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

  # Skapa egna tabeller med relevanta variabler
coef_table_final = pd.DataFrame({
    "Koefficient": model_final.params,
    "Standardfel": model_final.bse,
    "z": model_final.tvalues,
    "P>|z|": model_final.pvalues,
    "[0.025": model_final.conf_int()[0],
    "0.975]": model_final.conf_int()[1]
})

coef_table_final = coef_table_final.round(3)
coef_table_final = coef_table_final.drop(
    index="const"
)


model_info_final = pd.DataFrame({
    "Oberoende variabel": [model_final.model.endog_names],
    "Antal observationer": [int(model_final.nobs)],
    "R-squared": [model_final.rsquared],
    "Adj. R-squared": [model_final.rsquared_adj],
    "F-statistic": [model_final.fvalue],
    "Prob (F-statistic)": [model_final.f_pvalue],
    "AIC": [model_final.aic],
    "BIC": [model_final.bic]
})

model_info_final = model_info_final.round(3)



# --------------------------------------------------
# SLUTLIG VIF-KONTROLL
# --------------------------------------------------

vif_final = calculate_vif(
    X_reduced
)


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

    # Starkast association med demokratiindex, i ordning: Korruptionsindex (negativ korrelation), livstillfredsställelse, barn per kvinna, co2-utsläpp per capita, statliga utgifter som andel av BNP.



