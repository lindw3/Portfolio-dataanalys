import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

data = pd.read_parquet(
    "data/combined_dataset.parquet"
)

# --------------------------------------------------
# Explorativ analys av indikatorer
# --------------------------------------------------


# --------------------------------------------------
# Korrelationsanalys
# --------------------------------------------------

# Begränsar analysen till 2021 eftersom de flesta
# indikatorer har relativt hög täckning för detta år.
corr_data = data[
    data["år"] == 2021
].copy()

corr_data = corr_data.drop(
    columns=[
        "land",
        "år"
    ]
)

def create_corrplot(corr_data):

    corr = corr_data.corr()

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True
    )

    plt.title(
        "Correlation Matrix"
    )

    return plt.gcf()


# Fynd från corrplot: Ökad gini-koefficient = minskat lönegap.


# --------------------------------------------------
# 1. Gini vs lönegap
# Undersöker sambandet mellan ekonomisk ojämlikhet
# och könsskillnader i löner.
# --------------------------------------------------

lönegap_gini = (
    data[
        [
            "land",
            "år",
            "lönegap",
            "gini"
        ]
    ]
    .dropna()
)

lönegap_gini = lönegap_gini[
    lönegap_gini["år"] == 2021
]


def plot_gini_vs_lönegap(lönegap_gini):

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
                    format=".2f"
                ),

                alt.Tooltip(
                    "lönegap:Q",
                    title="Lönegap, %",
                    format=".1f"
                )
            ]
        )
    )


    zero_line = (
        alt.Chart(
            pd.DataFrame({"y": [0]})
        )
        .mark_rule(
            strokeWidth=2,
            color="black"
        )
        .encode(
            y="y:Q"
        )
    )


    return (
        (zero_line + points)
        .properties(
            width=850,
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



# --------------------------------------------------
# 2. GDP per capita vs Gini
# Undersöker sambandet mellan utvecklingsnivå
# och ekonomisk ojämlikhet.
# --------------------------------------------------

gdp_gini = (
    data[
        [
            "land",
            "år",
            "gdp_per_capita",
            "gini"
        ]
    ]
    .dropna()
)

gdp_gini = gdp_gini[
    gdp_gini["år"] == 2021
]


def plot_gdp_vs_gini(gdp_gini):

    points = (
        alt.Chart(gdp_gini)
        .mark_circle(size=70)
        .encode(

            x=alt.X(
                "gdp_per_capita:Q",
                title="GDP per capita ($), logaritmisk skala",
                scale=alt.Scale(
                    type="log"
                ),
                axis=alt.Axis(
                    values=[
                        1000,
                        2500,
                        5000,
                        10000,
                        25000,
                        50000
                    ],
                    format=",.0f"
                )
            ),

            y=alt.Y(
                "gini:Q",
                title="Gini-koefficient",
                scale=alt.Scale(
                    domain=[0.2, 0.6]
                ),
                axis=alt.Axis(
                    format=".2f"
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "land:N",
                    title="Land"
                ),

                alt.Tooltip(
                    "gdp_per_capita:Q",
                    title="GDP per capita, $",
                    format=",.0f"
                ),

                alt.Tooltip(
                    "gini:Q",
                    title="Gini-koefficient",
                    format=".2f"
                )
            ]
        )
    )


    return (
        points
        .properties(
            width=850,
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



# --------------------------------------------------
# 3. GDP per capita vs lönegap
# Undersöker om utvecklingsnivå hänger ihop
# med könsskillnader i löner.
# --------------------------------------------------

lönegap_gdp = (
    data[
        [
            "land",
            "år",
            "lönegap",
            "gdp_per_capita"
        ]
    ]
    .dropna()
)

lönegap_gdp = lönegap_gdp[
    lönegap_gdp["år"] == 2022
]


def plot_gdp_vs_lönegap(lönegap_gdp):

    points = (
        alt.Chart(lönegap_gdp)
        .mark_circle(size=70)
        .encode(

            x=alt.X(
                "gdp_per_capita:Q",
                title="GDP per capita ($), logaritmisk skala",
                scale=alt.Scale(
                    type="log"
                ),
                axis=alt.Axis(
                    values=[
                        1000,
                        2500,
                        5000,
                        10000,
                        25000,
                        50000
                    ],
                    format=",.0f"
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
                    "gdp_per_capita:Q",
                    title="GDP per capita, $",
                    format=",.0f"
                ),

                alt.Tooltip(
                    "lönegap:Q",
                    title="Lönegap, %",
                    format=".1f"
                )
            ]
        )
    )


    zero_line = (
        alt.Chart(
            pd.DataFrame({"y": [0]})
        )
        .mark_rule(
            strokeWidth=2,
            color="black"
        )
        .encode(
            y="y:Q"
        )
    )


    return (
        (zero_line + points)
        .properties(
            width=850,
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
