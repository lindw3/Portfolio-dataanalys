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
        figsize = (8, 6)
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

    plt.show()


# Fynd från corrplot: Ökad gini-koefficient = minskat lönegap.
# Fynd från corrplot: Ökad GDP per capita = lägre gini
# Fynd från corrplot: Ökad GDP per capita = ökat lönegap.
# Undersök dessa samband!

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


# --------------------------------------------------
# 4. GDP per capita vs andel kvinnor i arbete som andel av män
# Hypotes: kvinnors deltagande i arbetsmarknaden förändras när man blir medelinkomstland,
# där det främst blir högkvalificerad kvinnlig arbetskraft som deltar på arbetsmarknaden
# vilket är en förklaring till att lönegapet sedan ökar när landet blir rikare och kvinnor i gemen
# blir en del av arbetskraften i den mer "avancerade" arbetsmarknaden.
# --------------------------------------------------


gdp_andel_kvinnor_arbete = (
    data[
        [
            "land",
            "år",
            "gdp_per_capita",
            "andel_kvinnor_arbete"
        ]
    ]
    .dropna()
)

gdp_andel_kvinnor_arbete = gdp_andel_kvinnor_arbete[
    gdp_andel_kvinnor_arbete["år"] == 2022
]


def plot_gdp_vs_andel_kvinnor_arbete(gdp_andel_kvinnor_arbete):

    points = (
        alt.Chart(gdp_andel_kvinnor_arbete)
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
                "andel_kvinnor_arbete:Q",
                title="Andel kvinnor i arbete (%)",
                scale=alt.Scale(
                    domain=[0, 100]
                    ),
                    axis=alt.Axis(
                        values=[0, 20, 40, 60, 80, 100],
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
                    "andel_kvinnor_arbete:Q",
                    title="Andel kvinnor i arbete (%)",
                    format=".1f"
                    )
            ]
        )
    )

    return (
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

