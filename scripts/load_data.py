import pandas as pd


OWID_HEADERS = {
    "storage_options": {
        "User-Agent": "Our World In Data data fetch/1.0"
    }
}


def load_data():

    datasets = []


    # Livslängd
    livslängd = pd.read_csv(
        "https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    livslängd.columns = ["land", "kod", "år", "livslängd"]
    livslängd = livslängd[["land", "år", "livslängd"]]
    datasets.append(livslängd)


    # Suicid
    suicid = pd.read_csv(
        "https://ourworldindata.org/grapher/death-rate-from-suicides-gho.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    suicid.columns = ["land", "kod", "år", "suicid/100k"]
    suicid = suicid[["land", "år", "suicid/100k"]]
    datasets.append(suicid)


    # Fetma
    fetma = pd.read_csv(
        "https://ourworldindata.org/grapher/share-of-adults-defined-as-obese.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    fetma.columns = ["land", "kod", "år", "fetma_andel", "region"]
    fetma = fetma[["land", "år", "fetma_andel"]]
    datasets.append(fetma)


    # HDI
    hdi = pd.read_csv(
        "https://ourworldindata.org/grapher/human-development-index.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    hdi.columns = ["land", "kod", "år", "hdi", "region"]
    hdi = hdi[["land", "år", "hdi"]]
    datasets.append(hdi)


    # Demokratiindex
    demokrati = pd.read_csv(
        "data/electoral-democracy-index.csv"
    )
    demokrati.columns = [
        "land",
        "år",
        "demokratiindex",
        "region"
    ]
    demokrati = demokrati[["land", "år", "demokratiindex"]]
    datasets.append(demokrati)


    # CO2
    co2 = pd.read_csv(
        "https://ourworldindata.org/grapher/co-emissions-per-capita.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    co2.columns = ["land", "kod", "år", "co2_percapita"]
    co2 = co2[["land", "år", "co2_percapita"]]
    datasets.append(co2)


    # Energi
    energi = pd.read_csv(
        "data/energy-use-per-person.csv"
    )
    energi.columns = [
        "land",
        "år",
        "energi_percapita"
    ]
    energi = energi[["land", "år", "energi_percapita"]]
    datasets.append(energi)


    # Utbildning
    utbildning = pd.read_csv(
        "https://ourworldindata.org/grapher/total-government-expenditure-on-education-gdp.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    utbildning.columns = [
        "land",
        "kod",
        "år",
        "utbildning_andel_gdp"
    ]
    utbildning = utbildning[["land", "år", "utbildning_andel_gdp"]]
    datasets.append(utbildning)


    # Sjukvård
    sjukvård = pd.read_csv(
        "https://ourworldindata.org/grapher/public-health-expenditure-share-gdp.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    sjukvård.columns = [
        "land",
        "kod",
        "år",
        "sjukvård_andel_gdp"
    ]
    sjukvård = sjukvård[["land", "år", "sjukvård_andel_gdp"]]
    datasets.append(sjukvård)


    # Lönegap
    lönegap = pd.read_csv(
        "https://ourworldindata.org/grapher/gender-gap-in-average-wages-ilo.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    lönegap.columns = [
        "land",
        "kod",
        "år",
        "lönegap"
    ]
    lönegap = lönegap[["land", "år", "lönegap"]]
    datasets.append(lönegap)


    # Kvinnors arbetskraftsdeltagande relativt män
    andel_kvinnor_arbete = pd.read_csv(
        "https://ourworldindata.org/grapher/ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    andel_kvinnor_arbete.columns = [
        "land",
        "kod",
        "år",
        "andel_kvinnor_arbete",
        "region"
    ]
    andel_kvinnor_arbete = andel_kvinnor_arbete[
        ["land", "år", "andel_kvinnor_arbete"]
    ]
    datasets.append(andel_kvinnor_arbete)


    # Skolår
    skolår = pd.read_csv(
        "https://ourworldindata.org/grapher/average-years-of-schooling.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    skolår.columns = [
        "land",
        "kod",
        "år",
        "skolår"
    ]
    skolår = skolår[["land", "år", "skolår"]]
    datasets.append(skolår)


    # Bistånd
    bistånd = pd.read_csv(
        "https://ourworldindata.org/grapher/foreign-aid-given-as-a-share-of-national-income.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    bistånd.columns = [
        "land",
        "kod",
        "år",
        "bistånd_andel_bni",
        "annotation"
    ]
    bistånd = bistånd[
        ["land", "år", "bistånd_andel_bni"]
    ]
    datasets.append(bistånd)


    # Skatt
    skatt = pd.read_csv(
        "https://ourworldindata.org/grapher/tax-revenues-as-a-share-of-gdp-unu-wider.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    skatt.columns = [
        "land",
        "kod",
        "år",
        "skatt_andel_bnp",
        "region"
    ]
    skatt = skatt[
        ["land", "år", "skatt_andel_bnp"]
    ]
    datasets.append(skatt)


    # Statliga utgifter
    statligautgifter = pd.read_csv(
        "https://ourworldindata.org/grapher/historical-gov-spending-gdp.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    statligautgifter.columns = [
        "land",
        "kod",
        "år",
        "statligautgifter_andel_bnp",
        "annotation"
    ]
    statligautgifter = statligautgifter[
        ["land", "år", "statligautgifter_andel_bnp"]
    ]
    datasets.append(statligautgifter)


    # Gini
    gini = pd.read_excel(
        "data/gini.xlsx"
    )
    gini.columns = [
        "land",
        "år",
        "gini"
    ]
    datasets.append(gini)


    # GDP per capita
    gdp = pd.read_csv(
        "https://ourworldindata.org/grapher/gdp-per-capita-maddison-project-database.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    gdp.columns = [
        "land",
        "kod",
        "år",
        "gdp_per_capita",
        "annotation"
    ]
    gdp = gdp[
        ["land", "år", "gdp_per_capita"]
    ]
    datasets.append(gdp)


    # Handel
    handel = pd.read_csv(
        "https://ourworldindata.org/grapher/trade-as-share-of-gdp.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    handel.columns = [
        "land",
        "kod",
        "år",
        "handel_andel_gdp"
    ]
    handel = handel[
        ["land", "år", "handel_andel_gdp"]
    ]
    datasets.append(handel)


    # Livstillfredsställelse
    livstillfredsställelse = pd.read_csv(
        "https://ourworldindata.org/grapher/happiness-cantril-ladder.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    livstillfredsställelse.columns = [
        "land",
        "kod",
        "år",
        "livstillfredsställelse"
    ]
    livstillfredsställelse = livstillfredsställelse[
        ["land", "år", "livstillfredsställelse"]
    ]
    datasets.append(livstillfredsställelse)


    # Enpersonshushåll
    hushåll = pd.read_csv(
        "https://ourworldindata.org/grapher/one-person-households.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    hushåll.columns = [
        "land",
        "kod",
        "år",
        "hushåll_andel"
    ]
    hushåll = hushåll[
        ["land", "år", "hushåll_andel"]
    ]
    datasets.append(hushåll)


    # Barn per kvinna
    barn = pd.read_csv(
        "https://ourworldindata.org/grapher/children-born-per-woman.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    barn.columns = [
        "land",
        "kod",
        "år",
        "barn_per_kvinna"
    ]
    barn = barn[
        ["land", "år", "barn_per_kvinna"]
    ]
    datasets.append(barn)


    # Korruption
    korruption = pd.read_csv(
        "https://ourworldindata.org/grapher/political-corruption-index.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    korruption.columns = [
        "land",
        "kod",
        "år",
        "korruption_index",
        "region"
    ]
    korruption = korruption[
        ["land", "år", "korruption_index"]
    ]
    datasets.append(korruption)


    # Mord
    mord = pd.read_csv(
        "https://ourworldindata.org/grapher/homicide-rate-ghe.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    mord.columns = [
        "land",
        "kod",
        "år",
        "mord_percapita",
        "region"
    ]
    mord = mord[
        ["land", "år", "mord_percapita"]
    ]
    datasets.append(mord)


    # Död i väpnad konflikt
    död_konflikt = pd.read_csv(
        "https://ourworldindata.org/grapher/deaths-in-armed-conflicts.csv?v=1&csvType=full&useColumnShortNames=true",
        **OWID_HEADERS
    )
    död_konflikt.columns = [
        "land",
        "kod",
        "år",
        "död_i_konflikt_percapita_hög",
        "död_i_konflikt_percapita",
        "död_i_konflikt_percapita_låg"
    ]
    död_konflikt = död_konflikt[
        ["land", "år", "död_i_konflikt_percapita"]
    ]
    datasets.append(död_konflikt)


    return datasets