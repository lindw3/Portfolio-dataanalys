
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from pathlib import Path


    # Funktion för att evaluera modeller
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
            ),
        "recall": recall_score(
            y_test, 
            y_pred,
            zero_division=0
            ),
        "f1": f1_score(
            y_test, 
            y_pred,
            zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
    }


# --------------------------------------------------
# LADDA DATA
# --------------------------------------------------

demokrati = pd.read_csv("https://ourworldindata.org/explorers/democracy.csv?v=1&csvType=full&useColumnShortNames=true&Dataset=Regimes+of+the+World&Metric=%C2%ADPolitical+regime&Sub-metric=Main+classification", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
demokrati.columns = ['land', 'kod', 'år', 'demokrati']
demokrati = demokrati[['land', 'år', 'demokrati']]
demokrati = demokrati[demokrati["år"] == 2021]

# Gör om demokrati till en binär variabel. 2-3 = demokrati och 0-1 = autokrati.
demokrati["demokrati"] = (demokrati["demokrati"] >= 2).astype(int)

data = pd.read_parquet(
    "data/combined_dataset.parquet"
)



# --------------------------------------------------
# DATABEARBETNING
# --------------------------------------------------


    # Begränsar analysen till 2021 eftersom de flesta
    # indikatorer har relativt hög täckning detta år.
model_data = data[
    data["år"] == 2021
]

    # Slå ihop databasen med demokrati-data
model_data = model_data.merge(
    demokrati[["land", "år", "demokrati"]],
    on=["land", "år"],
    how="left"
)

    # Ta bort observationer där målvariabeln saknas
model_data = model_data.dropna(subset=["demokrati"])


    # Land och år används inte som prediktorer.
    # Exkludera även demokratiindex, som är ett mått på länders demokratiska funktioner
    # Exkludera även HDI då det är en "samlingsvariabel" för utbildning, livslängd och GDP per capita
model_data = model_data.drop(
    columns=[
        "land",
        "år",
        "demokratiindex",
        "hdi"
    ]
)


    # Tar bort variabler som helt saknar data.
model_data = model_data.dropna(
    axis=1,
    how="all"
)

    # Målvariabel
y = model_data["demokrati"]

X = model_data.drop(columns="demokrati")


    # Exkludera variabler med >25 % bortfall

missing_share = X.isna().mean()

variables_removed = missing_share[missing_share > 0.25].index.tolist()

removed_variables = (
    pd.DataFrame({
        "variable": variables_removed,
        "missing_share": missing_share[variables_removed]
    })
    .assign(
        missing_share=lambda x: (x["missing_share"] * 100).round(1)
    )
)
y
X = X.loc[:, missing_share <= 0.25]


    # Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=3
)


    # Imputering
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

    # Standardisering av data
scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)




# -----------------------------
# Testa modellerna
# -----------------------------

  # LOGISTISK REGRESSION

  # Gör GridSearch utifrån olika parametrar
C = np.logspace(-3, 2, 15)
parameters = {
    'C': C,
    "penalty": ["l1", "l2"]
    }

  # Skapa modellen
lr = LogisticRegression(solver='liblinear', max_iter=1000, random_state=3)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
logreg_grid = GridSearchCV(lr, parameters, cv=cv, scoring="accuracy")
logreg_grid.fit(X_train, y_train)

logreg_results = evaluate_model(
    logreg_grid.best_estimator_,
    X_test,
    y_test
)

logreg_coefficients = (
    pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": logreg_grid.best_estimator_.coef_[0]
    })
    .assign(abs_coef=lambda x: x.coefficient.abs())
    .sort_values("abs_coef", ascending=False)
)



    # GRADIENT BOOSTING TREE CLASSIFIER
grad_clf = GradientBoostingClassifier(n_estimators = 15, random_state=3)

grad_clf.fit(X_train, y_train)

parameters = {
    'n_estimators': [25, 50, 100],
    'learning_rate': [0.01, 0.1, 0.5, 1]
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

grad_grid = GridSearchCV(grad_clf, parameters, cv=cv, scoring="accuracy")

grad_grid.fit(X_train, y_train)

grad_grid.cv_results_

grad_results = evaluate_model(
    grad_grid.best_estimator_,
    X_test,
    y_test
)

grad_importance = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": grad_grid.best_estimator_.feature_importances_
    })
    .sort_values("importance", ascending=False)
)



    # KNN
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(1,31),
              'weights': ["uniform", "distance"],
              'metric': ["euclidean", "manhattan"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

knn_grid = GridSearchCV(knn, parameters, cv=cv, scoring="accuracy")
knn_grid.fit(X_train, y_train)

knn_results = evaluate_model(
    knn_grid.best_estimator_,
    X_test,
    y_test
)

    # RANDOM FOREST

rfc = RandomForestClassifier(random_state=3)
parameters = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

forest_grid = GridSearchCV(rfc, parameters, cv=cv, scoring="accuracy")
forest_grid.fit(X_train, y_train)

forest_results = evaluate_model(
    forest_grid.best_estimator_,
    X_test,
    y_test
)

forest_importance = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": forest_grid.best_estimator_.feature_importances_
    })
    .sort_values("importance", ascending=False)
)

 
    # STACKING CLASSIFIER: Logistisk Regression och Random Forest #

    # Skapa två nivåer: Först mix av LogReg och RFC för att träna modellen, sedan görs prediktionen med random forest
level_0_estimators = dict()
    # Använd de tidigare bästa estimerarna från logreg och forest
level_0_estimators["logreg"] = logreg_grid.best_estimator_
level_0_estimators["forest"] = forest_grid.best_estimator_
level_1_estimator = RandomForestClassifier(random_state=3)

  # KFold för att träna modellen med mindre samples av träningsdatan
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
stacking_clf = StackingClassifier(estimators=list(level_0_estimators.items()), 
                                    final_estimator=level_1_estimator, 
                                    passthrough=True, cv=kfold, stack_method="predict_proba")

stacking_clf.fit(X_train, y_train)

stacking_results = evaluate_model(
    stacking_clf,
    X_test,
    y_test
)

stacking_cv_scores = cross_val_score(
    stacking_clf,
    X_train,
    y_train,
    cv=kfold,
    scoring="accuracy"
)

stacking_cv_mean = stacking_cv_scores.mean()
stacking_cv_std = stacking_cv_scores.std()


# -----------------------------
# Resultat att exportera till rapporten
# -----------------------------

    # Baseline att utgå från och att jämföra modellerna med
baseline_accuracy = y.value_counts(normalize=True).max()

    # Bästa modellparametrarna
model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "Gradient Boost",
        "Stacking"
    ],
    "Mean CV accuracy": [
        logreg_grid.best_score_,
        knn_grid.best_score_,
        forest_grid.best_score_,
        grad_grid.best_score_,
        stacking_cv_mean
    ]
}).round(3)

    # Summera alla resultat
results = {
    "Logistic Regression": logreg_results,
    "Random Forest": forest_results,
    "KNN": knn_results,
    "Gradient Boosting": grad_results,
    "Stacking": stacking_results
}

    # Skapa en summering av resultat som kan importeras i .qmd-filen
model_summary = (
    pd.DataFrame(results)
    .T[["accuracy", "precision", "recall", "f1"]]
    .rename(columns={
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1"
    })
    .round(3)
)

    # Bästa modell: logistisk regression
    # Kan bero på mer linjära samband generellt sett
    # Kan bero på litet underlag där det är svårt att fånga upp komplexa samband - därför vinner den enklare modellen



# -----------------------------
# Modellförenkling med RFE
# Görs på den bästa modellen (logistisk regression)
# -----------------------------

feature_results = []

for n_features in range(3, X_train.shape[1] + 1):

    # RFE
    rfe = RFE(
        estimator=LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=3
        ),
        n_features_to_select=n_features
    )

    rfe.fit(
        X_train,
        y_train
    )

    selected_features = X_train.columns[rfe.support_]

    X_train_rfe = X_train[selected_features]


    # Modell
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=3
    )


    # Cross-validation
    cv_scores = cross_val_score(
        model,
        X_train_rfe,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    feature_results.append({
        "n_features": n_features,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "features": list(selected_features)
    })


rfe_summary = pd.DataFrame(feature_results)

    # Spara de variabler som inkluderades i den reducerade modellen
best_rfe_features = (
    rfe_summary
    .loc[
        rfe_summary["cv_accuracy_mean"].idxmax(),
        "features"
    ]
)


    # Skapa en reducerad modell utifrån de variabler som återstår efter RFE-analysen

X_train_final = X_train[best_rfe_features]
X_test_final = X_test[best_rfe_features]

C = np.logspace(-3, 2, 15)
parameters = {
    'C': C,
    "penalty": ["l1", "l2"]
    }

lr = LogisticRegression(solver='liblinear', max_iter=1000, random_state=3)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

logreg_grid_final = GridSearchCV(lr, parameters, cv=cv, scoring="accuracy")
logreg_grid_final.fit(X_train_final, y_train)

logreg_results_final = evaluate_model(
    logreg_grid_final.best_estimator_,
    X_test_final,
    y_test
)

logreg_final_summary = (
    pd.DataFrame({
        "Model": ["Logistic Regression"],
        "Accuracy": [logreg_results_final["accuracy"]],
        "Precision": [logreg_results_final["precision"]],
        "Recall": [logreg_results_final["recall"]],
        "F1": [logreg_results_final["f1"]]
    })
    .round(3)
)

logreg_coefficients_final = (
    pd.DataFrame({
        "feature": X_train_final.columns,
        "coefficient": logreg_grid_final.best_estimator_.coef_[0]
    })
    .assign(abs_coef=lambda x: x.coefficient.abs())
    .sort_values("abs_coef", ascending=False)
)


# --------------------------------------------------
# EXPORTERA RESULTAT TILL CSV-FILER
# --------------------------------------------------

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd()

results_dir = project_root / "data" / "results_ml"
results_dir.mkdir(parents=True, exist_ok=True)

exports = {
    "model_summary.csv": (
        model_summary.reset_index()
        .rename(columns={"index": "model"})
    ),
    "model_comparison.csv": model_comparison,
    "removed_variables.csv": removed_variables,
    "rfe_summary.csv": rfe_summary,
    "best_rfe_features.csv": pd.DataFrame({"best_rfe_features": best_rfe_features}),
    "logreg_coefficients.csv": logreg_coefficients,
    "gradient_boosting_importance.csv": grad_importance,
    "random_forest_importance.csv": forest_importance,
    "logreg_coefficients_final.csv": logreg_coefficients_final,
    "logreg_results_final.csv": pd.DataFrame({"logreg_results_final": [logreg_results_final]}),
    "logreg_final_summary.csv": logreg_final_summary,
    "baseline_accuracy.csv": pd.DataFrame({"baseline_accuracy": [baseline_accuracy]})
}

for filename, dataframe in exports.items():
    dataframe.to_csv(results_dir / filename, index=False)

