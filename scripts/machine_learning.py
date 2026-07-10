
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



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
# DATAFÖRBEREDELSE
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

# -----------------------------
# Målvariabel
# -----------------------------

y = model_data["demokrati"]

X = model_data.drop(columns="demokrati")


# -----------------------------
# Exkludera variabler med >25 % bortfall
# -----------------------------

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

X = X.loc[:, missing_share <= 0.25]


# -----------------------------
# Imputera återstående NaN
# -----------------------------

imputer = SimpleImputer(strategy="median")


# -----------------------------
# Train/Test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=1
)


# -----------------------------
# Imputering
# -----------------------------

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

  # LOGISTISK REGRESSION #

  # Gör GridSearch utifrån olika parametrar
C_values = range(1,11)
parameters = {'penalty': ['l1', 'l2'], 'C': C_values}

  # Skapa modellen
lr = LogisticRegression(solver='liblinear', max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
logreg_grid = GridSearchCV(lr, parameters, cv=cv)
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


  # DECISION TREE #

tree = DecisionTreeClassifier()
parameters = {'min_samples_split': [2,3,4], 'max_depth': [2,3,4]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

tree_grid = GridSearchCV(tree, parameters, cv=cv)
tree_grid.fit(X_train, y_train)

tree_results = evaluate_model(
    tree_grid.best_estimator_,
    X_test,
    y_test
)



# ADAPTIVE BOOSTING TREE CLASSIFIER # 

decision_stump = DecisionTreeClassifier(max_depth = 1)
ada_clf = AdaBoostClassifier(decision_stump, n_estimators = 5)
ada_clf.fit(X_train, y_train)

ada_results = evaluate_model(
    ada_clf,
    X_test,
    y_test
)


 # GRADIENT BOOSTING TREE CLASSIFIER # 
grad_clf = GradientBoostingClassifier(n_estimators = 15)
grad_clf.fit(X_train, y_train)

grad_results = evaluate_model(
    grad_clf,
    X_test,
    y_test
)

grad_importance = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": grad_clf.feature_importances_
    })
    .sort_values("importance", ascending=False)
)



  # KNN #
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(1,15)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

knn_grid = GridSearchCV(knn, parameters, cv=cv)
knn_grid.fit(X_train, y_train)

knn_results = evaluate_model(
    knn_grid.best_estimator_,
    X_test,
    y_test
)


  # RANDOM FOREST #

rfc = RandomForestClassifier()
parameters = {'min_samples_split': [2,3,4], 'max_depth': range(5,13)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

forest_grid = GridSearchCV(rfc, parameters, cv=cv)
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
level_0_estimators["logreg"] = LogisticRegression(random_state=3)
level_0_estimators["forest"] = RandomForestClassifier(random_state=3)

level_1_estimator = RandomForestClassifier(random_state=3)

  # KFold för att träna modellen med mindre samples av träningsdatan
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
stacking_clf = StackingClassifier(estimators=list(level_0_estimators.items()), 
                                    final_estimator=level_1_estimator, 
                                    passthrough=True, cv=kfold, stack_method="predict_proba")


stacking_clf.fit(X_train, y_train)

stacking_results = evaluate_model(
    stacking_clf,
    X_test,
    y_test
)



# -----------------------------
# Resultat att exportera till rapporten
# -----------------------------

    # Baseline att utgå från och att jämföra modellerna med
baseline_accuracy = y.value_counts(normalize=True).max()

    # Bästa modellparametrarna
model_parameters = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Random Forest"
    ],
    "Best parameters": [
        logreg_grid.best_params_,
        tree_grid.best_params_,
        knn_grid.best_params_,
        forest_grid.best_params_
    ],
    "CV score": [
        logreg_grid.best_score_,
        tree_grid.best_score_,
        knn_grid.best_score_,
        forest_grid.best_score_
    ]
}).round(3)

    # Summera alla resultat
results = {
    "Logistic Regression": logreg_results,
    "Decision Tree": tree_results,
    "Random Forest": forest_results,
    "KNN": knn_results,
    "AdaBoost": ada_results,
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
