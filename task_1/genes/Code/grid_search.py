from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Declaring what parameters to be used in all three grid searches
# In some cases, the parameters were reduced from previous runs to reduce computational time
param_grid_forest = { 
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2'],# 'auto' is deprecated, so it is not used
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }

param_grid_logreg = {
    'max_iter': [50, 100, 250, 500, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'metric': ['minkowski', 'cosine'],# 'euclidean', 'manhattan' also used in previous runs
    'weights': ['uniform', 'distance']
}

def grid_search_forest(x_train, y_train):
    model=RandomForestClassifier(random_state=12, n_jobs=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_forest, cv= 5, verbose=1)
    grid_search.fit(x_train, y_train)

    print(grid_search.best_params_)
    return grid_search.best_params_

def grid_search_logreg(x_train, y_train):
    model=LogisticRegression(random_state=12, n_jobs=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_logreg, cv= 5, verbose=1)
    grid_search.fit(x_train, y_train)

    print(grid_search.best_params_)
    return grid_search.best_params_

def grid_search_KNN(x_train, y_train):
    model=KNeighborsClassifier(n_jobs=-1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_knn, cv= 5, verbose=1)
    grid_search.fit(x_train, y_train)

    print(grid_search.best_params_)
    return grid_search.best_params_
