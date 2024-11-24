from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
from . import eval

def optimize_dtc(X_train, y_train, X_test, y_test):
    """
class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, monotonic_cst=None)
    """
    def objective(trial):
        param_grid = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'max_depth': trial.suggest_int('max_depth', 2, 1000),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 1000),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1000),
            'random_state': trial.suggest_int('random_state', 1, 100000),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        }

        model = DecisionTreeClassifier(
            criterion=param_grid['criterion'],
            max_depth=param_grid['max_depth'],
            min_samples_split=param_grid['min_samples_split'],
            min_samples_leaf=param_grid['min_samples_leaf'],
            random_state=param_grid['random_state'],
            splitter=param_grid['splitter']
        )

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)

    return study.best_params

def optimize_random_forest(X_train, y_train, X_test, y_test):
    def objective(trial):
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 1000),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 500),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1000),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_features': trial.suggest_int('n_features', 1, 30),
            'random_state': trial.suggest_int('random_state', 1, 100000),
        }

        model = RandomForestClassifier(n_estimators=param_grid['n_estimators'],
            max_depth=param_grid['max_depth'],
            min_samples_split=param_grid['min_samples_split'],
            min_samples_leaf=param_grid['min_samples_leaf'],
            max_features=param_grid['max_features']
        )

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=500)

    return study.best_params

def optimize_naive_bayes(X_train, y_train, X_test, y_test):
    def objective(trial):
        param_grid = {
            'alpha': trial.suggest_discrete_uniform('alpha', 0.1, 1.0, 0.1),
            'fit_prior': trial.suggest_categorical('fit_prior', [True, False]),
        }
        model = GaussianNB(alpha=param_grid['alpha'], fit_prior=param_grid['fit_prior'])

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def optimize_svm(X_train, y_train, X_test, y_test):
    def objective(trial):
        param_grid = {
            'C': trial.suggest_discrete_uniform('C', 0.1, 1.0, 0.1),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        }
        model = SVC(C=param_grid['C'], kernel=param_grid['kernel'])

        model.fit(X_train, y_train)        
        y_pred = model.predict_proba(X_test)
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def optimize_mlp(X_train, y_train, X_test, y_test):
    def objective(trial):
        param_grid = {
            # Instead of trial.suggest_int ,we need a tuple with ints
            'hidden_layer_sizes': (trial.suggest_int('hidden_layer_sizes', 10, 1000),trial.suggest_int('hidden_layer_sizes', 10, 1000)),
            'alpha': trial.suggest_discrete_uniform('alpha', 0.1, 1.0, 0.1),
            'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'max_iter': trial.suggest_int('max_iter', 100, 5000),
        }
        model = MLPClassifier(
            hidden_layer_sizes=param_grid['hidden_layer_sizes'],
            alpha=param_grid['alpha'],
            activation=param_grid['activation'],
            solver=param_grid['solver'],
            max_iter=param_grid['max_iter']
        )

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)
    return study.best_params

def optimize_logistic_regression(X_train, y_train, X_test, y_test):
    def objective(trial):
        param_grid = {
            'C': trial.suggest_discrete_uniform('C', 0.1, 1.0, 0.1),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
        }
        model = LogisticRegression(C=param_grid['C'], solver=param_grid['solver'])

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)        
        return eval.error_eval(y_test, y_pred)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def optimize_knn(X_train, y_train, X_test, y_test):
    def objective(trial):
        param_grid = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 100),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        }
        model = KNeighborsClassifier(n_neighbors=param_grid['n_neighbors'],
            weights=param_grid['weights'],
            algorithm=param_grid['algorithm']
        )

        model.fit(X_train, y_train)        
        y_pred = model.predict_proba(X_test)        
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def optimize_adaboost(X_train, y_train, X_test, y_test):    
    def objective(param_grid):
        model = AdaBoostClassifier(n_estimators=param_grid['n_estimators'],
            learning_rate=param_grid['learning_rate'],
            algorithm=param_grid['algorithm']
        )

        model.fit(X_train, y_train)        
        y_pred = model.predict_proba(X_test)        
        return eval.error_eval(y_test, y_pred)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def optimize_extra_trees(X_train, y_train, X_test, y_test):
    def objective(param_grid):
        model = ExtraTreesClassifier(n_estimators=param_grid['n_estimators'],
            max_depth=param_grid['max_depth'],
            min_samples_split=param_grid['min_samples_split'],
            min_samples_leaf=param_grid['min_samples_leaf'],
            max_features=param_grid['max_features']
        )

        model.fit(X_train, y_train)        
        y_pred = model.predict_proba(X_test)       
        return eval.error_eval(y_test, y_pred)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    return study.best_params

def optimize_gradient_boosting(X_train, y_train, X_test, y_test):
    def objective(param_grid):
        model = GradientBoostingClassifier(n_estimators=param_grid['n_estimators'],
            learning_rate=param_grid['learning_rate'],
            max_depth=param_grid['max_depth'],
            min_samples_split=param_grid['min_samples_split'],
            min_samples_leaf=param_grid['min_samples_leaf'],
            max_features=param_grid['max_features'],
            loss=param_grid['loss'],
            subsample=param_grid['subsample'],
            random_state=param_grid['random_state']
        )

        model.fit(X_train, y_train)        
        y_pred = model.predict_proba(X_test)        
        return eval.error_eval(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)        
    return study.best_params