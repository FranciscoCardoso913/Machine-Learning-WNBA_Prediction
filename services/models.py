from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from . import optimizer

def getting_models(X_train, y_train, X_test, y_test, optimize=False):
    models = []

    # Logistic Regression with additional parameters
    models.append((
        'LR',
        LogisticRegression(
            penalty='l2',           # Regularization type
            C=1.0,                  # Regularization strength
            solver='lbfgs',         # Solver
            max_iter=200,           # Increase iterations if convergence issues
            random_state=42         # Seed for reproducibility
        )
    ))

    if(optimize):
        dtc_params = optimizer.optimize_dtc(X_train, y_train, X_test, y_test)

        print("DTC PARAMS ARE: ", dtc_params)

        models.append((
            'DTC',
            DecisionTreeClassifier(
                criterion=dtc_params["criterion"],
                max_depth=dtc_params["max_depth"],
                min_samples_split=dtc_params["min_samples_split"],
                min_samples_leaf=dtc_params["min_samples_leaf"],
                random_state=dtc_params["random_state"],
                #min_impurity_decrease=dtc_params["min_impurity_decrease"],
                splitter=dtc_params["splitter"]
            )
        ))
    else:
        models.append(('DTC', DecisionTreeClassifier(
            criterion='gini',
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=3,
            random_state=82,
            splitter='best'
        )))

    # K-Nearest Neighbors
    models.append((
        'KNN',
        KNeighborsClassifier(
            n_neighbors=5,          # Number of neighbors
            weights='uniform',      # Weighting function: 'uniform' or 'distance'
            algorithm='auto'        # Algorithm for neighbor search
        )
    ))

    # Gaussian Naive Bayes (no additional hyperparameters needed)
    models.append(('GNB', GaussianNB()))

    # Multi-Layer Perceptron Classifier
    #mlp_params = optimizer.optimize_mlp(X_train, y_train, X_test, y_test)

    #print("MLP PARAMS ARE: ", mlp_params)

    models.append((
        'MLP',
        MLPClassifier(
            #hidden_layer_sizes=mlp_params["hidden_layer_sizes"],
            #alpha=mlp_params["alpha"],
            #activation=mlp_params["activation"],
            #solver=mlp_params["solver"],
            hidden_layer_sizes=696,  # Two layers with 100 and 50 neurons
            activation='relu',            # Activation function
            solver='lbfgs',                # Optimization algorithm
            alpha=0.1,                 # L2 penalty (regularization term)
            max_iter=3877,                 # Maximum iterations
            #random_state=42
        )
    ))

    # Random Forest Classifier
    #rc_params = optimizer.optimize_random_forest(X_train, y_train, X_test, y_test)

    #print("RC PARAMS ARE: ", rc_params)

    models.append((
        'RFC',
        RandomForestClassifier(
            # n_estimators=rc_params["n_estimators"],
            # max_depth=rc_params["max_depth"],
            # min_samples_split=rc_params["min_samples_split"],
            ## min_samples_leaf=rc_params["min_samples_leaf"],
            # max_features=rc_params["max_features"],
            # random_state=rc_params["random_state"],
            # n_features=rc_params["n_features"]
        
            n_estimators=100,         # Number of trees in the forest
            max_depth=10,             # Maximum depth of the tree
            min_samples_split=5,      # Minimum samples to split a node
            min_samples_leaf=2,       # Minimum samples in a leaf
            random_state=42
        )
    ))

    # AdaBoost Classifier
    models.append((
        'ABC',
        AdaBoostClassifier(
            n_estimators=50,          # Number of estimators
            learning_rate=1.0,        # Learning rate
            algorithm='SAMME.R',      # Algorithm type: 'SAMME' or 'SAMME.R'
            random_state=42
        )
    ))

    # Gradient Boosting Classifier
    models.append((
        'GBC',
        GradientBoostingClassifier(
            n_estimators=100,         # Number of boosting stages
            learning_rate=0.1,        # Learning rate
            max_depth=3,              # Maximum depth of individual estimators
            min_samples_split=5,      # Minimum samples to split a node
            min_samples_leaf=2,       # Minimum samples in a leaf
            random_state=42
        )
    ))
    return models
