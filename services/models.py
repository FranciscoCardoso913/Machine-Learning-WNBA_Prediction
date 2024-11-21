from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

def getting_models():
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

    # Decision Tree Classifier
    models.append((
        'DTC',
        DecisionTreeClassifier(
            criterion='gini',       # Split criterion: 'gini' or 'entropy'
            max_depth=10,           # Limit depth of the tree
            min_samples_split=5,    # Minimum samples to split a node
            min_samples_leaf=2,     # Minimum samples in a leaf
            random_state=42
        )
    ))

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
    models.append((
        'MLP',
        MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two layers with 100 and 50 neurons
            activation='relu',            # Activation function
            solver='adam',                # Optimization algorithm
            alpha=0.0001,                 # L2 penalty (regularization term)
            max_iter=600,                 # Maximum iterations
            random_state=42
        )
    ))

    # Random Forest Classifier
    models.append((
        'RFC',
        RandomForestClassifier(
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
