import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def error_eval(test, pred):
    err= 0
    i =0
    su= 0
    second_column = pred[:, 1]
    scaled = (second_column * 8) / second_column.sum()
    for index, value in test.items():
        label = 0
        if (value=='Y'): label=1
        err+= abs(scaled[i] - label)
        su += scaled[i]
        i+=1
    return err



def err_evaluation(models, X_train, X_test, y_train, y_test):
    max_acc = 12
    best_model = None
    for i in range(1):
        accuracies = [12]    
        local_best_model = None
        # Train and evaluate each model
        results = {}
        for name, model in models:
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)
            accuracy = error_eval(y_test, y_pred)
            
            if(accuracy < min(accuracies)):
                local_best_model = model

            # Store the result
            results[name] = accuracy
            accuracies.append(accuracy)
            print(f'{name} Error: {accuracy}')
        
        if(min(accuracies) < max_acc):
            best_model = local_best_model

        max_acc = min(max_acc, min(accuracies))

    print("Error: ", max_acc)
    return best_model

def normal_evaluation(models,X_train, X_test, y_train, y_test):
    max_acc = 0
    best_model = None
    for i in range(1):
        accuracies = [0]    
        local_best_model = None
        # Train and evaluate each model
        results = {}
        for name, model in models:
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            
            if(accuracy > max(accuracies)):
                local_best_model = model

            # Store the result
            results[name] = accuracy
            accuracies.append(accuracy)
            print(f'{name} Accuracy: {accuracy*100}%')
        
        if(max(accuracies) > max_acc):
            best_model = local_best_model

        max_acc = max(max_acc, max(accuracies))

    print("Accuracy: ", max_acc * 100)
    return best_model