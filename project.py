from pyexpat import features
from unittest.mock import inplace

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Set maximum rows to None (no truncation)
pd.set_option('display.max_rows', None)

# Set maximum columns to None (no truncation)
pd.set_option('display.max_columns', None)
# teams_merged = pd.merge(teams_data, teams_post_data, on=['tmID', 'year'], how='left')

#pmerged_df = pd.merge(teams, coaches, on='tmID', how='left', validate="many_to_many")  # you can also use 'left', 'right', or 'outer' depending on your needsDIDteam_
#print(pmerged_df.head())

def clear_players(players):
    players = players.drop(["pos", "deathDate", "birthDate"], axis=1)
    players.rename(columns={"bioID": "playerID"}, inplace=True)
    return players

def clear_awards(awards):
    awards = awards[(awards["award"] != "Kim Perrot Sportmanship") & (awards["award"] != "Kim Perrot Sportmanship Award")]
    return awards

def clear_teams(teams):
    teams = teams.drop(["lgID", "divID", "seeded", "confID", "name", "arena" ], axis=1) #confID, arena?
    return teams

def clear_coaches(coaches):
    coaches = coaches.drop(["lgID"], axis=1) #TODO: see year
    coaches.rename(columns={'won': 'coach_won', 'lost':'coach_lost'}, inplace=True)
    return coaches

def clear_players_teams(players_teams):
    players_teams = players_teams.drop(["lgID"], axis=1) #TODO: see year
    return players_teams

def clear_series_post(series_post):
    series_post = series_post.drop(["lgIDWinner", "lgIDLoser"], axis=1) #TODO: see year
    return  series_post

def clear_teams_post(teams_post):
    teams_post = teams_post.drop(["lgID"],axis=1)
    return  teams_post


awards_players = clear_awards(pd.read_csv('data/awards_players.csv'))
coaches = clear_coaches(pd.read_csv('data/coaches.csv'))
players_teams = clear_players_teams(pd.read_csv('data/players_teams.csv'))
players = clear_players(pd.read_csv('data/players.csv'))
series_post = clear_series_post(pd.read_csv('data/series_post.csv'))
teams_post = clear_teams_post(pd.read_csv('data/teams_post.csv'))
df = clear_teams(pd.read_csv('data/teams.csv'))



#merged_teams = pd.merge(merged_teams, series_post, on=["tmID", 'year'])
#print(merged_teams)

awards_count = awards_players.groupby('playerID').size().reset_index(name='awards_count')
#print(awards_count)
players = players.merge(awards_count, on=['playerID'], how='left')
players["awards_count"].fillna(0, inplace=True)

#print(players.sort_values("awards_count", ascending=False).head())

players_teams_merged = players.merge(players_teams, on=['playerID'])
#print(players_teams_merged.head())
team_awards = players_teams_merged.groupby(["tmID"])["awards_count"].sum().reset_index()


df = df.merge(team_awards, on=['tmID'])
df["awards_count"].fillna(0, inplace=True)






df = df.sort_values(by=['franchID', 'year'])
df['playoffNextYear'] = df['playoff'].shift(-1)
df.loc[df['franchID']!= df['franchID'].shift(-1),'playoffNextYear'] = None
df.dropna(subset= ['playoffNextYear'], inplace=True)

df['playoff'] = df['playoff'] == 'Y'
df = pd.merge(df, teams_post, on=["tmID", 'year'], how='left')
df.fillna(0, inplace=True)
df = pd.merge(df, coaches, on=["tmID", 'year'], how='left')
#print(df[df['playoff']==True])
features = ['won', 'lost','playoff', 'W','L', "coach_won", "coach_lost", "awards_count" ]

target = 'playoffNextYear'
# Splitting data into training (earlier seasons) and testing (recent seasons)
# Assuming year 5 is an arbitrary cutoff for training vs test data
train_data = df[df.year <=6].copy()  # Earlier seasons
test_data = df[df.year > 6].copy()    # Recent seasons

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
#models.append(('SVC', SVC()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('MLP', MLPClassifier(max_iter=600)))
models.append(('RFC', RandomForestClassifier()))
models.append(('ABC', AdaBoostClassifier(algorithm='SAMME')))
models.append(('GBC', GradientBoostingClassifier()))


# Train and evaluate each model
results = {}
for name, model in models:
    # Train the model
    model.fit(X_train, y_train)
    print(model)
    # Predict on the test data
    #y_pred = model.predict(X_test)
    y_pred = model.predict_proba(X_test)



    #y_pred_wins = [sublist[1] for sublist in y_pred]

    y_pred_wins = y_pred[:,1]
    print(y_pred_wins.shape)
    # Evaluate the accuracy
    #accuracy = accuracy_score(y_test,y_pred_wins)

    # Store the result
    #results[name] = accuracy
    #print(f'{name} Accuracy: {accuracy * 100:.2f}%')

# Make predictions for the next season using the best model
# For simplicity, let’s assume you want to predict with the last model in the list
best_model = models[-1][1]  # Example: MLPClassifier
next_season = test_data[test_data.year == 9]  # Replace '6' with the next season
X_next_season = next_season[features]

next_season_predictions = best_model.predict_proba(X_next_season)
next_season['predicted_playoff'] = next_season_predictions[:,1]

# Output predictions for the next season
print(next_season[['franchID', 'year', 'predicted_playoff']])