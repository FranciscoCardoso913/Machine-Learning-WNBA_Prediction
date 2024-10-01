from mailcap import subst
from pyexpat import features
from unittest.mock import inplace

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# teams_merged = pd.merge(teams_data, teams_post_data, on=['tmID', 'year'], how='left')

#pmerged_df = pd.merge(teams, coaches, on='tmID', how='left', validate="many_to_many")  # you can also use 'left', 'right', or 'outer' depending on your needsDIDteam_
#print(pmerged_df.head())

def clear_players(players):
    players = players.drop(["pos"], axis=1)
    return players

def clear_awards(awards):
    awards = awards[(awards["award"] != "Kim Perrot Sportmanship") & (awards["award"] != "Kim Perrot Sportmanship Award")]
    return awards

def clear_teams(teams):
    teams = teams.drop(["lgID", "divID", "seeded", "confID", "name", "arena" ], axis=1) #confID, arena?
    return teams

def clear_coaches(coaches):
    coaches = coaches.drop(["lgID"], axis=1) #TODO: see year
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

#merged_teams = pd.merge(df, teams_post, on=["tmID", 'year'])
#merged_teams = pd.merge(merged_teams, coaches, on=["tmID", 'year'])
#merged_teams = pd.merge(merged_teams, series_post, on=["tmID", 'year'])
#print(merged_teams)



df = df.sort_values(by=['franchID', 'year'])
df['playoffNextYear'] = df['playoff'].shift(-1)
df.loc[df['franchID']!= df['franchID'].shift(-1),'playoffNextYear'] = None
df.dropna(subset= ['playoffNextYear'], inplace=True)

df['playoff'] = df['playoff'] == 'Y'

features = ['homeW', 'awayW', 'attend','playoff']


target = 'playoffNextYear'

# Splitting data into training (earlier seasons) and testing (recent seasons)
# Assuming year 5 is an arbitrary cutoff for training vs test data
train_data = df[df.year <= 6].copy()  # Earlier seasons
test_data = df[df.year > 6].copy()    # Recent seasons

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions for the next season
# Assuming you want to predict for all teams in the most recent season
next_season = test_data[test_data.year == 9]  # Replace '6' with the next season
X_next_season = next_season[features]

next_season_predictions = model.predict(X_next_season)
next_season['predicted_playoff'] = next_season_predictions

# Output predictions
print(next_season[['franchID', 'year', 'predicted_playoff']])