import pandas as pd
from services.clean_data import *
from services.eval import *
from services.models import *
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# Set maximum rows to None (no truncation)
pd.set_option('display.max_rows', None)

# Set maximum columns to None (no truncation)
pd.set_option('display.max_columns', None)
# teams_merged = pd.merge(teams_data, teams_post_data, on=['tmID', 'year'], how='left')

#pmerged_df = pd.merge(teams, coaches, on='tmID', how='left', validate="many_to_many")  # you can also use 'left', 'right', or 'outer' depending on your needsDIDteam_
#print(pmerged_df.head())


awards_players = clear_awards(pd.read_csv('data/awards_players.csv'))
coaches = clear_coaches(pd.read_csv('data/coaches.csv'))
players_teams = clear_players_teams(pd.read_csv('data/players_teams.csv'))
players = clear_players(pd.read_csv('data/players.csv'))
series_post = clear_series_post(pd.read_csv('data/series_post.csv'))
teams_post = clear_teams_post(pd.read_csv('data/teams_post.csv'))
df = clear_teams(pd.read_csv('data/teams.csv'))

print(f"Number of rows where year == 9: {(df['year'] == 9).sum()}\n\n")
#merged_teams = pd.merge(merged_teams, series_post, on=["tmID", 'year'])
#print(merged_teams)

# awards_count = awards_players.groupby('playerID').size().reset_index(name='awards_count')
# #print(awards_count)
# players = players.merge(awards_count, on=['playerID'], how='left')
# players["awards_count"].fillna(0, inplace=True)
# 
# #print(players.sort_values("awards_count", ascending=False).head())
# 
# players_teams_merged = players.merge(players_teams, on=['playerID'])
# #print(players_teams_merged.head())
# team_awards = players_teams_merged.groupby(["tmID"])["awards_count"].sum().reset_index()
# 
# 
# df = df.merge(team_awards, on=['tmID'])
# df["awards_count"].fillna(0, inplace=True)

# Step 1: Calculate the total awards per player for each year
# Assuming 'award' is the column that lists the awards
player_awards_by_year = awards_players.groupby(['playerID', 'year']).size().reset_index(name='awards_count')

# Step 2: Apply a cumulative sum to get the total awards by year for each player
player_awards_by_year['cumulative_awards'] = player_awards_by_year.groupby('playerID')['awards_count'].cumsum()

# Example output columns: ['playerID', 'yearID', 'awards_count', 'cumulative_awards']

# Step 3: Merge cumulative player awards with team_players data
team_players_awards = pd.merge(players_teams, player_awards_by_year[['playerID', 'year', 'cumulative_awards']],
                               on=['playerID', 'year'], how='left')

# Fill missing values (for players with no awards) with 0
team_players_awards['cumulative_awards'].fillna(0, inplace=True)

# Step 4: Group by team and year to sum the cumulative awards for each team
team_awards_by_year = team_players_awards.groupby(['tmID', 'year'])['cumulative_awards'].sum().reset_index()

# Example output columns: ['tmID', 'yearID', 'cumulative_awards']

# Step 5: Merge cumulative team awards into your main dataframe
df = df.merge(team_awards_by_year, on=['tmID', 'year'], how='left')
print(f"\n\nNumber of rows where year == 9: {(df['year'] == 9).sum()}\n\n")

# Fill any missing awards with 0
df['cumulative_awards'].fillna(0, inplace=True)

df = df.sort_values(by=['franchID', 'year'])
df['playoffNextYear'] = df['playoff'].shift(-1)
df.loc[df['franchID']!= df['franchID'].shift(-1),'playoffNextYear'] = None
df.dropna(subset= ['playoffNextYear'], inplace=True)
print(f"\n\nNumber of rows where year == 9: {(df['year'] == 9).sum()}\n\n")

df['playoff'] = df['playoff'] == 'Y'
df = pd.merge(df, teams_post, on=["tmID", 'year'], how='left')
df.fillna(0, inplace=True)





# df = pd.merge(df, coaches, on=["tmID", 'year'], how='left')

coaches["WR"] = coaches["coach_won"] / (coaches["coach_won"] + coaches["coach_lost"])
# if stint 0, firstCoachWR = secondCoachWR = WR, if stint != 0, firstCoachWR = stint1 and secondCoachWR = stint2
# Create two separate DataFrames for first and second coach WR
first_coach_wr = coaches[coaches['stint'] == 1][['year', 'tmID', 'WR']].rename(columns={'WR': 'firstCoachWR'})
second_coach_wr = coaches[coaches['stint'] == 2][['year', 'tmID', 'WR']].rename(columns={'WR': 'secondCoachWR'})

# Step 3: If stint == 0, assign the same WR to both first and second coach
same_coach_wr = coaches[coaches['stint'] == 0][['year', 'tmID', 'WR']]
same_coach_wr['firstCoachWR'] = same_coach_wr['WR']
same_coach_wr['secondCoachWR'] = same_coach_wr['WR']
same_coach_wr = same_coach_wr[['year', 'tmID', 'firstCoachWR', 'secondCoachWR']]

# Step 4: Combine all the WR data (for all stints) into a single DataFrame
combined_wr = pd.concat([first_coach_wr, second_coach_wr, same_coach_wr], axis=0).drop_duplicates(subset=['year', 'tmID'])

# Step 5: Merge this combined data into the teams DataFrame
df = pd.merge(df, combined_wr, on=['year', 'tmID'], how='left')

all_time_best_players = players_teams.groupby('playerID')["points"].sum().reset_index().sort_values(by=['points'], ascending=False)
top_all_time_best_players = all_time_best_players.merge(players_teams, on=['playerID']).groupby('playerID')
top_all_time_best_players = top_all_time_best_players.head(5)

tmid_counts = top_all_time_best_players['tmID'].value_counts().reset_index()
tmid_counts.columns = ['tmID', 'tmID_count']

tmid_counts = tmid_counts.rename(columns={'tmID_count': 'number_of_top_players'})

df = df.merge(tmid_counts, on=['tmID'], how='left')
df['number_of_top_players'].fillna(0, inplace=True)

#print(df.columns.tolist())

# print(df)

# print(df[df["has_top_player"] == False])

# print(top_all_tie_best_players.head(5))
# print(all_time_best_players.columns.tolist())
# all_time_best_players = all_time_best_players.merge(players_teams, on=['playerID'])
# print(all_time_best_players.head(5))
# df = pd.merge(df, all_time_best_players, on=['tmID']) 

# print(df.head(5))
# print(df.columns.tolist())

df['shot_accuracy'] = (df['o_fgm'] + df['o_3pm']) / (df['o_fga']+ df['o_3pa'])
df['defensive_accuracy'] = (df['d_fgm'] + df['d_ftm'] + df['d_3pm']) / (df['d_fga'] + df['d_fta'] + df['d_3pa'])
df['win_rate'] = (df['won']) / (df['won'] + df['lost'])
df["fg_effeciency"] = (df['o_fgm']  + df['o_3pm']*0.5 )/ (df['o_fga'])
df["shoot_percentage"] = (df['o_pts']  )/ (2*(df['o_fga']+0.44*df['o_fta']))
df["n_playoff"] = (
    df.assign(playoff_numeric=df["playoff"])
    .groupby("franchID")["playoff_numeric"]  # Group by team
    .cumsum()  # Cumulative sum of playoff appearances
)

features = [
    "playoff", "W", "L", "cumulative_awards", "number_of_top_players", "rank", "firstRound", "semis", "finals", 
    "homeW", "homeL", "awayW", "awayL", "GP", "min", "confW", "confL", "attend","defensive_accuracy",
    "o_reb", "d_reb", "d_to", "d_stl", "d_blk","shot_accuracy","win_rate", "o_dreb","o_oreb","d_oreb","d_dreb",
    "fg_effeciency","shoot_percentage","n_playoff"
]

target = 'playoffNextYear'

# Splitting data into training (earlier seasons) and testing (recent seasons)
# Assuming year 5 is an arbitrary cutoff for training vs test data
train_data = df[df.year <= 8].copy()  # Earlier seasons
test_data = df[df.year > 8].copy()    # Recent seasons

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

models = getting_models(X_train, y_train, X_test, y_test, False)


# Using the test data
normal_evaluation(models,X_train, X_test, y_train, y_test)
best_model = err_evaluation(models,X_train, X_test, y_train, y_test)

print(X_train.shape)


"""
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(30, )),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100,callbacks=[early_stopping])
model.evaluate(X_test, y_test)

"""

# Make predictions for the next season using the best model
# For simplicity, let’s assume you want to predict with the last model in the list
# best_model = models[-1][1]  # Example: MLPClassifier
next_season = test_data[test_data.year == 9]  # Replace '6' with the next season
X_next_season = next_season[features]

next_season_predictions = best_model.predict(X_next_season)
# next_season_predictions = best_model.predict_proba(X_next_season)
next_season['predicted_playoff'] = next_season_predictions
# next_season['predicted_playoff'] = next_season_predictions[:,1]

# Output predictions for the next season
print(next_season[['franchID', 'year', 'predicted_playoff']])
