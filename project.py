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
coaches = pd.read_csv('data/coaches.csv')
coaches_s11 = pd.read_csv('data/s11/coaches.csv')
coaches = clear_coaches(pd.concat([coaches, coaches_s11]))
players_teams = pd.read_csv('data/players_teams.csv')
players_teams_s11 = pd.read_csv('data/s11/players_teams.csv')
players_teams = clear_players_teams(pd.concat([players_teams, players_teams_s11]))
players = clear_players(pd.read_csv('data/players.csv'))
series_post = clear_series_post(pd.read_csv('data/series_post.csv'))
teams_post = clear_teams_post(pd.read_csv('data/teams_post.csv'))
df = pd.read_csv('data/teams.csv')
df_s11 = pd.read_csv('data/s11/teams.csv')
df = clear_teams(pd.concat([df, df_s11]))

# AWARDS
# Step 1: Calculate the total awards per player for each year
player_awards_by_year = awards_players.groupby(['playerID', 'year']).size().reset_index(name='awards_count')

# Step 2: Ensure all years are represented for each player
# Create a full set of years for each player
all_years = pd.DataFrame({'year': range(1, awards_players['year'].max() + 1)})  # Assuming years start from 1
player_years = player_awards_by_year['playerID'].unique()
full_player_data = pd.DataFrame([(player, year) for player in player_years for year in range(1, awards_players['year'].max() + 1)], columns=['playerID', 'year'])

# Merge with original player data to fill in missing years (with NaN awards_count for missing years)
player_awards_by_year = pd.merge(full_player_data, player_awards_by_year, on=['playerID', 'year'], how='left')

# Fill missing awards_count with 0 where no award was given that year
player_awards_by_year['awards_count'].fillna(0, inplace=True)

# Step 3: Calculate cumulative awards per player (cumsum over awards_count)
player_awards_by_year['cumulative_awards'] = player_awards_by_year.groupby('playerID')['awards_count'].cumsum()

# Step 4: Calculate total awards per coach for each year (using coachID instead of playerID)
coach_awards_by_year = awards_players[awards_players['playerID'].isin(coaches['coachID'])]
coach_awards_by_year = coach_awards_by_year.groupby(['playerID', 'year']).size().reset_index(name='awards_count')

# Step 5: Ensure all years are represented for each coach
# Create a full set of years for each coach
coach_ids = coach_awards_by_year['playerID'].unique()  # Same as 'coachID' here
full_coach_data = pd.DataFrame([(coach, year) for coach in coach_ids for year in range(1, awards_players['year'].max() + 1)], columns=['playerID', 'year'])

# Merge with original coach data to fill in missing years
coach_awards_by_year = pd.merge(full_coach_data, coach_awards_by_year, on=['playerID', 'year'], how='left')

# Fill missing awards_count with 0 where no award was given that year
coach_awards_by_year['awards_count'].fillna(0, inplace=True)

# Step 6: Calculate cumulative awards per coach (cumsum over awards_count)
coach_awards_by_year['cumulative_awards'] = coach_awards_by_year.groupby('playerID')['awards_count'].cumsum()

# Rename playerID to coachID to distinguish coaches
coach_awards_by_year.rename(columns={'playerID': 'coachID'}, inplace=True)

# Combine both player and coach awards
combined_awards_by_year = pd.concat([player_awards_by_year, coach_awards_by_year])

# --- Step 7: Merging with the next year's stats (for players and coaches) ---
# Shift players_teams data to associate players of year X+1 with teams in year X
players_teams_next_year = players_teams.copy()
players_teams_next_year['year'] -= 1  # Shift years back to merge with current awards
players_teams_next_year = players_teams_next_year[players_teams_next_year['year'] > 0]

# Merge cumulative awards for players
team_players_awards = pd.merge(
    players_teams_next_year,
    combined_awards_by_year[['playerID', 'year', 'cumulative_awards']],
    on=['playerID', 'year'],
    how='left'
)

# Shift coaches data to associate coaches of year X+1 with teams in year X
coaches_next_year = coaches.copy()
coaches_next_year['year'] -= 1  # Shift years back to merge with current awards
coaches_next_year = coaches_next_year[coaches_next_year['year'] > 0]

# Merge cumulative awards for coaches
team_coaches_awards = pd.merge(
    coaches_next_year,
    combined_awards_by_year[['coachID', 'year', 'cumulative_awards']],
    on=['coachID', 'year'],
    how='left'
)

# --- Combine and Aggregate ---
# Standardize column names to allow concatenation
team_coaches_awards = team_coaches_awards.rename(columns={'coachID': 'playerID'})  # Treat coaches as players for processing

# Concatenate player and coach awards
team_players_awards = pd.concat([team_players_awards, team_coaches_awards], ignore_index=True)

# Group by team and year, summing cumulative awards
team_players_awards = team_players_awards.groupby(['tmID', 'year'], as_index=False).agg({'cumulative_awards': 'sum'})

# Fill missing values (for players/coaches with no awards) with 0
team_players_awards['cumulative_awards'].fillna(0, inplace=True)

# Step 11: Merge the result with the main team dataframe (df)
df = df.merge(team_players_awards[['tmID', 'year', 'cumulative_awards']], on=['tmID', 'year'], how='left')

# Step 12: Fill missing values for teams that don't have any data for the next year
df['cumulative_awards'].fillna(0, inplace=True)

# COACHES
# Step 1: Calculate cumulative stats for each coach (up to their current year)
coaches['cumulative_wins'] = coaches.groupby('coachID')['coach_won'].cumsum()
coaches['cumulative_losses'] = coaches.groupby('coachID')['coach_lost'].cumsum()
coaches['cumulative_WR'] = coaches['cumulative_wins'] / (coaches['cumulative_wins'] + coaches['cumulative_losses'])

# Step 2: Shift the year for coaches to assign the correct future coach for each team
future_coaches = coaches[['coachID', 'tmID', 'year']].copy()
future_coaches['year'] = future_coaches['year'] + 1  # Assign future coach to the current year (team year)

# Step 3: Merge cumulative WR for the coaches on the team (using current year stats for the future coach)
cumulative_stats = coaches[['coachID', 'year', 'cumulative_WR']]
future_coaches_stats = pd.merge(future_coaches, cumulative_stats, on=['coachID', 'year'], how='left')

# Step 4: Merge future coaches' stats into the teams DataFrame
df = pd.merge(df, future_coaches_stats, on=['tmID', 'year'], how='left')

# Step 5: Handle missing values (teams with no coach yet assigned in future years)
df['cumulative_WR'].fillna(0, inplace=True)

# TOP PLAYERS
all_time_best_players = players_teams.groupby('playerID')["points"].sum().reset_index().sort_values(by=['points'], ascending=False)
top_all_time_best_players = all_time_best_players.merge(players_teams, on=['playerID']).groupby('playerID')
top_all_time_best_players = top_all_time_best_players.head(5)

tmid_counts = top_all_time_best_players['tmID'].value_counts().reset_index()
tmid_counts.columns = ['tmID', 'tmID_count']

tmid_counts = tmid_counts.rename(columns={'tmID_count': 'number_of_top_players'})

df = df.merge(tmid_counts, on=['tmID'], how='left')
df['number_of_top_players'].fillna(0, inplace=True)

# MORE STATS
df['shot_accuracy'] = (df['o_fgm'] + df['o_3pm']) / (df['o_fga']+ df['o_3pa'])
df['defensive_accuracy'] = (df['d_fgm'] + df['d_ftm'] + df['d_3pm']) / (df['d_fga'] + df['d_fta'] + df['d_3pa'])
df['win_rate'] = (df['won']) / (df['won'] + df['lost'])
df["fg_effeciency"] = (df['o_fgm']  + df['o_3pm']*0.5 )/ (df['o_fga'])
df["shoot_percentage"] = (df['o_pts']  )/ (2*(df['o_fga']+0.44*df['o_fta']))

# PLAYOFF NEXT YEAR
df.loc[df['year'] == 11, 'playoff'] = 0
df = df.sort_values(by=['franchID', 'year'])
df['playoffNextYear'] = df['playoff'].shift(-1)
df.loc[df['franchID']!= df['franchID'].shift(-1),'playoffNextYear'] = None
df.dropna(subset= ['playoffNextYear'], inplace=True)

df['playoff'] = df['playoff'] == 'Y'
df['playoffNextYear'] = df['playoffNextYear'] == 'Y'
df = pd.merge(df, teams_post, on=["tmID", 'year'], how='left')
df.fillna(0, inplace=True)
df = df.drop_duplicates(subset=['tmID', 'year'])

df["n_playoff"] = (
    df.assign(playoff_numeric=df["playoff"])
    .groupby("franchID")["playoff_numeric"]  # Group by team
    .cumsum()  # Cumulative sum of playoff appearances
)

features = [
    "playoff", "W", "L", "cumulative_awards", "number_of_top_players", "rank", "firstRound", "semis", "finals",
    "homeW", "homeL", "awayW", "awayL", "GP", "min", "confW", "confL", "attend","defensive_accuracy",
    "o_reb", "d_reb", "d_to", "d_stl", "d_blk","shot_accuracy","win_rate", "o_dreb","o_oreb","d_oreb","d_dreb",
    "fg_effeciency","shoot_percentage"
]

target = 'playoffNextYear'
train_test_split = 8
# Splitting data into training (earlier seasons) and testing (recent seasons)
# Assuming year 5 is an arbitrary cutoff for training vs test data
train_data = df[df.year <= train_test_split].copy() # Earlier seasons
test_data = df[(df['year'] >= train_test_split) & (df['year'] < 10)].copy() # Recent seasons

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
next_season = df[df.year == 10]  # Replace '6' with the next season
X_next_season = next_season[features]

# next_season_predictions = best_model.predict(X_next_season)
next_season_predictions = best_model.predict_proba(X_next_season)
# next_season['predicted_playoff'] = next_season_predictions
next_season['predicted_playoff'] = next_season_predictions[:,1]
next_season['next_year'] = next_season['year'] + 1
# Output predictions for the next season
print(next_season[['franchID', 'next_year', 'predicted_playoff']])
