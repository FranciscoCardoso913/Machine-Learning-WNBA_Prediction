import pandas as pd

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
    teams = teams.drop(["lgID","franchID", "divID", "seeded", "confID", "name", "arena" ], axis=1) #confID, arena?
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
teams = clear_teams(pd.read_csv('data/teams.csv'))

merged_teams = pd.merge(teams, teams_post, on=["tmID", 'year'])

print(merged_teams.head(5))
