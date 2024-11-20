def clear_players(players):
    players = players.drop(["pos", "deathDate", "birthDate"], axis=1)
    players.rename(columns={"bioID": "playerID"}, inplace=True)
    return players

def clear_awards(awards):
    awards = awards[(awards["award"] != "Kim Perrot Sportmanship") & (awards["award"] != "Kim Perrot Sportmanship Award")]
    return awards

def clear_teams(teams):
    teams = teams.drop(["lgID", "divID", "seeded", "confID", "name", "arena" ], axis=1) #confID, arena?
    
    teams["firstRound"] = teams["firstRound"].replace({"W": 1, "L": 0})
    teams["semis"] = teams["semis"].replace({"W": 1, "L": 0})
    teams["finals"] = teams["finals"].replace({"W": 1, "L": 0})
    
    teams["firstRound"] = teams["firstRound"].fillna(0)
    teams["semis"] = teams["semis"].fillna(0)
    teams["finals"] = teams["finals"].fillna(0)
    
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