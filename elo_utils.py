K, HOME_ADV, START = 20, 60, 1500   # standard Elo settings, fixed in advance and never tuned

def add_elo(df):
    """Adds home_elo, away_elo, elo_diff. Each rating is the value BEFORE the match;
    ratings are updated AFTER the match, so no row ever sees its own result."""
    df = df.sort_values(['date', 'id']).reset_index(drop=True).copy()
    elo, h_elo, a_elo = {}, [], []
    for r in df.itertuples():
        eh = elo.get(r.home_team, START)
        ea = elo.get(r.away_team, START)
        h_elo.append(eh)
        a_elo.append(ea)
        exp_home = 1 / (1 + 10 ** (-(eh + HOME_ADV - ea) / 400))
        score = {'H': 1.0, 'D': 0.5, 'A': 0.0}[r.FTR]
        elo[r.home_team] = eh + K * (score - exp_home)
        elo[r.away_team] = ea - K * (score - exp_home)
    df['home_elo'], df['away_elo'] = h_elo, a_elo
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df
