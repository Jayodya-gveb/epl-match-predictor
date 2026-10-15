"""Build the files the app needs. Run once:  python build_assets.py

Reads (from the data/ folder):
    matches_clean.csv, train.csv, val.csv, test.csv   (outputs of Parts 1 and 2)
Writes (to the assets/ folder):
    model.joblib, team_state.csv, h2h.csv, meta.json
"""
import json
import os

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from elo_utils import K, HOME_ADV, START, add_elo

os.makedirs('assets', exist_ok=True)

FEATURES = [
    'home_form_last5_overall', 'away_form_last5_overall',
    'home_venue_form', 'away_venue_form',
    'home_gd_last5', 'away_gd_last5',
    'h2h_home_pts', 'h2h_available',
    'form_diff', 'venue_form_diff', 'gd_diff',
    'home_elo', 'away_elo', 'elo_diff',
]

# ---------------------------------------------------------------- 1. model
# Same features as Parts 2 to 4 (blanks already filled in Part 2), plus Elo.
parts = [pd.read_csv(f'data/{n}.csv', parse_dates=['date']) for n in ('train', 'val', 'test')]
full = add_elo(pd.concat(parts))
assert full[FEATURES].isna().sum().sum() == 0, 'Unexpected blanks in the feature table'

# Settings frozen from Part 3 tuning (Random Forest, class weights).
model = RandomForestClassifier(
    n_estimators=400, max_depth=4, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1,
)
# NOTE: for the app the model is refitted on ALL 8 seasons (after the evaluation was finished).
# The test scores in the report belong to the train+val version.
model.fit(full[FEATURES], full['target'])
joblib.dump(model, 'assets/model.joblib')

# ----------------------------------------------- 2. current state of each team
m = pd.read_csv('data/matches_clean.csv', parse_dates=['date'])
m = m.sort_values(['date', 'id']).reset_index(drop=True)

points = {'H': (3, 0), 'D': (1, 1), 'A': (0, 3)}
rows = []
for r in m.itertuples():
    hp, ap = points[r.FTR]
    gd = r.home_team_goal - r.away_team_goal
    rows.append((r.id, r.date, r.season, r.home_team, 'H', hp, gd, r.away_team))
    rows.append((r.id, r.date, r.season, r.away_team, 'A', ap, -gd, r.home_team))
long = pd.DataFrame(rows, columns=['id', 'date', 'season', 'team', 'venue', 'points', 'gd', 'opp'])
long = long.sort_values(['team', 'date', 'id'])

records = {}
for team, g in long.groupby('team'):
    records[team] = {
        'form5': g['points'].tail(5).mean(),
        'gd5': g['gd'].tail(5).mean(),
        'home_form5': g.loc[g['venue'] == 'H', 'points'].tail(5).mean(),
        'away_form5': g.loc[g['venue'] == 'A', 'points'].tail(5).mean(),
        'last5': ''.join({3: 'W', 1: 'D', 0: 'L'}[int(p)] for p in g['points'].tail(5)),
        'matches': len(g),
        'last_season': g['season'].iloc[-1],
    }
state = pd.DataFrame.from_dict(records, orient='index')
state.index.name = 'team'

# Elo ratings after the very last match (same settings as elo_utils.py)
elo = {}
for r in m.itertuples():
    eh, ea = elo.get(r.home_team, START), elo.get(r.away_team, START)
    expected_home = 1 / (1 + 10 ** (-(eh + HOME_ADV - ea) / 400))
    score = {'H': 1.0, 'D': 0.5, 'A': 0.0}[r.FTR]
    elo[r.home_team] = eh + K * (score - expected_home)
    elo[r.away_team] = ea - K * (score - expected_home)
state['elo'] = pd.Series(elo)
state.to_csv('assets/team_state.csv')

# Head-to-head: average points a team earned against a given opponent (either ground)
h2h = (long.groupby(['team', 'opp'])['points'].agg(['mean', 'count']).reset_index()
       .rename(columns={'mean': 'h2h_pts', 'count': 'n'}))
h2h.to_csv('assets/h2h.csv', index=False)

# ------------------------------------------------------------------ 3. meta
meta = {
    'features': FEATURES,
    'means': full[FEATURES].mean().to_dict(),
    'last_season': str(m['season'].iloc[-1]),
    'n_matches': int(len(m)),
    'date_range': [str(m['date'].min().date()), str(m['date'].max().date())],
    'sklearn_version': sklearn.__version__,
}
with open('assets/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print('Assets built:', meta['n_matches'], 'matches,', len(state), 'teams')
print('Top 5 Elo at the end of the data:')
print(state['elo'].sort_values(ascending=False).head(5).round(0).to_string())
