"""Premier League match result predictor.
A Streamlit demo of the IT3091 Group 49 project. Run with:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

AUTHOR = 'ENURI_JAYODYA'              
GITHUB_URL = 'https://github.com/Jayodya-gveb/epl-match-predictor.git'          
LINKEDIN_URL = 'https://www.linkedin.com/in/enuri-jayodya-b78238311'# ----------------------------------------------------------------------

def _fix(url):
    return url if (not url or url.startswith(('http://', 'https://'))) else 'https://' + url

GITHUB_URL = _fix(GITHUB_URL)
LINKEDIN_URL = _fix(LINKEDIN_URL)

st.set_page_config(page_title='Match Result Predictor', page_icon='⚽', layout='centered')

LABELS = {0: 'Away win', 1: 'Draw', 2: 'Home win'}
GREEN, AMBER, RED = '#22c55e', '#f59e0b', '#ef4444'

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 760px;}
.hero {background: linear-gradient(135deg, #14532d 0%, #0f766e 55%, #1e3a8a 100%);
       padding: 28px 26px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: 6px 0 0 0; color: #d1fae5; font-size: 1rem;}
.team {background: #111c2e; border: 1px solid #1f2d44; border-radius: 14px; padding: 16px; text-align: center;}
.team .role {font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #94a3b8;}
.team .tname {font-size: 1.25rem; font-weight: 700; margin: 4px 0; color: #f8fafc;}
.team .elo {color: #94a3b8; font-size: .9rem; margin-bottom: 8px;}
.vs {text-align: center; font-weight: 800; color: #64748b; padding-top: 52px; font-size: 1.1rem;}
.pill {display: inline-block; width: 24px; height: 24px; line-height: 24px; border-radius: 50%;
       margin: 0 2px; font-size: .72rem; font-weight: 700; color: #0b1220;}
.verdict {background: #111c2e; border: 1px solid #1f2d44; border-radius: 14px; padding: 18px 20px; margin: 16px 0 10px 0;}
.verdict .small {color: #94a3b8; font-size: .8rem; letter-spacing: .1em; text-transform: uppercase;}
.verdict .big {font-size: 1.8rem; font-weight: 800; color: #f8fafc; margin: 2px 0 8px 0;}
.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem; font-weight: 700; color: #0b1220;}
.probbar {display: flex; height: 38px; border-radius: 10px; overflow: hidden; margin: 6px 0 10px 0;}
.probbar div {display: flex; align-items: center; justify-content: center; color: #0b1220; font-weight: 700; font-size: .85rem;}
.legend {display: flex; justify-content: space-between; color: #cbd5e1; font-size: .9rem;}
.legend b {color: #f8fafc;}
.note {color: #94a3b8; font-size: .85rem; margin-top: 8px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load('assets/model.joblib')
    state = pd.read_csv('assets/team_state.csv', index_col='team')
    h2h = pd.read_csv('assets/h2h.csv')
    with open('assets/meta.json') as f:
        meta = json.load(f)
    return model, state, h2h, meta


try:
    model, state, h2h, meta = load_assets()
except FileNotFoundError:
    st.error('Model files not found. Run  python build_assets.py  first (see README.md).')
    st.stop()

FEATURES = meta['features']


def build_features(home, away, elo_h, elo_a, form_h, form_a):
    """Turn the two selected teams into the 14 features the model was trained on."""
    h, a = state.loc[home], state.loc[away]
    pair = h2h[(h2h['team'] == home) & (h2h['opp'] == away)]
    if len(pair):
        h2h_pts, h2h_ok = float(pair['h2h_pts'].iloc[0]), 1
    else:  # these two teams never met in the data
        h2h_pts, h2h_ok = meta['means']['h2h_home_pts'], 0
    row = {
        'home_form_last5_overall': form_h,
        'away_form_last5_overall': form_a,
        'home_venue_form': h['home_form5'],
        'away_venue_form': a['away_form5'],
        'home_gd_last5': h['gd5'],
        'away_gd_last5': a['gd5'],
        'h2h_home_pts': h2h_pts,
        'h2h_available': h2h_ok,
        'home_elo': elo_h,
        'away_elo': elo_a,
    }
    row['form_diff'] = row['home_form_last5_overall'] - row['away_form_last5_overall']
    row['venue_form_diff'] = row['home_venue_form'] - row['away_venue_form']
    row['gd_diff'] = row['home_gd_last5'] - row['away_gd_last5']
    row['elo_diff'] = elo_h - elo_a
    return pd.DataFrame([row])[FEATURES], row


def pills(seq):
    colors = {'W': GREEN, 'D': AMBER, 'L': RED}
    return ''.join(f'<span class="pill" style="background:{colors.get(c, "#64748b")}">{c}</span>'
                   for c in str(seq) if c in colors)


def team_card(name, role, elo, seq):
    return (f'<div class="team"><div class="role">{role}</div><div class="tname">{name}</div>'
            f'<div class="elo">Elo rating {elo:.0f}</div><div>{pills(seq)}</div></div>')


def prob_bar(p_home, p_draw, p_away):
    def seg(p, color, label):
        text = f'{p * 100:.0f}%' if p >= 0.12 else ''
        return f'<div style="width:{p * 100:.1f}%;background:{color}">{text}</div>'
    bar = seg(p_home, GREEN, 'Home') + seg(p_draw, AMBER, 'Draw') + seg(p_away, RED, 'Away')
    legend = (f'<div class="legend"><span><b style="color:{GREEN}">●</b> Home win <b>{p_home * 100:.0f}%</b></span>'
              f'<span><b style="color:{AMBER}">●</b> Draw <b>{p_draw * 100:.0f}%</b></span>'
              f'<span><b style="color:{RED}">●</b> Away win <b>{p_away * 100:.0f}%</b></span></div>')
    return f'<div class="probbar">{bar}</div>{legend}'


st.markdown('<div class="hero"><h1>⚽ Match Result Predictor</h1>'
            '<p>English Premier League · machine learning · explained probabilities</p></div>',
            unsafe_allow_html=True)

tab_predict, tab_how, tab_perf = st.tabs(['Predict', 'How it works', 'Model performance'])

# ------------------------------------------------------------------ Predict
with tab_predict:
    show_all = st.checkbox(f"Include teams that were not in the {meta['last_season']} season")
    pool = state if show_all else state[state['last_season'] == meta['last_season']]
    teams = sorted(pool.index)

    c1, c2 = st.columns(2)
    home = c1.selectbox('Home team', teams, index=0)
    away = c2.selectbox('Away team', teams, index=min(1, len(teams) - 1))

    if home == away:
        st.warning('Pick two different teams.')
        st.stop()

    with st.expander('What-if mode: change the ratings and form'):
        st.caption("Start values are the teams' ratings at the end of the data. Move the sliders to test a scenario.")
        lo, hi = int(state['elo'].min()) - 100, int(state['elo'].max()) + 100
        e1, e2 = st.columns(2)
        elo_h = e1.slider('Home Elo', lo, hi, int(round(state.loc[home, 'elo'])), 5, key=f'eh_{home}')
        elo_a = e2.slider('Away Elo', lo, hi, int(round(state.loc[away, 'elo'])), 5, key=f'ea_{away}')
        f1, f2 = st.columns(2)
        form_h = f1.slider('Home form (avg points, last 5)', 0.0, 3.0,
                           float(round(state.loc[home, 'form5'], 1)), 0.1, key=f'fh_{home}')
        form_a = f2.slider('Away form (avg points, last 5)', 0.0, 3.0,
                           float(round(state.loc[away, 'form5'], 1)), 0.1, key=f'fa_{away}')

    X, row = build_features(home, away, elo_h, elo_a, form_h, form_a)
    proba = model.predict_proba(X)[0]
    probs = pd.Series(proba, index=[LABELS[c] for c in model.classes_])

    left, mid, right = st.columns([5, 1, 5])
    left.markdown(team_card(home, 'Home', elo_h, state.loc[home].get('last5', '')), unsafe_allow_html=True)
    mid.markdown('<div class="vs">VS</div>', unsafe_allow_html=True)
    right.markdown(team_card(away, 'Away', elo_a, state.loc[away].get('last5', '')), unsafe_allow_html=True)

    best, p = probs.idxmax(), probs.max()
    if p >= 0.55:
        level, color = 'Higher confidence', GREEN
    elif p >= 0.45:
        level, color = 'Medium confidence', AMBER
    else:
        level, color = 'Low confidence: close call', RED
    st.markdown(f'<div class="verdict"><div class="small">Most likely result</div>'
                f'<div class="big">{best}</div>'
                f'<span class="chip" style="background:{color}">{level}</span>'
                f'{prob_bar(probs["Home win"], probs["Draw"], probs["Away win"])}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="note">The last-5 pills show each team\'s most recent results in the data '
                '(W win, D draw, L loss, oldest first). Confidence labels are a simple rule of thumb: in our test season the '
                'model was right more often when it was more confident, but results stay uncertain. '
                'These are probabilities, not promises.</div>', unsafe_allow_html=True)

    with st.expander('See the numbers the model used'):
        shown = pd.DataFrame({
            home + ' (home)': [row['home_elo'], row['home_form_last5_overall'],
                               row['home_venue_form'], row['home_gd_last5']],
            away + ' (away)': [row['away_elo'], row['away_form_last5_overall'],
                               row['away_venue_form'], row['away_gd_last5']],
        }, index=['Elo rating', 'Form (avg points, last 5)',
                  'Venue form (home games / away games)', 'Goal difference per match, last 5']).round(2)
        st.table(shown)
        if row['h2h_available']:
            st.write(f"Head-to-head: {home} averaged {row['h2h_home_pts']:.2f} points against {away} in past meetings.")
        else:
            st.write('Head-to-head: these teams have no past meeting in the data, so an average value was used.')

# ----------------------------------------------------------------- How it works
with tab_how:
    st.markdown(f"""
**What it does.** It estimates the chance of a home win, a draw and an away win for an English Premier League match.

**Data.** {meta['n_matches']:,} matches from {meta['date_range'][0]} to {meta['date_range'][1]} (Kaggle European Soccer Database).
Ratings and form shown here are as of the end of that period, so this is a demo of the system, not a forecast for the current season.

**Clues (features) the model uses**
- **Elo rating:** a team-strength score that goes up after wins and down after losses (bigger changes for surprising results).
- **Recent form:** average points in the last 5 matches, overall and at home or away.
- **Goal difference:** goals scored minus conceded in the last 5 matches.
- **Head-to-head:** points a team earned in earlier meetings with this opponent.

**Model.** A Random Forest (many decision trees voting together) with class weights, so draws and away wins are not ignored.
It was chosen by comparing four models on validation seasons, and tested once on a locked season.

**Leakage control.** Every clue uses only matches played before kickoff, and the data was split by time, never randomly.
""")

# ------------------------------------------------------------- Performance
with tab_perf:
    st.markdown('**Sealed test season (2015/16, 380 matches)**')
    perf = pd.DataFrame({
        'Model': ['Always predict Home', 'Random Forest (primary)', 'Gradient Boosting', 'Logistic Regression'],
        'Accuracy': [0.413, 0.439, 0.455, 0.466],
        'Macro-F1': [0.195, 0.428, 0.442, 0.421],
        'Log-loss (lower is better)': [1.089, 1.053, 1.056, 1.050],
    })
    st.table(perf.set_index('Model'))
    st.markdown("""
**Honest reading.** The accuracy gain over always guessing a home win is small (about 2.6 points for the primary model)
and not statistically reliable on one season. The probabilities improved a little (log-loss 1.053 against 1.089).
The model is most useful as a confidence-aware aid: on its most confident quarter of predictions accuracy was 64%,
on its least confident quarter 33%.

These figures come from the model trained on 2008/09 to 2014/15. The model in this app was refitted on all seasons.
""")

st.divider()
links = []
if GITHUB_URL:
    links.append(f'[GitHub]({GITHUB_URL})')
if LINKEDIN_URL:
    links.append(f'[LinkedIn]({LINKEDIN_URL})')
who = f'Built by {AUTHOR}. ' if AUTHOR else ''
st.caption(f"{who}{' · '.join(links)}  \nEducational demo trained on 2008/09 to 2015/16 data. Not for betting or financial decisions.")
