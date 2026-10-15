# Match Result Predictor (IT3091 Group 49, Sports Analytics)

A small Streamlit app that estimates home win / draw / away win probabilities for English Premier League matches.

## Folder
football_app/
  app.py              the web app
  build_assets.py     builds the model files (run once)
  elo_utils.py        Elo rating helper (same file as in the notebooks)
  requirements.txt
  data/               put matches_clean.csv, train.csv, val.csv, test.csv here
  assets/             created by build_assets.py

## Run
1. Copy the 4 CSV files into data/
2. pip install -r requirements.txt
3. python build_assets.py
4. streamlit run app.py      (opens http://localhost:8501)

Stop the app with Ctrl+C in the terminal.

## Notes
- Ratings and form are as of the end of the data (2015/16 season): this is a demo of the system, not a current-season forecast.
- The app model is refitted on all 8 seasons. The test scores in the report belong to the train+val model.
- Educational use only, not for betting.
