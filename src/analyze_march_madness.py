import pandas as pd
import seaborn as sns

import sklearn.metrics as ms

### CONSTANTS ###

DATA_DIR = '../data/kaggle/'

### FUNCTIONS ###

def extract_game_info(id_str):

    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):

    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16


# load results from compact files (men's and women's separately)
df_m = pd.read_csv( DATA_DIR +'MNCAATourneySeeds.csv' )
df_m = df_m[ df_m[ 'Season' ] >= 2021 ]
df_w = pd.read_csv( DATA_DIR +'WNCAATourneySeeds.csv' )
df_w = df_w[ df_w[ 'Season' ] >= 2021 ]

# concatenate men and women's results
seed_df = pd.concat( [df_m, df_w], ignore_index=True )
#print(seed_df.shape)

submit_df = pd.read_csv( DATA_DIR + 'SampleSubmissionStage2.csv' )
#submit_df = pd.read_csv( DATA_DIR + 'SeedBenchmarkStage1.csv' )


submit_df[['Season', 'TeamID1', 'TeamID2']] = \
    submit_df['ID'].apply(extract_game_info).tolist()
seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)
seed_df['Season'] = seed_df['Season'] + 1

print(f"Seed shape: {seed_df.shape}")
print(f"Seed head:")
print(seed_df.head())

# Merge seed information for TeamID1
submit_df = pd.merge( submit_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                      left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                      how='left' )
submit_df = submit_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

#print(submit_df.shape)

# Merge seed information for TeamID2
submit_df = pd.merge(submit_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                     left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                     how='left')
submit_df = submit_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

submit_df['SeedDiff'] = submit_df['SeedValue1'] - submit_df['SeedValue2']

#print(submit_df.shape)
#print(submit_df.dropna())

# Update 'Pred' column
submit_df['Pred'] = 0.5 + (0.03 * submit_df['SeedDiff'])

# Drop unnecessary columns
submit_df = submit_df[['ID', 'Pred']].fillna(0.5)

# Preview your submission
stats = submit_df.iloc[:, 1].describe()
print(stats)

# Create a dataframe of ground truth values
solution_df = submit_df.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submit_df['Pred']
brier_score = ms.brier_score_loss(y_true, y_pred)
print(f"Brier score: {brier_score}")

submit_df.to_csv( DATA_DIR + 'submission.csv', index=False)

