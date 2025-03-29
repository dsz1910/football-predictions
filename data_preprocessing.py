from virtual_table import VirtualTable
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class DataPreprocessor:

    def __init__(self):
        vt = VirtualTable()
        vt.simulate_all_seasons()
        self.initial_data = vt.stats
        self.final_data = None
        self.target = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        self._clean_data()
        self._fill_lack_of_data()
        self._change_formation_to_num_data()
        #self._extract_all_features()

    def _fill_lack_of_data(self):
        cols_not_to_impute = ['away_coach', 'away_formation', 'away_name', 'home_coach', 
                            'home_formation', 'home_name', 'league', 'match_date', 'round']
        cols_to_impute = [col for col in self.initial_data.columns if col not in cols_not_to_impute]
        data_to_impute = self.initial_data[cols_to_impute].copy()
        
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=30, max_depth=20), max_iter=15)
        imputed_data = imputer.fit_transform(data_to_impute)
        
        imputed_data = pd.DataFrame(imputed_data, columns=cols_to_impute, index=self.initial_data.index)
        self.initial_data[cols_to_impute] = imputed_data

    def _clean_data(self):
        self.initial_data = self.initial_data[~self.initial_data['home_name'].str.contains(
            '(', na=False, regex=False)]
        self.initial_data = self.initial_data[self.initial_data['home_poss'].notnull()]

    def _change_formation_to_num_data(self):
        self.initial_data['home_defenders_num'] = self.initial_data['home_formation'].apply(
            lambda x: int(x[0]) if pd.notna(x) else np.nan).astype('Int64')
        
        self.initial_data['away_defenders_num'] = self.initial_data['away_formation'].apply(
            lambda x: int(x[0]) if pd.notna(x) else np.nan).astype('Int64')

        self.initial_data['home_formation'] = self.initial_data['home_formation'].apply(
            lambda x: int(x.replace(' - ', '')) if pd.notna(x) else np.nan).astype('Int64')
        
        self.initial_data['away_formation'] = self.initial_data['away_formation'].apply(
            lambda x: int(x.replace(' - ', '')) if pd.notna(x) else np.nan).astype('Int64')

    def _extract_all_features(self):
        final_stats = []

        for game in self.initial_data.itertuples(index=True):
            final_stats.append(self._extract_match_features(game))
        self.final_data = final_stats

    def _extract_match_features(self, game):
        data = {}
        home_matches = self._previous_matches(game, game.home_name)
        away_matches = self._previous_matches(game, game.away_name)
        home_games_count, away_games_count = home_matches.shape[0], away_matches.shape[0]

        if home_games_count < 5 or away_games_count < 5:
            return data
        
        stop = 10 if home_games_count > 6 < away_games_count else 7
        stats_tuple = (
    'xG',
    'shots',
    'acc_shots',
    'inacc_shots',
    'passes',
    'acc_passes',
    'corners',
    'goalkeepers_saves',
    'free_kicks',
    'offsides',
    'fouls',
    'goals',
    'mean_raiting',
    'excluded_count',
    'position')
        for i in range(3, stop, 3):
            for stats in stats_tuple:
                data[f'home_{stats}_mean_{i}'], data[f'away_{stats}_mean_{i}'], data[f'{stats}_mean_diff_{i}'] = \
                    self._stats_avg(game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], stats)
                
                data[f'home_opp_{stats}_mean_{i}'], data[f'away_opp_{stats}_mean_{i}'], \
                    data[f'opp_{stats}_mean_diff{i}'] = self._stats_avg(
                        game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], stats, stats_against=True)
                
            data[f'home_poss_{i}'], data[f'away_poss_{i}'], data[f'poss_diff_{i}'] = self._stats_avg(
                game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], 'poss')

        return data
    
    @staticmethod
    def _stats_avg(game, home, away, col, stats_against=False):
        result = []
        order = ('away', 'home') if stats_against else ('home', 'away')

        for team, matches in ((game.home_name, home), (game.away_name, away)):
            result.append(((matches[matches['home_name'] == team][f'{order[0]}_{col}'].sum() + \
            matches[matches['away_name'] == team][f'{order[1]}_{col}'].sum()) / matches.shape[0]))
        
        return result[0], result[1], result[0] - result[1]

    def _previous_matches(self, game, team):
        matches = self.initial_data[(self.initial_data['match_date'] < game.match_date) & 
            (self.initial_data['season'] == game.season) &
            ((self.initial_data['home_name'] == team) | (self.initial_data['away_name'] == team))].sort_values(
                by='match_date', ascending=False).head(9).reset_index(drop=True)
        return matches


if __name__ == '__main__':
    dp = DataPreprocessor()
    print(dp.initial_data.loc[0])
    for game in dp.initial_data.itertuples(index=True):
        previous_matches, away = dp._previous_matches(game)
        #print(home[['home_name', 'home_goals', 'away_name', 'away_goals']])
        #print(away[['home_name', 'home_goals', 'away_name', 'away_goals']])
        #print(*dp.goals_sum_and_diff(game, home, away))
        break