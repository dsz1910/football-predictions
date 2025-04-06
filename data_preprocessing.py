from virtual_table import VirtualTable
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from math import sqrt


class DataPreprocessor:

    def __init__(self):
        vt = VirtualTable()
        vt.simulate_all_seasons()
        self.data = vt.stats
        self.imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, max_depth=10),
            max_iter=30, initial_strategy='mean', imputation_order='ascending', skip_complete=False)
    
    def preprocess_data(self):
        self._clean_data()
        self._fill_lack_of_data()
        self._change_formation_to_num_data()
        self._extract_all_features()
        self._fill_lack_of_data(True)

    def _fill_lack_of_data(self, all_cols=False):
        cols_not_to_impute = []
        if not all_cols:
            cols_not_to_impute = ['away_coach', 'away_formation', 'away_name', 'home_coach', 
                            'home_formation', 'home_name', 'league', 'match_date', 'round']

        cols_to_impute = [col for col in self.data.columns if col not in cols_not_to_impute]
        data_to_impute = self.data[cols_to_impute].copy()
        
<<<<<<< HEAD
        imputed_data = self.imputer.fit_transform(data_to_impute)
=======
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, max_depth=10),
            max_iter=30, initial_strategy='mean', imputation_order='ascending', skip_complete=False)
        imputed_data = imputer.fit_transform(data_to_impute)
        
>>>>>>> 3a91774dc6cfd36ec777676ac15bbfdd8eacf5f2
        imputed_data = pd.DataFrame(imputed_data, columns=cols_to_impute, index=self.data.index)
        self.data[cols_to_impute] = imputed_data

    def _clean_data(self):
        self.data = self.data[~self.data['home_name'].str.contains(
            '(', na=False, regex=False)]
        self.data = self.data[self.data['home_poss'].notnull()]

    def _change_formation_to_num_data(self):
        self.data['home_defenders_num'] = self.data['home_formation'].apply(
            lambda x: int(x[0]) if pd.notna(x) else np.nan).astype('Int64')
        
        self.data['away_defenders_num'] = self.data['away_formation'].apply(
            lambda x: int(x[0]) if pd.notna(x) else np.nan).astype('Int64')

        self.data['home_formation'] = self.data['home_formation'].apply(
            lambda x: int(x.replace(' - ', '')) if pd.notna(x) else np.nan).astype('Int64')
        
        self.data['away_formation'] = self.data['away_formation'].apply(
            lambda x: int(x.replace(' - ', '')) if pd.notna(x) else np.nan).astype('Int64')

    def _extract_all_features(self):
        final_stats = []
        self.data['position_diff'] = self.data['home_position'] - self.data['away_position']

        for game in self.data.itertuples(index=True):
            final_stats.append(self._extract_match_features(game))

        self.final_data = final_stats
        self.data = pd.DataFrame(final_stats)
        self._save_data()

    def _extract_match_features(self, game):
        data = {}
        home_matches = self._previous_matches(game, game.home_name)
        away_matches = self._previous_matches(game, game.away_name)

        if home_matches.shape[0] < 5 or away_matches.shape[0] < 5:
            return data
        
        stop = 10 if home_matches.shape[0] > 6 < away_matches.shape[0] else 7
        stats_tuple = (
    'xG',
    'shots',
    'acc_shots',
    'inacc_shots',
    'passes',
    'acc_passes',
    'corners',
    'goalkeeper_saves',
    'free_kicks',
    'offsides',
    'fouls',
    'goals',
    'mean_raiting',
    'excluded_count')
        
        for i in range(3, stop, 3):
            for stats in stats_tuple:
                data[f'home_{stats}_mean_{i-2}-{i}'], data[f'away_{stats}_mean_{i-2}-{i}'], \
                    data[f'{stats}_mean_diff_{i-2}-{i}'] = self._stats_avg(
                        game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], stats)
                
                data[f'home_opp_{stats}_mean_{i-2}-{i}'], data[f'away_opp_{stats}_mean_{i-2}-{i}'], \
                    data[f'opp_{stats}_mean_diff_{i-2}-{i}'] = self._stats_avg(
                        game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], stats, stats_against=True)
            
            data[f'home_poss_{i-2}-{i}'], data[f'away_poss_{i-2}-{i}'], data[f'poss_diff_{i-2}-{i}'] = \
                self._stats_avg(game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], 'poss')
            
            data[f'home_opp_positions_{i-2}-{i}'] = self._opponents_positions_feture(
                game.home_name, home_matches.iloc[i-3:i])
            
            data[f'away_opp_positions_{i-2}-{i}'] = self._opponents_positions_feture(
                game.away_name, away_matches.iloc[i-3:i])
            
            data[f'opp_positions_diff_{i-2}-{i}'] = data[f'home_opp_positions_{i-2}-{i}'] - data[
                f'away_opp_positions_{i-2}-{i}'] 
   
            data[f'points_per_match_diff'] = self._all_points_per_match(game.home_name, game, game.home_points) - \
                self._all_points_per_match(game.away_name, game, game.away_points)
            
            data[f'home_points_per_match_{i-2}-{i}'] = self._points_per_match_in_3_games(
                home_matches.iloc[i-3:i], game.home_name)
            
            data[f'away_points_per_match_{i-2}-{i}'] = self._points_per_match_in_3_games(
                away_matches.iloc[i-3:i], game.away_name)

        data['position_diff'] = game.position_diff

        data['home_formation'] = self._get_most_frequent(game.home_name, home_matches, 'formation')
        data['away_formation'] = self._get_most_frequent(game.away_name, away_matches, 'formation')

        data['home_defenders_num'] = self._get_most_frequent(game.home_name, home_matches, 'defenders_num')
        data['away_defenders_num'] = self._get_most_frequent(game.away_name, away_matches, 'defenders_num')

        data['home_coach_matches'] = self._coach_matches(home_matches, game.home_name, game.home_coach)
        data['away_coach_matches'] = self._coach_matches(away_matches, game.away_name, game.away_coach)
<<<<<<< HEAD

        data['result'] = game.result
        data['home_']

=======
        
>>>>>>> 3a91774dc6cfd36ec777676ac15bbfdd8eacf5f2
        return data
    
    @staticmethod
    def _get_most_frequent(team, matches, col):
        value = pd.concat([matches[matches['home_name'] == team][f'home_{col}'],
            matches[matches['away_name'] == team][f'away_{col}']]).value_counts().idxmax()
        return value
    
    @staticmethod
    def _points_per_match_in_3_games(games, team):
        points = (games[(games['home_name'] == team) & (games['result'] == 1)].shape[0] * 3) + \
                 (games[(games['away_name'] == team) & (games['result'] == 2)].shape[0] * 3) + \
                 games[games['result'] == 0].shape[0]
        return points / games.shape[0]
    
    def _all_points_per_match(self, team, game, points):
        matches_count = self.data[(self.data['match_date'] < game.match_date) & 
            (self.data['season'] == game.season) & 
            ((self.data['home_name'] == team) | (self.data['away_name'] == team))].shape[0]
            
        return points / matches_count

    @staticmethod
    def _coach_matches(games, team, coach):
        ret =  games[((games['home_coach'] == coach) & (games['home_name'] == team)) |
                     ((games['away_coach'] == coach) & (games['away_name'] == team))].shape[0]
        return ret
    
    @staticmethod
    def _opponents_positions_feture(team, games):
        home_unq = games[games['home_name'] == team]['away_position'].unique()
        away_unq = games[games['away_name'] == team]['home_position'].unique()
        unq_positions = np.concat([home_unq, away_unq])
        return np.unique(unq_positions).mean() - sqrt(unq_positions.max() - unq_positions.min())
    
    @staticmethod
    def _stats_avg(game, home, away, col, stats_against=False):
        result = []
        order = ('away', 'home') if stats_against else ('home', 'away')

        for team, matches in ((game.home_name, home), (game.away_name, away)):
            result.append(((matches[matches['home_name'] == team][f'{order[0]}_{col}'].sum() + \
            matches[matches['away_name'] == team][f'{order[1]}_{col}'].sum()) / matches.shape[0]))
        
        return result[0], result[1], result[0] - result[1]

    def _previous_matches(self, game, team):
        matches = self.data[(self.data['match_date'] < game.match_date) & 
            (self.data['season'] == game.season) &
            ((self.data['home_name'] == team) | (self.data['away_name'] == team))].sort_values(
                by='match_date', ascending=False).head(9).reset_index(drop=True)
        return matches
    
    def _save_data(self):
        self.data.to_csv('train_data.csv', index=False, encoding='utf-8')
        self.data.to_excel('train_data.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.preprocess_data()
