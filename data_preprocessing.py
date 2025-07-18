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
    
    def preprocess_data(self):
        self._clean_data()
        self._fill_lack_of_data()
        self._change_formation_to_num_data()
        self._extract_all_features()
        self._fill_lack_of_data(all_cols=True)
        self._save_data()

    def _fill_lack_of_data(self, all_cols=False, n_estimators=100, max_depth=5, max_iter=10, n_nearest_features=50):
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=n_estimators, 
            max_depth=max_depth), max_iter=max_iter, initial_strategy='mean', imputation_order='ascending',
            skip_complete=True, n_nearest_features=n_nearest_features, random_state=42, verbose=1)
        
        cols_not_to_impute = ['home_name', 'away_name', 'result']
        if not all_cols:
            cols_not_to_impute = ['away_coach', 'away_formation', 'away_name', 'home_coach', 
                            'home_formation', 'home_name', 'league', 'match_date', 'round']
            
        else:
            self.data = self.data[self.data.isna().mean(axis=1) < 0.5]

        cols_to_impute = [col for col in self.data.columns if col not in cols_not_to_impute]
        data_to_impute = self.data[cols_to_impute].copy()

        imputed_data = imputer.fit_transform(data_to_impute)
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
            result = self._extract_match_features(game)
            if result:
                final_stats.append(result)
            if game.Index % 1000 == 0:
                print(game.Index)

        self.final_data = final_stats
        self.data = pd.DataFrame(final_stats)

    def _extract_match_features(self, game):
        home_matches = self._previous_matches(game, game.home_name)
        away_matches = self._previous_matches(game, game.away_name)

        if home_matches.shape[0] < 5 or away_matches.shape[0] < 5:
            return None
        
        data = {}
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
                data[f'home_{stats}_{i-2}-{i}'], data[f'away_{stats}_{i-2}-{i}'], \
                    data[f'{stats}_diff_{i-2}-{i}'] = self._mean_with_dispersion_penalty(
                        game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], stats)
                
                data[f'home_opp_{stats}_{i-2}-{i}'], data[f'away_opp_{stats}_{i-2}-{i}'], \
                    data[f'opp_{stats}_diff_{i-2}-{i}'] = self._mean_with_dispersion_penalty(
                        game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], stats, stats_against=True)
            
            data[f'home_poss_{i-2}-{i}'], data[f'away_poss_{i-2}-{i}'], data[f'poss_diff_{i-2}-{i}'] = \
                self._mean_with_dispersion_penalty(game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], 'poss')
            
            data[f'home_opp_positions_{i-2}-{i}'], data[f'away_opp_positions_{i-2}-{i}'], \
            data[f'opp_positions_diff_{i-2}-{i}']  = self._mean_with_dispersion_penalty(
                game, home_matches.iloc[i-3:i], away_matches.iloc[i-3:i], 'position', stats_against=True)
   
            data[f'points_per_match_diff'] = self._all_points_per_match(game.home_name, game, game.home_points) - \
                self._all_points_per_match(game.away_name, game, game.away_points)
            
            data[f'home_points_per_match_{i-2}-{i}'] = self._points_per_match_in_3_games(
                home_matches.iloc[i-3:i], game.home_name)
            
            data[f'away_points_per_match_{i-2}-{i}'] = self._points_per_match_in_3_games(
                away_matches.iloc[i-3:i], game.away_name)

        data['home_position'] = game.home_position
        data['away_position'] = game.away_position
        data['position_diff'] = game.position_diff

        data['home_formation'] = self._get_most_frequent(game.home_name, home_matches, 'formation')
        data['away_formation'] = self._get_most_frequent(game.away_name, away_matches, 'formation')

        data['home_defenders_num'] = self._get_most_frequent(game.home_name, home_matches, 'defenders_num')
        data['away_defenders_num'] = self._get_most_frequent(game.away_name, away_matches, 'defenders_num')

        data['home_coach_matches'] = self._coach_matches(home_matches, game.home_name, game.home_coach)
        data['away_coach_matches'] = self._coach_matches(away_matches, game.away_name, game.away_coach)

        data['result'] = game.result
        data['home_name'] = game.home_name
        data['away_name'] = game.away_name
        data['match_date'] = game.match_date

        data['sts_home'], data['sts_draw'], data['sts_away'] = game.sts_home, game.sts_draw, game.sts_away
        data['fortuna_home'], data['fortuna_draw'], data['fortuna_away'] = \
            game.fortuna_home, game.fortuna_draw, game.fortuna_away
        data['superbet_home'], data['superbet_draw'], data['superbet_away'] = \
            game.superbet_home, game.superbet_draw, game.superbet_away

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
    def _mean_with_dispersion_penalty(game, home, away, col, stats_against=False):
        all_stats = []
        order = ('away', 'home') if stats_against else ('home', 'away')

        for team, matches in ((game.home_name, home), (game.away_name, away)):
            stats = np.concat([matches[matches['home_name'] == team][f'{order[0]}_{col}'], 
                      matches[matches['away_name'] == team][f'{order[1]}_{col}']])
            all_stats.append(stats)
            
        all_stats = [x.mean() - sqrt(x.max() - x.min()) for x in all_stats]
        return all_stats[0], all_stats[1], all_stats[0] - all_stats[1]

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
    '''for row in dp.data.itertuples(index=True):
        home_games = dp._previous_matches(row, row.home_name)
        away_games = dp._previous_matches(row, row.away_name)
        ret = dp._opponents_positions_feture(row.home_name, home_games), dp._opponents_positions_feture(row.away_name, away_games)
        ret_b = dp._mean_with_dispersion_penalty(row, home_games, away_games, 'position', stats_against=True)
        print(ret)
        print(ret_b)
        break'''
    dp.preprocess_data()