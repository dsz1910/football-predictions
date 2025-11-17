from virtual_table import VirtualTable
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from time import perf_counter
import pickle


class DataPreprocessor:

    def __init__(self, data_as_time_series):
        self.data_as_time_series = data_as_time_series
        vt = VirtualTable()
        vt.simulate_all_seasons()
        self.data = vt.stats
    
    def preprocess_data(self):
        self._clean_data()
        self._fill_lack_of_data()
        self._change_formation_to_numeric_data()

    def _fill_lack_of_data(self, all_cols=False, n_estimators=300, max_depth=5, max_iter=30, 
                           n_nearest_features=40, min_samples_leaf=50, min_samples_split=100):
        
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=n_estimators, 
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
            bootstrap=True, n_jobs=-1, max_features='sqrt'), max_iter=max_iter, initial_strategy='mean', 
            imputation_order='ascending', skip_complete=True, n_nearest_features=n_nearest_features, 
            random_state=42, verbose=2)
        
        non_imputed_columns = ['home_name', 'away_name', 'result', 'match_date']
        
        if not all_cols:
            non_imputed_columns = ['away_coach', 'away_formation', 'away_name', 'home_coach', 
                            'home_formation', 'home_name', 'league', 'match_date', 'round', 'match_date']

        cols_to_impute = [col for col in self.data.columns if col not in non_imputed_columns]
        data_to_impute = self.data[cols_to_impute].copy()

        imputed_data = imputer.fit_transform(data_to_impute)
        imputed_data = pd.DataFrame(imputed_data, columns=cols_to_impute, index=self.data.index)
        self.data[cols_to_impute] = imputed_data

    def _clean_data(self):
        self.data = self.data.loc[~self.data['home_name'].str.contains(
            '(', na=False, regex=False)]
        self.data = self.data.loc[self.data['home_poss'].notnull()]
        self.data = self.data.loc[self.data.groupby('home_name').home_name.transform('count') > 3]
        self.data = self.data.loc[self.data.groupby('away_name').away_name.transform('count') > 3]

    def _change_formation_to_numeric_data(self):
        if not self.data_as_time_series:
            self.data['home_defenders_num'] = self.data['home_formation'].apply(
                lambda x: int(x[0]) if pd.notna(x) else np.nan).astype('Int64')
                
            self.data['away_defenders_num'] = self.data['away_formation'].apply(
                lambda x: int(x[0]) if pd.notna(x) else np.nan).astype('Int64')

        self.data['home_formation'] = self.data['home_formation'].apply(
            lambda x: int(x.replace(' - ', '')) if pd.notna(x) else 0).astype('int64')
        
        self.data['away_formation'] = self.data['away_formation'].apply(
            lambda x: int(x.replace(' - ', '')) if pd.notna(x) else 0).astype('int64')
        
    def _coach_matches(self, games, team, coach, time_series_preprocessing=False):
        if not time_series_preprocessing:
            return games[
                ((games['home_coach'] == coach) & (games['home_name'] == team)) |
                ((games['away_coach'] == coach) & (games['away_name'] == team))
                ].shape[0]
        
        games['coach'] = np.where(games['home_name'] == team,
                                              games['home_coach'],
                                              games['away_coach'])

        games['rivals_name'] = np.where(games['home_name'] != team,
                                        games['home_name'],
                                        games['away_name'])
        
        games['rivals_coach'] = np.where(games['home_name'] == games['rivals_name'],
                                         games['home_coach'],
                                         games['away_coach'])
        
        games.sort_values(by='match_date', inplace=True)
        games['coach_matches'] = games.groupby('coach').cumcount() + 1
        games['rivals_coach_matches'] = games.apply(
            lambda row: self._get_rivals_coach_matches(
                row['rivals_name'], row['season'], row['rivals_coach'], row['match_date']), 
            axis=1
            )
        games.drop(['rivals_name', 'rivals_coach', 'coach'], axis=1, inplace=True)

    def _save_data(self):
        if self.data_as_time_series:
            with open('time_series_dataset.pkl', 'wb') as file:
                pickle.dump(self.data, file)
        else:
            self.data.to_csv('stats_dataset.csv', index=False, encoding='utf-8')
            self.data.to_excel('stats_dataset.xlsx', index=False, engine='openpyxl')


class TimeSeriesPreprocessor(DataPreprocessor):

    def __init__(self, min_time_series_length, coach_data):
        super().__init__(True)
        self.min_time_series_length = min_time_series_length
        self.coach_data = coach_data

    def preprocess_data(self):
        super().preprocess_data()
        self._get_all_time_series()
        self._save_data()

    def _set_result_from_team_perspectives(self):
        self.data['result_from_home_perspective'] = self.data['result'].replace(2, -1, inplace=False)
        self.data['result_from_away_perspective'] = -self.data['result_from_home_perspective']

    def _get_result_from_team_perspective(self, team, team_time_series):
        mask = team_time_series['home_name'] == team

        team_time_series['result_from_team_perspective'] = np.where(
            mask,
            team_time_series['result_from_home_perspective'],
            team_time_series['result_from_away_perspective']
            )
        
        team_time_series['at_home'] = mask.astype(int)

    def _extract_static_data(self, ts):
        row = ts.loc[ts['match_date'].idxmax(), :]
        ts.drop(index=row.name, inplace=True)
        ts.drop('result', axis=1, inplace=True)

        static = {'home_points' : row['home_points'],
                  'home_position' : row['home_position'],
                  'away_points' : row['away_points'],
                  'away_position' : row['away_position'],
                  'goals_scored_ratio' : None,
                  'goals_conceded_ratio' : None,
                  'rating_ratio' : None,
                  'xG_ratio' : None,
                  'result' : row['result']
                  }
        
        home_previous_games = self._get_previous_matches_from_data_frame(
            row['home_name'], row['season'], row['match_date'])
        away_previous_games = self._get_previous_matches_from_data_frame(
            row['away_name'], row['season'], row['match_date'])
        
        if not home_previous_games.shape[0] or not away_previous_games.shape[0]:
            return None
        
        home_goals_sum = self._sum_choosen_stats(row['home_name'],'goals', home_previous_games)
        away_goals_sum = self._sum_choosen_stats(row['away_name'], 'goals', away_previous_games)
        static['goals_scored_ratio'] = home_goals_sum / away_goals_sum if away_goals_sum else home_goals_sum

        home_goals_conceded = self._sum_choosen_stats(
            row['home_name'], 'goals', home_previous_games, against=True)
        away_goals_conceded = self._sum_choosen_stats(
            row['away_name'], 'goals', away_previous_games, against=True)
        static['goals_conceded_ratio'] = home_goals_conceded / away_goals_conceded if away_goals_conceded \
                                                                                    else home_goals_conceded

        home_xg_sum = self._sum_choosen_stats(row['home_name'], 'xG', home_previous_games)
        away_xg_sum = self._sum_choosen_stats(row['away_name'], 'xG', away_previous_games)
        static['xG_ratio'] = home_xg_sum / away_xg_sum if away_xg_sum else home_xg_sum

        home_raiting_mean = self._choosen_stats_mean(
            row['home_name'], 'mean_rating', home_previous_games)
        away_raiting_mean = self._choosen_stats_mean(
            row['away_name'], 'mean_rating', away_previous_games)
        static['raiting_ratio'] = home_raiting_mean / away_raiting_mean
            
        static = pd.DataFrame(static, index=[0])
        ts_and_static = [static, ts]

        return ts_and_static
    
    def _get_previous_matches_from_data_frame(self, team, season, date):
        games = self.data.loc[
                (self.data['season'] == season) &
                (self.data['match_date'] < date) &
                ((self.data['home_name'] == team) | (self.data['away_name'] == team))
                ]
        return games

    def _choosen_stats_mean(self, team, stats, games, against=False):
        stats_sum = self._sum_choosen_stats(team, stats, against=against, games=games)
        return stats_sum / games.shape[0]
    
    def _sum_choosen_stats(self, team, stats, games, against=False):
        if against:
            return games[games['home_name'] == team][f'away_{stats}'].sum() + \
                games[games['away_name'] == team][f'home_{stats}'].sum()
        
        return games[games['home_name'] == team][f'home_{stats}'].sum() + \
            games[games['away_name'] == team][f'away_{stats}'].sum()

    def _create_rolling_datasets_for_team(self, df, season):
        time_series_collection = list(df.rolling(window=55))
        time_series_collection = {
            (ts.loc[ts['match_date'].idxmax(), 'match_date'],
             ts.loc[ts['match_date'].idxmax(), 'home_name'],
             season
             ) : ts
              for ts in time_series_collection
              }
        return time_series_collection

    def _get_all_time_series(self):
        all_time_series = {}
        self._set_result_from_team_perspectives()
        
        for season in range(int(self.data['season'].min()), int(self.data['season'].max() + 1)):
            season_time_series = self._get_time_series_for_season(season)
            all_time_series.update(season_time_series)

        self._delete_time_series_with_modest_data(all_time_series)
        self.data = all_time_series

    def _delete_time_series_with_modest_data(self, data):
        to_remove = []

        for key, val in data.items():
            if len(val[1]) < self.min_time_series_length or len(val[2]) < self.min_time_series_length:
                to_remove.append(key)
        
        for key in to_remove:
            del data[key]

    def _get_time_series_for_season(self, season):
        season_time_series = {}
        season_matches = self.data.query('season == @season').copy()
        teams = season_matches['home_name'].unique()

        for team in teams:
            matches = season_matches.query('home_name == @team or away_name == @team').copy()
            team_time_series = self._get_time_series_for_team(team, matches, season)
            self._add_time_series_to_season(season_time_series, team_time_series)

        return season_time_series
    
    def _add_time_series_to_season(self, season_ts, team_ts):
        for key, val in team_ts.items():
            if key not in season_ts.keys():
                ret = self._extract_static_data(val)
                if ret is not None:
                    team_ts[key] = ret
                    season_ts[key] = [*team_ts[key]]
            else:
                row = team_ts[key].loc[team_ts[key]['match_date'] == key[0]]
                team_ts[key].drop(index=row.index, inplace=True)
                team_ts[key].drop('result', axis=1, inplace=True)

                if key[1] == row['home_name'].iloc[0]:
                    season_ts[key].insert(1, team_ts[key])
                else:
                    season_ts[key].append(team_ts[key])

    def _get_time_series_for_team(self, team, matches, season):
        matches.sort_values(by='match_date', inplace=True)
        self._get_result_from_team_perspective(team, matches)

        if self.coach_data:
            self._coach_matches(matches, team, None, time_series_preprocessing=True)

        matches.drop([
                      'result_from_home_perspective',
                      'league',
                      'result_from_away_perspective',
                      'away_poss',
                      'sts_home',
                      'sts_draw',
                      'sts_away',
                      'fortuna_home',
                      'fortuna_draw',
                      'fortuna_away',
                      'superbet_home',
                      'superbet_draw',
                      'superbet_away'
                      ],
                      axis=1,
                      inplace=True)
        
        if 'home_coach' in matches.columns:
            matches.drop(columns=['home_coach', 'away_coach'])
        
        team_time_series = self._create_rolling_datasets_for_team(matches, season)
        return team_time_series
    
    def _get_rivals_coach_matches(self, rival, season, coach_name, match_date):
        return self.data.loc[
            (self.data['season'] == season) &
            (self.data['match_date'] < match_date) &
            (((self.data['home_name'] == rival) & (self.data['home_coach'] == coach_name)) |
             ((self.data['away_name'] == rival) & self.data['away_coach'] == coach_name))
        ].shape[0]
    
class TimeSeriesPreprocessorForOldMatches(TimeSeriesPreprocessor):

    def __init__(self, min_time_series_length):
        super().__init__(min_time_series_length, False)


    def _fill_lack_of_data(self, all_cols=False, n_estimators=300, max_depth=5, max_iter=30,
                           n_nearest_features=40, min_samples_leaf=50, min_samples_split=100):
        
        self.data[['home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards']] = \
            self.data[['home_red_cards', 'away_red_cards', 'home_yellow_cards', 'away_yellow_cards']].fillna(0)

        super()._fill_lack_of_data(
            all_cols, n_estimators, max_depth, max_iter, n_nearest_features, min_samples_leaf, min_samples_split)
        
    def preprocess_data(self):
        self._clean_data()
        self._fill_lack_of_data()
        self._get_all_time_series()
        self._save_data()

    def _extract_static_data(self, ts):
        row = ts.loc[ts['match_date'].idxmax(), :]

        ts.drop(index=row.name, inplace=True)
        ts.drop('result', axis=1, inplace=True)

        home_previous_games = self._get_previous_matches_from_data_frame(
            row['home_name'], row['season'], row['match_date'])
        away_previous_games = self._get_previous_matches_from_data_frame(
            row['away_name'], row['season'], row['match_date'])
        
        if not home_previous_games.shape[0] or not away_previous_games.shape[0]:
            return None

        static = {'home_points' : row['home_points'],
                  'home_position' : row['home_position'],
                  'away_points' : row['away_points'],
                  'away_position' : row['away_position'],
                  'result' : row['result']
                  }

        static = pd.DataFrame(static, index=[0])
        ts_and_static = [static, ts]

        return ts_and_static

class GetGroupedStats(DataPreprocessor):

    def __init__(self):
        super().__init__(False)

    def preprocess_data(self):
        super().preprocess_data()
        self._extract_all_features_per_3_match_groups()
        self._fill_lack_of_data(all_cols=True, n_nearest_features=100)
        self._save_data()

    def _extract_all_features_per_3_match_groups(self):
        final_stats = []
        self.data['position_diff'] = self.data['home_position'] - self.data['away_position']

        for game in self.data.itertuples(index=True):
            result = self._extract_match_stats_per_3_match_groups(game)
            if result:
                final_stats.append(result)
            if game.Index % 1000 == 0:
                print(game.Index)

        self.final_data = final_stats
        self.data = pd.DataFrame(final_stats)

    def _extract_match_stats_per_3_match_groups(self, game):
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
            'mean_rating',
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
        matches_count = self.data[
            (self.data['match_date'] < game.match_date) & 
            (self.data['season'] == game.season) & 
            ((self.data['home_name'] == team) | (self.data['away_name'] == team))
            ].shape[0]
        return points / matches_count

    @staticmethod
    def _mean_with_dispersion_penalty(game, home, away, col, stats_against=False):
        all_stats = []
        order = ('away', 'home') if stats_against else ('home', 'away')

        for team, matches in ((game.home_name, home), (game.away_name, away)):
            stats = np.concatenate([matches[matches['home_name'] == team][f'{order[0]}_{col}'], 
                      matches[matches['away_name'] == team][f'{order[1]}_{col}']])
            all_stats.append(stats)
            
        all_stats = [x.mean() - np.std(x) for x in all_stats]
        return all_stats[0], all_stats[1], all_stats[0] - all_stats[1]

    @staticmethod
    def _stats_avg(game, home, away, col, stats_against=False):
        result = []
        order = ('away', 'home') if stats_against else ('home', 'away')

        for team, matches in ((game.home_name, home), (game.away_name, away)):
            result.append(
                ((matches[matches['home_name'] == team][f'{order[0]}_{col}'].sum() + 
                  matches[matches['away_name'] == team][f'{order[1]}_{col}'].sum()) /
                  matches.shape[0]))
        
        return result[0], result[1], result[0] - result[1]

    def _previous_matches(self, game, team, all_previous=True):
        matches = self.data.loc[
            (self.data['match_date'] < game.match_date) & 
            (self.data['season'] == game.season) &
            ((self.data['home_name'] == team) | (self.data['away_name'] == team))
            ].sort_values(by='match_date', ascending=False)
        
        if all_previous:
            return matches
        return matches.head(9).reset_index(drop=True)
    

if __name__ == '__main__':
    # get time series
    '''ts_preprocessor = TimeSeriesPreprocessor(5)
    start = perf_counter()
    ts_preprocessor.preprocess_data()
    end = perf_counter()
    print(f'Data preprocessing time: {end - start}')'''

    # get time series including old matches
    ts_preprocessor = TimeSeriesPreprocessorForOldMatches(3)
    start = perf_counter()
    ts_preprocessor.preprocess_data()
    end = perf_counter()
    print(f'Data preprocessing time: {end - start}')

    # group stats
    '''gs_preprocessor = GetGroupedStats()
    start = perf_counter()
    gs_preprocessor.preprocess_data()
    end = perf_counter()
    print(f'Data preprocessing time: {end - start}')'''