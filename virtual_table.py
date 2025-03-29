import pandas as pd
import numpy as np


class VirtualTable:

    def __init__(self):
        self.stats = pd.read_csv('stats.csv')
        self.stats['match_date'] = pd.to_datetime(self.stats['match_date'], format='%d.%m.%Y %H:%M')

    def _create_table(self, season):
        names = self.stats[self.stats['season'] == season]['home_name'].unique()
        table = {name : {'points' : 0, 'position' : np.nan} for name in names}
        return table

    def simulate_all_seasons(self):
        for i in range(1, self.stats['season'][len(self.stats) - 1] + 1):
            self._simulate_season(i)

    def _update_season(self, table):
        table = sorted(table.items(), key=lambda item: item[1]['points'], reverse=True)
        table = {item[0] : {'position' : i + 1, 'points' : item[1]['points']} for i, item in enumerate(table)}
        return table

    def _simulate_season(self, season):
        table = self._create_table(season)
        season_stats = self.stats[self.stats['season'] == season].copy()
        season_stats = season_stats.sort_values(by='match_date')
        previous_round = 1

        for game in season_stats.itertuples(index=True):
            curr_round = game.round
            if curr_round > previous_round:
                table = self._update_season(table)

            try:
                home = table[game.home_name]
                away = table[game.away_name]
                season_stats.loc[game.Index, ['home_position', 'home_points']] = home['position'], home['points']
                season_stats.loc[game.Index, ['away_position', 'away_points']] = away['position'], away['points']

                match game.result:
                    case 1:
                        table[game.home_name]['points'] += 3
                    case 0:
                        table[game.away_name]['points'] += 1
                        table[game.home_name]['points'] += 1
                    case 2:
                        table[game.away_name]['points'] += 3
            except KeyError:
                pass
            previous_round = curr_round

        self.stats = self.stats.combine_first(season_stats)

            
if __name__ == '__main__':
    t = VirtualTable()
    t.simulate_all_seasons()
    print(t.stats[(t.stats['season'] == 10) & t.stats['round']][['home_points', 'home_name', 'away_points', 'away_name', 'round', 'league']])