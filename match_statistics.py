from page_interactor import PageInteractor
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import pandas as pd
import numpy as np
import multiprocessing as mp


class GetStatistics(PageInteractor):
    
    def __init__(self, matches_url):
        self.matches_url = matches_url
        self.data = self._read_data_from_file()

    def get_all_stats(self):
        drivers = [webdriver.Firefox() for _ in range(5)]
        for driver in drivers:
            self.wait_and_click_button(driver, By.ID, 'onetrust-reject-all-handler')

    def _read_data_from_file(self):
        if 1:
            columns = ['home_goals', 'home_poss', 'home_xG', 'home_passes', 'home_acc_passes', 
                       'home_shots', 'home_acc_shots', 'home_excluded_count', 'home_corners', 
                       'home_goalkeeper_saves', 'home_free_kicks', 'home_offsides', 
                       'home_fouls', 'home_mean_raitng', 'home_formation', 'home_inacc_shots'

                       'away_goals', 'away_poss', 'away_xG', 'away_passes', 'away_acc_passes',
                       'away_shots', 'away_acc_shots', 'away_excluded_count', 'away_corners',
                       'away_goalkeeper_saves', 'away_free_kicks', 'away_offsides', 
                       'away_fouls', 'away_mean_raitng', 'away_formation', 'away_inacc_shots'

                       'result', 'match_date', 'round', 'league', 'season'
                       ]
            return pd.DataFrame(columns=columns)
        
    def _get_match_stats(self, driver, url):
        self.get_website(driver, url)
        final_data = {}
        
        sleep(0.5)
        if self.is_element_present(driver, By.ID, 'onetrust-reject-all-handler'):
            self.wait_and_click_button(driver, By.ID, 'onetrust-reject-all-handler')

        self.wait_until_element_is_visible(driver, By.CLASS_NAME, 'wcl-category_ITphf')
        all_stats = driver.find_elements(By.CLASS_NAME, 'wcl-category_ITphf')
        all_stats = [stats.text.split('\n') for stats in all_stats]
        match_stats = {stats[1] : (stats[0], stats[2]) for stats in all_stats}

        final_data['home_poss'], final_data['away_poss'] = self._get_possesion(match_stats['Posiadanie piłki'])

        if 'Oczekiwane bramki (xG)' in match_stats:
            final_data['home_xG'], final_data['away_xG'] = self._split_home_and_away(
                match_stats['Oczekiwane bramki (xG)'], to_type=float
            )
        else:
            final_data['home_xG'] = final_data['away_xG'] = np.nan

        final_data['home_shots'], final_data['away_shots'] = self._split_home_and_away(
            match_stats['Sytuacje bramkowe']
        )

        final_data['home_acc_shots'], final_data['away_acc_shots'] = self._split_home_and_away(
            match_stats['Strzały na bramkę']
        )

        final_data['home_inacc_shots'], final_data['away_inacc_shots'] = self._split_home_and_away(
            match_stats['Strzały niecelne']
        )

        if 'Podania' in match_stats.keys():
                final_data['home_passes'], final_data['away_passes'], \
                final_data['home_acc_passes'], final_data['away_acc_passes'] = \
                    self._get_passes(match_stats['Podania'])
        else:
                final_data['home_passes'] = final_data['away_passes'] = \
                    final_data['home_acc_passes'] = final_data['away_acc_passes'] = np.nan

        final_data['home_corners'], final_data['away_corners'] = self._split_home_and_away(
            match_stats['Rzuty rożne'])

        return final_data
    
    @staticmethod
    def _split_home_and_away(data, to_type=int):
        return to_type(data[0]), to_type(data[1])

    @staticmethod
    def _get_inacc_shots(shots):
        return int(shots[0]), int(shots[1])
    
    @staticmethod
    def _get_passes(passes):
        all_passes_per_team = []
        acc_passes_per_team = []
        for team in passes:
            left_paren = team.find('(')
            slash = team.find('/')
            acc_passes_per_team.append(team[left_paren+1 : slash])
            all_passes_per_team.append(team[slash+1 : -1])
        
        return all_passes_per_team[0], all_passes_per_team[1], acc_passes_per_team[0], acc_passes_per_team[1]
    
    @staticmethod
    def _get_acc_shots(shots):
        return int(shots[0]), int(shots[1])
    
    @staticmethod
    def _get_shots(shots):
        return int(shots[0]), int(shots[1])
        
    @staticmethod
    def _get_xg(xg):
        return float(xg[0]), float(xg[1])
    
    @staticmethod
    def _get_possesion(poss):
        return int(poss[0][:-1]), int(poss[1][:-1])

    def _save_stats(self):
        pass


if __name__ == '__main__':
    stats_getter = GetStatistics([
        'https://www.flashscore.pl/mecz/xQyu2gQ7/#/szczegoly-meczu/statystyki-meczu/0',
        'https://www.flashscore.pl/mecz/U5ierGcp/#/szczegoly-meczu/statystyki-meczu/0',
        ])
    
    driver = webdriver.Firefox()
    for url in stats_getter.matches_url:
        data = stats_getter._get_match_stats(driver, url)
        for key, value in data.items():
            print(key, ': ', value)
        print()
    driver.quit()