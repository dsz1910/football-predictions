from page_interactor import PageInteractor
from match_links import GetLinks
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep, perf_counter
import pandas as pd
import numpy as np
import multiprocessing as mp


class ScrapeStatistics(PageInteractor):
    
    def __init__(self, matches):
        self.matches = matches
        self.data = self._read_data_from_file()

    def get_all_stats(self):
        driver = webdriver.Firefox()
        #i = 0
        for url, season_idx in self.matches:
            #if i == 10:
             #   break
            stats = self._get_match_stats(driver, url, season_idx)
            self.data = pd.concat([self.data, pd.DataFrame([stats])], ignore_index=True)
            #i += 1

        self._save_stats()

    def _read_data_from_file(self):
        if 1:
            columns = ['home_goals', 'home_poss', 'home_xG', 'home_passes', 'home_acc_passes', 
                       'home_shots', 'home_acc_shots', 'home_excluded_count', 'home_corners', 
                       'home_goalkeeper_saves', 'home_free_kicks', 'home_offsides', 
                       'home_fouls', 'home_mean_raitng', 'home_formation', 'home_inacc_shots',

                       'away_goals', 'away_poss', 'away_xG', 'away_passes', 'away_acc_passes',
                       'away_shots', 'away_acc_shots', 'away_excluded_count', 'away_corners',
                       'away_goalkeeper_saves', 'away_free_kicks', 'away_offsides', 
                       'away_fouls', 'away_mean_raitng', 'away_formation', 'away_inacc_shots',

                       'result', 'match_date', 'round', 'league', 'season'
                       ]
            return pd.DataFrame(columns=columns)
        
    def _get_match_stats(self, driver, url, season_idx):
        self.get_website(driver, url)
        final_data = {}
        
        sleep(1)
        if self.is_element_present(driver, By.ID, 'onetrust-reject-all-handler'):
            self.wait_and_click_button(driver, By.ID, 'onetrust-reject-all-handler')

        self.wait_until_element_is_visible(driver, By.CLASS_NAME, 'wcl-category_ITphf')
        all_stats = self.find_elements(driver, By.CLASS_NAME, 'wcl-category_ITphf')
        all_stats = [stats.text.split('\n') for stats in all_stats]
        match_stats = {stats[1] : (stats[0], stats[2]) for stats in all_stats}

        stats_category = {'Posiadanie piłki': 'poss', 'Oczekiwane bramki (xG)': 'xG', 'Sytuacje bramkowe': 'shots',
                          'Strzały na bramkę': 'acc_shots', 'Strzały niecelne': 'inacc_shots', 'Podania': 'passes',
                          'Rzuty rożne': 'corners', 'Interwencje bramkarzy': 'goalkeeper_saves',
                          'Rzuty wolne': 'free_kicks', 'Spalone': 'offsides', 'Faule' : 'fouls'}
        
        for key, value in stats_category.items():
            if key == 'Podania':
                final_data[f'home_{value}'], final_data[f'home_acc_{value}'], \
                    final_data[f'away_{value}'], final_data[f'away_acc_{value}'] = \
                        self._get_passes(match_stats)
                
            elif key == 'Posiadanie piłki':
                final_data[f'home_{value}'], final_data[f'away_{value}'] = self._get_possesion(match_stats[key])

            else:
                final_data[f'home_{value}'], final_data[f'away_{value}'] = self._split_home_and_away(
                    match_stats, key, float if key == 'Oczekiwane bramki (xG)' else int)

        final_data['season'] = season_idx
        final_data['match_date'] = self._get_time(driver)
        final_data['league'], final_data['round'] = self._get_league_and_round(driver)
        final_data['home_goals'], final_data['away_goals'] = self._get_goals(driver)
        final_data['result'] = self._get_result(final_data['home_goals'], final_data['away_goals'])

        url = url.replace('statystyki-meczu/0', 'sklady')
        self.get_website(driver, url)

        final_data['home_formation'], final_data['away_formation'] = self._get_formation(driver)
        final_data['home_mean_raiting'], final_data['away_mean_raiting'] = self._get_mean_raiting(driver)
        final_data['home_excluded_count'], final_data['away_excluded_count'] = self._get_excluded_players_count(
            driver)
        
        return final_data
    
    def _get_excluded_players_count(self, driver):
        if not self.is_element_present(
            driver, By.CSS_SELECTOR, '.wcl-caption_xZPDJ.wcl-scores-caption-05_f2TCB.wcl-description_iZZUi'):
            return 0, 0
        
        excluded = self.find_elements(
            driver, By.CSS_SELECTOR, '.wcl-caption_xZPDJ.wcl-scores-caption-05_f2TCB.wcl-description_iZZUi')
        excluded = [player.location['x'] for player in excluded]
        home_excluded_count = excluded.count(320)
        awaay_excluded_count = len(excluded) - home_excluded_count
        return home_excluded_count, awaay_excluded_count


    def _get_mean_raiting(self, driver):
        self.wait_until_element_is_visible(driver, By.CSS_SELECTOR, '[data-testid="wcl-scores-caption-05"]')
        raitings = self.find_elements(driver, By.CSS_SELECTOR, '[data-testid="wcl-scores-caption-05"]')
        return float(raitings[0].text), float(raitings[1].text) 

    def _get_formation(self, driver):
        sleep(1)
        self.wait_until_element_is_visible(driver, By.XPATH,
        """//*[contains(@class, 'wcl-headerSection') 
        and contains(@class, 'wcl-text') and contains(@class, 'wcl-spaceBetween')]""")

        formations = self.find_element(
            driver, By.XPATH, """//*[contains(@class, 'wcl-headerSection') 
        and contains(@class, 'wcl-text') and contains(@class, 'wcl-spaceBetween')]""").text
        
        formations = formations.split(('\n'))
        return formations[0], formations[2]


    @staticmethod
    def _get_result(home, away):
        return (home < away) + (home != away)
    
    def _get_goals(self, driver):
        result = self.find_element(driver, By.CLASS_NAME, 'detailScore__wrapper').text
        dash = result.find('-')
        home_goals = int(result[ : dash - 1])
        away_goals = int(result[dash + 2 : ])
        return home_goals, away_goals

    def _get_league_and_round(self, driver):
        league_and_round = self.find_element(driver, By.CLASS_NAME, 'tournamentHeader__country').text

        round_start = league_and_round.find('KOLEJKA')
        league_round = np.nan if round_start == -1 else int(league_and_round[round_start + 8 : ])

        league_start = league_and_round.find(':') + 2
        league_end = league_and_round.find('-') - 1
        league_name = league_and_round[league_start : league_end]

        return league_name, league_round

    def _get_time(self, driver):
        self.wait_until_element_is_visible(driver, By.CLASS_NAME, 'duelParticipant__startTime')
        start_time = self.find_element(driver, By.CLASS_NAME, 'duelParticipant__startTime').text
        return start_time

    @staticmethod
    def _split_home_and_away(data, key, to_type=int):
        if key in data:
            return to_type(data[key][0]), to_type(data[key][1])
        return np.nan, np.nan
    
    @staticmethod
    def _get_passes(stats):
        if 'Podania' not in stats:
            return np.nan, np.nan, np.nan, np.nan
        
        passes = []
        for team in stats['Podania']:
            left_paren = team.find('(')
            slash = team.find('/')
            passes.append(int(team[slash+1 : -1])) # add all passes done by team
            passes.append(int(team[left_paren+1 : slash])) # add accurate passes count
        return passes
    
    @staticmethod
    def _get_possesion(poss):
        return int(poss[0][:-1]), int(poss[1][:-1])

    def _save_stats(self):
        self.data.to_csv('stats.csv', index=False, encoding='utf-8')
        #self.data.to_excel('stats.xlsx', index=False, engine='openpyxl')


if __name__ == '__main__':
    driver = webdriver.Firefox()
    links_scraper = GetLinks(driver, ['https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa-2023-2024/wyniki/',
                        ])
    
    links_scraper.get_all_links()

    start = perf_counter()
    stats_scraper = ScrapeStatistics(links_scraper.links)
    stats_scraper.get_all_stats()
    end = perf_counter()
    print("pobieranie statystyk trwało: ", end - start)

    print(stats_scraper.data)