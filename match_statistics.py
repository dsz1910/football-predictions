from page_interactor import PageInteractor
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep, perf_counter
from queue import Queue
import pandas as pd
import numpy as np
import threading
import pickle
import functools


class ScrapeStatistics(PageInteractor):
    
    lock = threading.Lock()

    def __init__(self, threads_num=3):
        self.matches = self._read_links()
        self.data = []
        self.threads_num = threads_num
        self.task_queue = Queue()
        self.workers = None

    def error_handler(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):    
            for _ in range(5):
                try:
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    driver = kwargs['driver']
                    driver.get(driver.current_url)
                    print(e)
                    sleep(1)
            return None
        return wrapper

    def get_all_stats(self):
        for i, task in enumerate(self.matches):
            if i % 1000 == 0:
                self.task_queue.put((0, 0))
            self.task_queue.put(task)

        self.workers = [Worker(self.task_queue, self.data, self._get_match_stats, self.save_stats) for _ in range(self.threads_num)]

        for worker in self.workers:
            worker.start()
            
        for worker in self.workers:
            worker.join()

        for worker in self.workers:
            self.quit_website(worker.driver)

        self.data = pd.DataFrame(self.data)

        self.save_stats()
        self.save_stats('excel')
        
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
                
        final_data = self._get_stats_without_driver(match_stats, driver=driver)

        final_data['home_name'], final_data['away_name'] = self._get_team_names(driver=driver)
        final_data['season'] = season_idx
        final_data['match_date'] = self._get_time(driver=driver)
        final_data['league'], final_data['round'] = self._get_league_and_round(driver=driver)
        final_data['home_goals'], final_data['away_goals'] = self._get_goals(driver=driver)
        final_data['result'] = self._get_result(final_data['home_goals'], final_data['away_goals'])

        url = url.replace('statystyki-meczu/0', 'sklady')
        self.get_website(driver, url)

        final_data['home_formation'], final_data['away_formation'] = self._get_formation(driver=driver)
        final_data['home_mean_raiting'], final_data['away_mean_raiting'] = self._get_mean_raiting(driver=driver)
        final_data['home_excluded_count'], final_data['away_excluded_count'] = self._get_excluded_players_count(
            driver=driver)
                
        return final_data
    
    @error_handler
    def _get_stats_without_driver(self, match_stats, *, driver):
        data = {}
        stats_category = {'Posiadanie piłki': 'poss', 'Oczekiwane bramki (xG)': 'xG', 'Sytuacje bramkowe': 'shots',
                        'Strzały na bramkę': 'acc_shots', 'Strzały niecelne': 'inacc_shots', 'Podania': 'passes',
                        'Rzuty rożne': 'corners', 'Interwencje bramkarzy': 'goalkeeper_saves',
                        'Rzuty wolne': 'free_kicks', 'Spalone': 'offsides', 'Faule' : 'fouls'}
                
        for key, value in stats_category.items():
            if key == 'Podania':
                data[f'home_{value}'], data[f'home_acc_{value}'], \
                    data[f'away_{value}'], data[f'away_acc_{value}'] = \
                        self._get_passes(match_stats)
                        
            elif key == 'Posiadanie piłki':
                data[f'home_{value}'], data[f'away_{value}'] = self._get_possesion(match_stats[key])

            else:
                data[f'home_{value}'], data[f'away_{value}'] = self._split_home_and_away(
                match_stats, key, float if key == 'Oczekiwane bramki (xG)' else int)

        return data

    @error_handler
    def _get_team_names(self, *, driver):
        self.wait_until_element_is_visible(driver, By.CLASS_NAME, 
            'participant__participantName')
        
        names = self.find_elements(driver, By.CLASS_NAME, 'participant__participantName')
        return names[0].text, names[-1].text
    
    @error_handler
    def _get_excluded_players_count(self, *, driver):
        if not self.is_element_present(
            driver, By.CSS_SELECTOR, '.wcl-caption_xZPDJ.wcl-scores-caption-05_f2TCB.wcl-description_iZZUi'):
            return 0, 0
        
        excluded = self.find_elements(
            driver, By.CSS_SELECTOR, '.wcl-caption_xZPDJ.wcl-scores-caption-05_f2TCB.wcl-description_iZZUi')
        excluded = [player.location['x'] for player in excluded]
        home_excluded_count = excluded.count(320)
        awaay_excluded_count = len(excluded) - home_excluded_count
        return home_excluded_count, awaay_excluded_count

    @error_handler
    def _get_mean_raiting(self, *, driver):
        self.wait_until_element_is_visible(driver, By.CSS_SELECTOR, '[data-testid="wcl-scores-caption-05"]')
        raitings = self.find_elements(driver, By.CSS_SELECTOR, '[data-testid="wcl-scores-caption-05"]')
        return float(raitings[0].text), float(raitings[1].text) 

    @error_handler
    def _get_formation(self,*, driver):
        self.wait_until_element_is_visible(driver, By.XPATH,
        """//*[contains(@class, 'wcl-headerSection') 
        and contains(@class, 'wcl-text') and contains(@class, 'wcl-spaceBetween')]""")

        formations = self.find_element(
        driver, By.XPATH, """//*[contains(@class, 'wcl-headerSection') 
            and contains(@class, 'wcl-text') and contains(@class, 'wcl-spaceBetween')]""").text
        
        formations = formations.split(('\n'))
        formations = ['werf']
        return formations[0], formations[2]
    
    @staticmethod
    def _get_result(home, away):
        return (home < away) + (home != away)
    
    @error_handler
    def _get_goals(self, *, driver):
        result = self.find_element(driver, By.CLASS_NAME, 'detailScore__wrapper').text
        dash = result.find('-')
        home_goals = int(result[ : dash - 1])
        away_goals = int(result[dash + 2 : ])
        return home_goals, away_goals

    @error_handler
    def _get_league_and_round(self, *, driver):
        league_and_round = self.find_element(driver, By.CLASS_NAME, 'tournamentHeader__country').text

        round_start = league_and_round.find('KOLEJKA')
        league_round = np.nan if round_start == -1 else int(league_and_round[round_start + 8 : ])

        league_start = league_and_round.find(':') + 2
        league_end = league_and_round.find('-') - 1
        league_name = league_and_round[league_start : league_end]

        return league_name, league_round

    @error_handler
    def _get_time(self, *, driver):
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

    def save_stats(self, format='csv', data=None):
        if data is None:
            data = self.data
        data = pd.DataFrame(data)

        file_formats = {'csv' : lambda: data.to_csv('stats.csv', index=False, encoding='utf-8'),
                        'excel' : lambda: data.to_excel('stats.xlsx', index=False, engine='openpyxl')}
        file_formats.get(format)()

    def _read_links(self):
        with open('match_links_with_season_indexes.pkl', 'rb') as file:
            return pickle.load(file)

class Worker(threading.Thread):

    def __init__(self, task_queue, results, task, saver):
        super().__init__()
        self.driver = None
        self.task_queue = task_queue
        self.results = results
        self.task = task
        self.saver = saver

    def run(self):
        self.driver = webdriver.Chrome()

        while not self.task_queue.empty():
            match, season_idx = self.task_queue.get()
            if isinstance(match, str):
                result = self.task(self.driver, match, season_idx)
                with ScrapeStatistics.lock:
                    self.results.append(result)
            else:
                self.saver(data=self.results)
                self.saver(format='excel')
                print('saved data')
                
if __name__ == '__main__':
    start = perf_counter()
    stats_scraper = ScrapeStatistics(6)
    stats_scraper.get_all_stats()
    end = perf_counter()
    print("pobieranie statystyk trwało: ", end - start)
    stats_scraper.save_stats('excel')
    print(stats_scraper.data)