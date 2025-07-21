from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from time import sleep, perf_counter
from page_interactor import PageInteractor
import pickle


class GetLinks(PageInteractor):

    def __init__(self, driver):
        self.driver = driver
        self.links = []
        self.leagues_to_scrape = self._read_leagues_to_scrape()
        self._season_idx_generator = self._generate_season_idx()
    
    @staticmethod
    def _generate_season_idx():
        idx = 0
        while True:
            idx += 1
            yield idx

    def get_all_links(self):
        for league in self.leagues_to_scrape:
            self._get_league_matches_links(league)
        self.quit_website(self.driver)
        self._save_links()

    @staticmethod
    def _read_leagues_to_scrape():
        with open('leagues_to_scrape.txt', 'r') as file:
            leagues_lst = [line.strip() for line in file]
        return leagues_lst

    def _save_links(self):
        with open('match_links_with_season_indexes.pkl', 'wb') as file:
            pickle.dump(self.links, file)
        
    def _get_league_matches_links(self, league):
        self.get_website(self.driver, league)
        idx = next(self._season_idx_generator)

        if league == self.leagues_to_scrape[0]:
            # click cookies
            self.wait_and_click_button(self.driver, By.ID, 'onetrust-reject-all-handler')

            # waiting until cookies get disappeared
            self.wait_until_element_is_not_visible(self.driver, By.ID, 'onetrust-reject-all-handler')

        while True:

            if self.is_element_present(self.driver, By.CSS_SELECTOR,
                    '.wclButtonLink.wcl-buttonLink_crZm3.wcl-primary_un6zG.wcl-underline_viepI'):
                
                # click show more matches
                self.wait_and_click_button(self.driver, By.CSS_SELECTOR,
                    '.wclButtonLink.wcl-buttonLink_crZm3.wcl-primary_un6zG.wcl-underline_viepI')
                #self.wait_and_click_button(self.driver, By.CSS_SELECTOR, '.event__more.event__more--static')
                sleep(2)
            else:
                break

        # get matches links
        matches = self.find_elements(self.driver, By.CSS_SELECTOR, 'a.eventRowLink')
        new_links = [(link.get_attribute('href') + '/statystyki-meczu/0', idx) 
                     for link in matches]        # '/match-statistics/0'
        self.links += new_links


if __name__ == '__main__':
    start = perf_counter()

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=TwójUserAgent')
    driver = webdriver.Chrome(options=options)

    links_getter = GetLinks(driver)
    links_getter.get_all_links()
    end = perf_counter()
    print(end - start)