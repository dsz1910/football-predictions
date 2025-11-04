from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from time import sleep, perf_counter
from page_interactor import PageInteractor
import pickle


class GetLinks(PageInteractor):

    def __init__(self, driver, leagues_file):
        self.leagues_file = leagues_file
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

    def _read_leagues_to_scrape(self):
        with open(self.leagues_file, 'r') as file:  # file with links to leagues
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
                    '.wclButtonLink.wcl-buttonLink_jmSkY.wcl-primary_aIST5.wcl-underline_rL72U'):
                    #'.wclButtonLink.wcl-buttonLink_crZm3.wcl-primary_un6zG.wcl-underline_viepI'
                
                # click show more matches
                try:
                    self.wait_and_click_button(self.driver, By.CSS_SELECTOR,
                        '.wclButtonLink.wcl-buttonLink_jmSkY.wcl-primary_aIST5.wcl-underline_rL72U')
                except:
                    continue

                sleep(1)
            else:
                break

        # get matches links
        matches = self.find_elements(self.driver, By.CSS_SELECTOR, 'a.eventRowLink')

        for link in matches:
            link = link.get_attribute('href')
            put_index = link.index('?m')
            self.links.append((link[ : put_index] + 'szczegoly/statystyki/0/' + link[put_index : ], idx))


if __name__ == '__main__':
    start = perf_counter()

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=ąćęó')
    driver = webdriver.Chrome(options=options)

    leagues_file = 'leagues_to_scrape_2_set.txt'
    links_getter = GetLinks(driver, leagues_file)
    links_getter.get_all_links()
    end = perf_counter()
    print(end - start)