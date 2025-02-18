from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
from page_interactor import PageInteractor


class GetLinks(PageInteractor):

    def __init__(self, driver, leagues_to_scrape):
        self.driver = driver
        self.links = {league : None for league in leagues_to_scrape}
        self.leagues_to_scrape = leagues_to_scrape
    
    def get_all_links(self):
        for league in self.leagues_to_scrape:
            self._get_league_matches_links(league)
        self.quit_website(self.driver)
        
    def _get_league_matches_links(self, league):
        self.get_website(self.driver, league)

        if league == self.leagues_to_scrape[0]:
            # click cookies
            self.wait_and_click_button(self.driver, By.ID, 'onetrust-reject-all-handler')

            # waiting until cookies get disappeared
            self.wait_until_element_is_not_visible(self.driver, By.ID, 'onetrust-reject-all-handler')

        while True:
            if self.is_element_present(self.driver, By.CSS_SELECTOR, '.event__more.event__more--static'):
                self.wait_until_element_is_not_visible(self.driver, By.CLASS_NAME, 'loadingOverlay')
                
                # Waiting until 'show more matches' button shows up in DOM
                #self.wait_until_element_present_on_dom(self.driver, By.CSS_SELECTOR, '.event__more.event__more--static')

                # click show more matches
                self.wait_and_click_button(self.driver, By.CSS_SELECTOR, '.event__more.event__more--static')
                sleep(0.5)
            else:
                break

        # get matches links
        matches = self.driver.find_elements(By.CSS_SELECTOR, 'a.eventRowLink')
        links = [link.get_attribute('href') + '/statystyki-meczu/0' for link in matches]
        self.links[league] = links


if __name__ == '__main__':
    driver = webdriver.Firefox()
    leagues_to_scrap = ['https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/hiszpania/laliga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/wlochy/serie-a-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/niemcy/bundesliga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/francja/ligue-1-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/championship-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/league-one-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/league-two-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/niemcy/2-bundesliga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/europa/liga-mistrzow/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/francja/ligue-2-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/hiszpania/laliga2-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/europa/liga-europejska/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/europa/liga-konfetrencji/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/austria/bundesliga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/belgia/jupiler-league-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/chorwacja/hnl-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/holandia/eredivisie-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/dania/superliga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/portugalia/liga-portugal-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/turcja/super-lig-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/brazylia/serie-a-betano-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/brazylia/serie-a-betano-2023/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/premier-league/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/championship/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/league-one/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/anglia/league-two/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/hiszpania/laliga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/wlochy/serie-a/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/wlochy/serie-b/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/wlochy/serie-b-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/wlochy/serie-b-2022-2023/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/wlochy/serie-a-2022-2023/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/niemcy/bundesliga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/francja/ligue-1/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/niemcy/2-bundesliga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/austria/bundesliga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/belgia/jupiler-league/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/chorwacja/hnl/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/holandia/eredivisie/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/dania/superliga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/portugalia/liga-portugal/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/turcja/super-lig/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/brazylia/serie-b/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/arabia-saudyjska/pierwsza-liga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/arabia-saudyjska/pierwsza-liga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/australia/a-league-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/australia/a-league/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/egipt/pierwsza-liga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/grecja/super-league-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/grecja/super-league/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/izrael/ligat-ha-al/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/japonia/j1-league-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/slowacja/nike-liga/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/slowacja/nike-liga-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/szwajcaria/super-league-2023-2024/wyniki/',
                        'https://www.flashscore.pl/pilka-nozna/szwajcaria/super-league/wyniki/'
                        ]

    links_getter = GetLinks(driver, leagues_to_scrap)
    links_getter.get_all_links()
    
    print(sum([len(value) for value in links_getter.links.values()]))