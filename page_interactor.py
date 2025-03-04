from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException


class PageInteractor:
    
    def get_website(self, driver, url):
        while True:
            try:
                driver.get(url)
                break
            except:
                pass

    def quit_website(self, driver):
        driver.quit()

    def wait_until_element_is_not_visible(self, driver, by_what, text, time=30):
        WebDriverWait(driver, time).until(
            EC.invisibility_of_element_located((by_what, text)))
        
    def wait_until_element_is_visible(self, driver, by_what, text, time=30):
                WebDriverWait(driver, time).until(
                    EC.visibility_of_element_located((by_what, text)))

    def wait_and_click_button(self, driver, by_what, text, time=30):
        button = WebDriverWait(driver, time).until(
            EC.element_to_be_clickable((by_what, text)))
        button.click()
        return button
    
    def wait_until_element_present_on_dom(self, driver, by_what, text, time=30):
        value = WebDriverWait(driver, time).until(
                EC.presence_of_element_located((by_what, text)))
        return value

    def is_element_present(self, driver, by_what, text):
        try:
            driver.find_element(by_what, text)
            return True
        except (NoSuchElementException, StaleElementReferenceException):
            return False
        
    def find_element(self, driver, by_what, text):
        return driver.find_element(by_what, text)
    
    def find_elements(self, driver, by_what, text):
        return driver.find_elements(by_what, text)