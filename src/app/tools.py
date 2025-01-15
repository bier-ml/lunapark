import time

# from linkedin_scraper import Person, actions
from pypdf import PdfReader

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager


class PDFToText:
    def __init__(self, file):
        self.file = file
        self.reader = PdfReader(self.file)

    def extract_text(self) -> str:
        text = ""
        for page in self.reader.pages:
            text += page.extract_text()
        return text


# class LinkedInProfileParser:
#     def __init__(self, email: str, password: str):
#         self.linkedin_email = email
#         self.linkedin_password = password
#         self.driver = self._init_driver()
#         self._login()

#     def _init_driver(self):
#         options = webdriver.ChromeOptions()
#         options.add_argument("--headless")  # Run browser in headless mode
#         return webdriver.Chrome(
#             service=Service(ChromeDriverManager().install()), options=options
#         )

#     def _login(self):
#         actions.login(self.driver, self.linkedin_email, self.linkedin_password)

#     def get_text_resume(self, url: str) -> str:
#         try:
#             person = Person(url, driver=self.driver)
#             time.sleep(3)  # Wait for the page to load
#             return str(person) or None
#         except Exception as e:
#             return f"Error: {e}"

#     def close(self):
#         self.driver.quit()


# Example usage:
# parser = LinkedInProfileParser(linkedin_email, linkedin_password)
# df['cv'] = df.progress_apply(parser.fill_cv, axis=1)
# parser.close()
