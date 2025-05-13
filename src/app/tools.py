# from linkedin_scraper import Person, actions

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
import io
import re
import tempfile
from typing import BinaryIO, Optional

import pypdf
from PyPDF2 import PdfReader


class PDFToText:
    """Utility class to extract text from PDF files."""

    def __init__(self, file):
        self.file = file
        self.reader = PdfReader(self.file)

    def extract_text(self) -> str:
        """Extract and clean text from the PDF.

        Returns:
            str: Cleaned text from the PDF
        """
        if not self.reader.pages:
            return ""

        text = ""
        for page in self.reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

        return self._clean_resume_text(text)

    def _clean_resume_text(self, text: str) -> str:
        """Clean text by removing unnecessary whitespace and characters.

        Args:
            text: Raw text from PDF

        Returns:
            str: Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)

        # Replace multiple newlines with a single newline
        text = re.sub(r"\n+", "\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


# class LinkedInProfileParser:
#     def __init__(self, email: str, password: str):
#         self.linkedin_email = email
#         self.linkedin_password = password
#         self.driver = self._init_driver()
#         self._login()

#     def _init_driver(self):
#         options = webdriver.Ch romeOptions()
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
