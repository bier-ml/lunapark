# from linkedin_scraper import Person, actions

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
import re

from PyPDF2 import PdfReader


class PDFToText:
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
            text += page.extract_text()
        return self._clean_resume_text(text)

    def _clean_resume_text(self, text: str) -> str:
        """Clean and normalize text extracted from a resume PDF.

        Handles common PDF parsing issues like:
        - Broken words due to hyphenation
        - Inconsistent spacing and line breaks
        - Special characters and encoding issues
        - Common PDF artifacts
        - Multiple spaces and formatting issues

        Args:
            text: Raw text extracted from PDF

        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""

        # Fix common encoding issues
        text = text.encode("ascii", "ignore").decode("ascii")

        # Fix hyphenated words split across lines
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # Replace various types of dashes/hyphens with standard dash
        text = re.sub(r"[˗‐‑‒–—―]", "-", text)

        # Replace multiple types of quotes with standard quotes
        text = re.sub(r"[" "‛‚]", "'", text)
        text = re.sub(r'[""‟„]', '"', text)

        # Replace various whitespace characters with a single space
        text = re.sub(
            r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", text
        )

        # Remove unwanted characters and symbols
        text = re.sub(r"[<>\\|{}[\]~`]", "", text)

        # Fix common PDF artifacts
        text = re.sub(r"•", "", text)  # Remove bullet points
        text = re.sub(r"©|®|™", "", text)  # Remove common symbols

        # Normalize whitespace
        text = re.sub(r"\s*\n\s*", "", text)  # Replace newlines with space
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space

        # Fix common date formats (ensure consistent spacing)
        text = re.sub(r"(\d{4})\s*/\s*(\d{4})", r"\1-\2", text)

        # Remove extra spaces around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Final cleanup
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
