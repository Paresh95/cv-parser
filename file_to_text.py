import PyPDF2
from PIL import Image
import pytesseract


class ExtractText:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def from_pdf(self) -> str:
        with open(self.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text

    def from_image(self) -> str:
        return pytesseract.image_to_string(Image.open(self.file_path))

    def from_file_path(self):
        if self.file_path.endswith(".pdf"):
            return self.from_pdf()
        elif self.file_path.endswith(".png"):
            return self.from_image()
