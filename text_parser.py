import re
from typing import Union
from transformers import pipeline


class SimpleTextParser:
    def __init__(self, text: str):
        self.text = text

    def get_name(self) -> Union[str, str]:
        full_name = self.text.split("\n")[0].strip().lower().split()
        first_name = full_name[0]
        last_name = "".join(full_name[1:])
        return first_name, last_name

    def get_emails(self) -> list:
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.findall(email_pattern, self.text)

    def get_phone_numbers(self) -> list:
        phone_number_pattern = r"""
        (\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}
        [-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"""
        return re.findall(phone_number_pattern, self.text)

    def parse_all(self) -> dict:
        first_name, last_name = self.get_name()
        return {
            "first_name": first_name,
            "last_name": last_name,
            "emails": self.get_emails(),
            "phone_numbers": self.get_phone_numbers(),
        }


class HuggingFaceTextParser:
    def __init__(self, text: str, model_name: str):
        self.text = text
        self.model_name = model_name
        self.qa_pipeline = pipeline(
            "question-answering", model=model_name, tokenizer=model_name
        )

    def get_name(self, confidence: float = 1e-10) -> Union[str, str]:
        input = {"question": "What is my name?", "context": self.text}
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            first_name, last_name = "None", "None"
        else:
            full_name = result["answer"].lower().split()
            first_name = full_name[0]
            last_name = "".join(full_name[1:])
        return first_name, last_name

    def get_emails(self, confidence: float = 1e-10) -> str:
        input = {"question": "What is my email?", "context": self.text}
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            emails = ""
        else:
            emails = result["answer"]
        return emails

    def get_phone_numbers(self, confidence: float = 1e-10) -> str:
        input = {"question": "What is my phone number?", "context": self.text}
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            phone_numbers = ""
        else:
            phone_numbers = result["answer"]
        return phone_numbers

    def get_university(self, confidence: float = 1e-10) -> str:
        input = {
            "question": "What university or school did I attend?",
            "context": self.text,
        }
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            univeristy = ""
        else:
            univeristy = result["answer"]
        return univeristy

    def get_university_degree_type(self, confidence: float = 1e-10) -> str:
        input = {
            "question": f"What degree type did I study at {self.get_university()}?",
            "context": self.text,
        }
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            degree_type = ""
        else:
            degree_type = result["answer"]
        return degree_type

    def get_university_degree_course(self, confidence: float = 1e-10) -> str:
        input = {
            "question": f"What subject did I study at the {self.get_university()}?",
            "context": self.text,
        }
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            degree_course = ""
        else:
            degree_course = result["answer"]
        return degree_course

    def get_job_titles(self, confidence: float = 1e-10) -> str:
        input = {"question": "What job titles have I had?", "context": self.text}
        result = self.qa_pipeline(input)
        if result["score"] < confidence:
            job_titles = ""
        else:
            job_titles = result["answer"]
        return job_titles

    def parse_all(self) -> dict:
        first_name, last_name = self.get_name()
        return {
            "first_name": first_name,
            "last_name": last_name,
            "emails": self.get_emails(),
            "phone_numbers": self.get_phone_numbers(),
            "university": self.get_university(),
            "university_degree_type": self.get_university_degree_type(),
            "university_degree_course": self.get_university_degree_course(),
            "job_titles": self.get_job_titles(),
        }
