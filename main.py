import os
from file_to_text import ExtractText
from text_parser import SimpleTextParser, HuggingFaceTextParser
import pandas as pd


if __name__ == "__main__":
    os.environ["CURL_CA_BUNDLE"] = ""  # bypass VPN hugging face issue
    model_name = "deepset/tinyroberta-squad2"

    # test case 1: simple parser
    df = pd.DataFrame()
    for file in os.listdir("data/"):
        if file.endswith((".pdf", "png")):
            file_path = "data/" + file
            text = ExtractText(file_path).from_file_path()
            output = SimpleTextParser(text).parse_all()
            print(output)
            df = df.append(output, ignore_index=True)

    print(df.head())

    # test case 2: hugging face parser
    df = pd.DataFrame()
    for file in os.listdir("data/"):
        if file.endswith((".pdf", "png")):
            file_path = "data/" + file
            text = ExtractText(file_path).from_file_path()
            output = HuggingFaceTextParser(text, model_name).parse_all()
            print(output)
            df = df.append(output, ignore_index=True)

    print(df.head())
