# Dependencies:
# pip install google-genai PyPDF2 python-dotenv

import os
from google import genai
from google.genai import types
import PyPDF2
from dotenv import load_dotenv

def read_pdf_text(file_path):
    """Extract text from a PDF file."""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def generate_summary(input_text):
    """Use Google GenAI to generate a summary."""
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemma-3-27b-it"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"Summarize this text in  10 point vise :\n{input_text}")
            ],
        ),
    ]

    config = types.GenerateContentConfig(response_mime_type="text/plain")

    print("Summary:\n")
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents, # type: ignore
        config=config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    load_dotenv()  # Load .env file for GEMINI_API_KEY
    pdf_text = read_pdf_text("test8.pdf")
    generate_summary(pdf_text)
