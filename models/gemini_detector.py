import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# Load variables from .env
load_dotenv()


class FakeNewsOutput(BaseModel):
    fake: bool = Field(
        description="True if the news appears to be fake, otherwise False."
    )

    reasoning: str = Field(
        description="A short explanation explaining why the news is fake or real."
    )


class GeminiDetector:

    def __init__(self):

        # Read API key from .env
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Please add it to your .env file."
            )

        # Create Gemini client
        self.client = genai.Client(
            api_key=api_key
        )

    def detect(self, news_text):

        prompt = f"""
You are an expert fake news detection system.

Analyze the news text given below.

Determine whether the news appears to be:

1. Fake News
or
2. Real News

Consider:
- misleading claims
- fabricated information
- suspicious wording
- factual consistency
- exaggerated statements
- credibility of the information

News text:

{news_text}

Return a boolean value for fake and a short reasoning.
"""

        try:

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,

                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FakeNewsOutput,
                    temperature=0.2
                )
            )

            # Convert Gemini JSON response into Pydantic object
            result = FakeNewsOutput.model_validate_json(
                response.text
            )

            return result.fake, result.reasoning

        except Exception as e:

            print("Gemini API Error:", e)

            raise