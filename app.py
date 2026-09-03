from flask import Flask, render_template, request

from models.ml_detector import MLDetector
from models.gemini_detector import GeminiDetector

app = Flask(__name__)

ml_detector = MLDetector()

try:
    gemini_detector = GeminiDetector()
except Exception as e:
    print("Gemini initialization failed:", e)
    gemini_detector = None


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/authenticity")
def authenticity():
    return render_template("authenticity.html")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        news_text = request.form.get("news_text", "").strip()

        if not news_text:
            return render_template(
                "index.html",
                error="Please enter some news text."
            )

        # ML prediction
        ml_fake, ml_reasoning = ml_detector.detect(news_text)

        final_result = "Fake News" if ml_fake else "Real News"

        gemini_reasoning = "Gemini verification currently unavailable."
        gemini_result = "Unavailable"

        # Try Gemini only if available
        if gemini_detector:
            try:
                gemini_fake, gemini_reasoning = gemini_detector.detect(news_text)

                gemini_result = (
                    "Fake News" if gemini_fake else "Real News"
                )

                # Gemini can override ML when API works
                final_result = gemini_result

            except Exception as e:
                print("Gemini API unavailable:", e)

        return render_template(
            "result.html",
            news_text=news_text,
            final_result=final_result,
            ml_result=("Fake News" if ml_fake else "Real News"),
            ml_reasoning=ml_reasoning,
            gemini_result=gemini_result,
            gemini_reasoning=gemini_reasoning
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)