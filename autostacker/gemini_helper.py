import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_explanation(metrics: dict, language="English") -> str:
    prompt = f"""
    I trained 3 machine learning models. Their performance metrics are:
    {metrics}

    Please explain:
    1. Which model is best and why?
    2. How can we improve the results?
    3. Explain this in {language}.
    """
    response = model.generate_content(prompt)
    return response.text.strip()
