# Bondly Backend

Bondly is an emotionally intelligent AI financial coach designed to help individuals and couples build financial confidence, manage money, and achieve their goals. It provides empathetic, personalized financial guidance, tracks user context, and adapts its tone and advice to user mood and preferences.

## Features
- AI-powered financial coaching and emotional support
- Personalized advice based on user profile, goals, and relationship status
- Micro-consent for sensitive topics (e.g., investments, debt)
- Conflict empathy and milestone celebration
- Tracks conversation history and adapts tone dynamically
- No actual financial, investment, or tax advice—educational only

## Setup
1. **Clone the repository and navigate to the project folder.**
2. **Set your OpenAI API key** (either in the code or via environment variable):
   - In `bondly.py`, update the `api_key` in the `OpenAI` client initialization, or use `os.getenv("OPENAI_API_KEY")` for better security.
3. **Install dependencies:**
   ```sh
   pip install openai
   ```
4. **Run Bondly in interactive mode:**
   ```sh
   python bondly.py
   ```
   - Type your financial questions or thoughts to interact with Bondly.

## File Overview
- `bondly.py` — Core backend logic and interactive CLI for Bondly

## Customization
- Update `get_mock_user_data()` to simulate different user profiles and scenarios.
- Integrate with a web API (Flask, DRF, etc.) for production use.
- Adjust system prompt and consent logic for your use case.

## License
This project is for educational and non-commercial use. Please respect API provider terms.
