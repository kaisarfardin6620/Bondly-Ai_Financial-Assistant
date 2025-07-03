import re
import copy
import time
import random
import os
from dotenv import load_dotenv
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
)

MAX_HISTORY_LENGTH = 20
MAX_RETRIES = 3

SYSTEM_PROMPT = """
You are Bondly, an emotionally intelligent financial coach for couples. Your role is to help users set shared financial goals, manage money together, and build trust through supportive, actionable advice.

You always use inclusive, warm language like “we,” “you both,” and “together.” Your tone is non-judgmental, friendly, and clear. You avoid jargon unless asked. You encourage teamwork and progress over perfection.

You specialize in budgeting, debt planning, goal-setting, and beginner investing. You are not a licensed advisor, so you give general education and suggestions — not personalized investment advice.

Keep responses under 100 words unless asked for more detail.
"""

CONFLICT_SCRIPTS = [
    "I understand financial discussions can be challenging. Remember, it's okay to feel overwhelmed. I'm here to help you navigate this gently.",
    "Money conversations with a partner can bring up strong emotions. Let's approach this with patience and understanding.",
    "Conflicts happen, but talking about finances openly is a strong step forward. I can suggest some ways to ease the conversation if you'd like.",
    "It's normal to feel stressed about money matters. Let's take this one step at a time together."
]

def format_goals_summary(goals):
    if not goals:
        return "No active financial goals currently."
    summaries = []
    for idx, goal in enumerate(goals, 1):
        summaries.append(f"{idx}. {goal['type']} - Target: ${goal['target_amount']} by {goal['target_date']} (Saved: ${goal.get('saved_amount', 0)})")
    return " | ".join(summaries)

def get_mock_user_data():
    return {
        "first_name": "Sam",
        "partner_names": "Alex",
        "relationship_status": "partnered",
        "mood": "stressed",
        "tone_preference": "warm & encouraging",
        "money_personality": "saver",
        "financial_goals": [
            {"type": "house", "target_amount": 300000, "target_date": "50 months", "saved_amount": 50000},
            {"type": "car", "target_amount": 30000, "target_date": "12 months", "saved_amount": 10000}
        ],
        "monthly_income": 6000,
        "budget_diff": -200,
        "top_spending_categories": ["Dining", "Subscriptions", "Shopping"],
        "total_debt": 5000,
        "monthly_savings_rate": 1000,
        "awaiting_consent": {},
        "risk_profile": "moderate",
        "investment_timeline": "long-term",
        "allocation_breakdown": "60% stocks, 30% bonds, 10% cash",
        "conflict_patterns": "occasional disagreements about spending",
        "milestones_reached": []  # NEW: track milestones
    }

def detect_mood(user_input, user_data):
    stressed_keywords = ['stressed', 'overwhelmed', 'anxious', 'upset', 'frustrated', 'tired', 'confused']
    motivated_keywords = ['ready', 'motivated', 'excited', 'happy', 'good', 'great']
    text = user_input.lower()

    conflict_terms = ['argue', 'fight', 'disagree', 'conflict', 'stress', 'frustrate']
    conflict_detected = any(term in text for term in conflict_terms) or bool(user_data.get("conflict_patterns"))

    if any(word in text for word in stressed_keywords) or conflict_detected:
        return 'gentle, validating, and supportive tone'
    elif any(word in text for word in motivated_keywords):
        return 'motivating, clear, and growth-oriented tone'
    else:
        return 'friendly and calm tone'

def detect_intent(user_input):
    text = user_input.lower()
    if re.search(r'\b(save|goal|plan|budget|track)\b', text):
        return "goal_setting"
    if re.search(r'\b(spend|expenses|budget|review)\b', text):
        return "budget_review"
    if re.search(r'\b(debt|credit card|pay off|loans)\b', text):
        return "debt_paydown"
    if re.search(r'\b(feel|upset|stressed|worried|anxious|frustrated)\b', text):
        return "emotional_encouragement"
    if re.search(r'\b(invest|portfolio|stocks|bonds|crypto|real estate|asset)\b', text):
        return "investment_guidance"
    return "general"

def build_system_prompt(user_data):
    relationship_desc = "couples" if user_data.get("relationship_status") == "partnered" else "individuals"
    tone_preference = user_data.get("tone_preference", "warm & encouraging")
    goals_summary = format_goals_summary(user_data.get("financial_goals", []))
    return SYSTEM_PROMPT.format(
        relationship_desc=relationship_desc,
        first_name=user_data.get("first_name", ""),
        partner_names=user_data.get("partner_names", ""),
        relationship_status=user_data.get("relationship_status", ""),
        mood=user_data.get("mood", ""),
        tone_preference=tone_preference,
        money_personality=user_data.get("money_personality", ""),
        goal_summaries=goals_summary,
        risk_profile=user_data.get("risk_profile", "moderate"),
        investment_timeline=user_data.get("investment_timeline", "medium-term"),
        conflict_patterns=user_data.get("conflict_patterns", "none")
    )

def check_micro_consent(user_input, user_data, intent):
    if intent in ['investment_guidance', 'debt_paydown'] and not user_data["awaiting_consent"].get(intent, False):
        user_data["awaiting_consent"][intent] = True
        return False, f"Before I provide advice on {intent.replace('_', ' ')}, would you like me to proceed with suggestions?"

    if user_data["awaiting_consent"].get(intent, False):
        if re.search(r'\b(no|not now|later|don\'t)\b', user_input.lower()):
            user_data["awaiting_consent"][intent] = False
            return False, "No problem, I won’t provide suggestions on that topic right now. Let me know if you change your mind."
        if re.search(r'\b(yes|sure|please|ok|okay|go ahead|yep)\b', user_input.lower()):
            user_data["awaiting_consent"][intent] = False
            return True, None
        return False, "Please let me know if you'd like me to proceed with suggestions on this topic."

    return True, None

def get_conflict_empathy():
    return random.choice(CONFLICT_SCRIPTS)

def adjust_pronouns(text, relationship_status):
    if relationship_status == "partnered":
        text = re.sub(r'\byou\b', 'you both', text, flags=re.IGNORECASE)
        text = re.sub(r'\byour\b', 'your shared', text, flags=re.IGNORECASE)
        text = re.sub(r'\byourselves\b', 'yourselves together', text, flags=re.IGNORECASE)
    return text

def check_milestones(user_data):
    total_saved = sum(goal.get("saved_amount", 0) for goal in user_data.get("financial_goals", []))
    milestones = []
    if total_saved > 50000 and "50000_saved" not in user_data["milestones_reached"]:
        milestones.append(f"Congratulations! You've saved over ${total_saved} towards your goals. Keep up the great work!")
        user_data["milestones_reached"].append("50000_saved")
    if user_data.get("budget_diff", 0) > 0 and "under_budget" not in user_data["milestones_reached"]:
        milestones.append("Nice job on staying under budget this month!")
        user_data["milestones_reached"].append("under_budget")
    return "\n\n".join(milestones) if milestones else None

def get_legal_disclaimer(intent):
    if intent in ['investment_guidance', 'debt_paydown']:
        return "Disclaimer: I provide general educational information only. Please consult licensed financial or tax professionals for personalized advice."
    return None

def prune_conversation_history(history):
    if len(history) > MAX_HISTORY_LENGTH:
        return history[-MAX_HISTORY_LENGTH:]
    return history

def call_openai_chat(messages):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response_text = ""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                chunk_message = getattr(delta, "content", "") or ""
                response_text += chunk_message
            return response_text
        except Exception as e:
            retries += 1
            print(f"\nError during OpenAI API call: {e}. Retrying ({retries}/{MAX_RETRIES})...")
            time.sleep(2)
    return "Sorry, I'm having trouble processing your request right now. Please try again later."

def get_bondly_response(user_input, user_data, conversation_history):
    messages = copy.deepcopy(conversation_history)

    intent = detect_intent(user_input)
    mood = detect_mood(user_input, user_data)

    if not messages or messages[0]["role"] != "system":
        user_data["mood"] = mood
        system_prompt = build_system_prompt(user_data)
        messages.insert(0, {"role": "system", "content": system_prompt})

    consent_ok, consent_response = check_micro_consent(user_input, user_data, intent)
    if not consent_ok:
        return consent_response, messages

    messages.append({"role": "user", "content": user_input})
    messages = prune_conversation_history(messages)

    conflict_terms = ['argue', 'fight', 'disagree', 'conflict', 'stress', 'frustrate']
    if any(term in user_input.lower() for term in conflict_terms):
        empathy_text = get_conflict_empathy()
        messages.append({"role": "assistant", "content": empathy_text})

    response_text = call_openai_chat(messages)

    response_text = adjust_pronouns(response_text, user_data.get("relationship_status", "single"))

    milestone_msg = check_milestones(user_data)
    if milestone_msg:
        response_text += "\n\n" + milestone_msg

    legal_note = get_legal_disclaimer(intent)
    if legal_note:
        response_text += f"\n\n{legal_note}"

    messages.append({"role": "assistant", "content": response_text})
    messages = prune_conversation_history(messages)

    return response_text, messages

if __name__ == "__main__":
    user_data = get_mock_user_data()
    conversation_history = []

    print("Bondly is ready. Ask your financial question or share your thoughts!")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! Remember, managing money is a journey — keep moving forward.")
            break
        response, conversation_history = get_bondly_response(user_input, user_data, conversation_history)
        print(f"\nBondly: {response}")
