import os
import re
import time
import random
import logging
from functools import lru_cache
from openai import OpenAI, APIError
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")
client = OpenAI(api_key=OPENAI_API_KEY)

MAX_HISTORY_LENGTH = 20
MAX_RETRIES = 3

# --- System Prompt ---
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

# --- Helper: Format multiple goals summary ---
def format_goals_summary(goals: List[Dict]) -> str:
    """Formats a summary of financial goals."""
    if not goals:
        return "No active financial goals currently."
    summaries = [f"{idx}. {goal['type']} - Target: ${goal['target_amount']} by {goal['target_date']} (Saved: ${goal.get('saved_amount', 0)})" for idx, goal in enumerate(goals, 1)]
    return " | ".join(summaries)

# --- Detect user mood with conflict awareness ---
def detect_mood(user_input: str, user_data: Dict) -> str:
    """Detects user mood based on input and conflict patterns."""
    stressed_keywords = ['stressed', 'overwhelmed', 'anxious', 'upset', 'frustrated', 'tired', 'confused']
    motivated_keywords = ['ready', 'motivated', 'excited', 'happy', 'good', 'great']
    text = user_input.lower()
    conflict_terms = ['argue', 'fight', 'disagree', 'conflict', 'stress', 'frustrate']
    conflict_detected = any(term in text for term in conflict_terms) or bool(user_data.get("conflict_patterns"))
    if any(word in text for word in stressed_keywords) or conflict_detected:
        return 'gentle, validating, and supportive tone'
    elif any(word in text for word in motivated_keywords):
        return 'motivating, clear, and growth-oriented tone'
    return 'friendly and calm tone'

# --- Detect user intent with more granularity ---
def detect_intent(user_input: str) -> str:
    """Detects user intent based on input keywords."""
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

# --- Build system prompt with tone and multi-goal summary, with caching ---
@lru_cache(maxsize=1000)
def build_system_prompt(first_name: str, partner_names: str, relationship_status: str, tone_preference: str, goals_tuple: tuple, mood: str, money_personality: str, risk_profile: str, investment_timeline: str, conflict_patterns: str) -> str:
    """Builds a personalized system prompt with caching."""
    goals = [{"type": g[0], "target_amount": g[1], "target_date": g[2], "saved_amount": g[3]} for g in goals_tuple]
    relationship_desc = "couples" if relationship_status == "partnered" else "individuals"
    goals_summary = format_goals_summary(goals)
    prompt = SYSTEM_PROMPT.format(
        relationship_desc=relationship_desc,
        first_name=first_name,
        partner_names=partner_names,
        relationship_status=relationship_status,
        mood=mood,
        tone_preference=tone_preference,
        money_personality=money_personality,
        goal_summaries=goals_summary,
        risk_profile=risk_profile,
        investment_timeline=investment_timeline,
        conflict_patterns=conflict_patterns
    )
    logger.info("Built system prompt for user: %s", first_name)
    return prompt

# --- Micro-consent handling ---
def check_micro_consent(user_input: str, user_data: Dict, intent: str) -> Tuple[bool, Optional[str]]:
    """Checks for user consent before providing advice on sensitive topics."""
    sensitive_intents = ['investment_guidance', 'debt_paydown']

    if intent not in sensitive_intents:
        return True, None

    if intent not in user_data.get("granted_consents", {}):
        user_data.setdefault("granted_consents", {})[intent] = False

    if not user_data["granted_consents"][intent]:
        if re.search(r'\b(yes|sure|please|ok|okay|go ahead|yep)\b', user_input.lower()):
            user_data["granted_consents"][intent] = True
            return True, None
        return False, f"Before I provide advice on {intent.replace('_', ' ')}, would you like me to proceed with suggestions?"

    return True, None

# --- Conflict empathy message generator ---
def get_conflict_empathy() -> str:
    """Returns a random empathy message for conflict situations."""
    return random.choice(CONFLICT_SCRIPTS)

# --- Pronoun replacement with improved context awareness ---
def adjust_pronouns(text: str, relationship_status: str) -> str:
    """Adjusts pronouns for couples or individuals, avoiding awkward phrasing and double replacements."""
    if not text or relationship_status != "partnered":
        return text
    # Avoid replacing if already 'you both', 'your shared', etc.
    def replace_you(match):
        word = match.group(0)
        # Only replace standalone 'you' not followed by 'both'
        if re.match(r'you(?! both)', word, re.IGNORECASE):
            return re.sub(r'you', 'you both', word, flags=re.IGNORECASE)
        return word
    def replace_your(match):
        word = match.group(0)
        if re.match(r'your(?! shared)', word, re.IGNORECASE):
            return re.sub(r'your', 'your shared', word, flags=re.IGNORECASE)
        return word
    # Use word boundaries and avoid double replacements
    text = re.sub(r'\byou\b(?! both)', replace_you, text, flags=re.IGNORECASE)
    text = re.sub(r'\byour\b(?! shared)', replace_your, text, flags=re.IGNORECASE)
    text = re.sub(r'\byourselves\b(?! together)', 'yourselves together', text, flags=re.IGNORECASE)
    # Remove accidental doubles
    text = re.sub(r'\byou both both\b', 'you both', text, flags=re.IGNORECASE)
    text = re.sub(r'\byour shared shared\b', 'your shared', text, flags=re.IGNORECASE)
    text = re.sub(r'\byourselves together together\b', 'yourselves together', text, flags=re.IGNORECASE)
    text = re.sub(r"you both('re|’re) both", r"you both\\1", text, flags=re.IGNORECASE)
    return text

# --- Milestone celebration (only once per month) ---
def check_milestones(user_data: Dict) -> Optional[str]:
    """Checks and celebrates financial milestones. Budget milestone only once per month."""
    total_saved = sum(goal.get("saved_amount", 0) for goal in user_data.get("financial_goals", []))
    if total_saved > 50000 and not user_data.get("milestone_celebrated", False):
        user_data["milestone_celebrated"] = True
        return f"Congratulations! You've saved over ${total_saved} towards your goals. Keep up the great work!"
    # Budget milestone: only if under budget, and only once per month
    from datetime import datetime
    current_month = datetime.now().strftime("%Y-%m")
    last_celebrated = user_data.get("budget_milestone_last_month")
    if user_data.get("budget_diff", 0) < 0 and last_celebrated != current_month:
        user_data["budget_milestone_last_month"] = current_month
        return "Great job! You've managed to spend less than your budget this month. This positive habit will help your financial health!"
    return None

# --- Legal disclaimer for specific intents ---
def get_legal_disclaimer(intent: str) -> Optional[str]:
    """Returns legal disclaimer if necessary."""
    if intent in ("investment_guidance", "debt_paydown"):
        return "Note: I am not a licensed financial advisor. Please consult a professional for personalized advice."
    return None

# --- Prune conversation history to maintain context window ---
def prune_conversation_history(messages: List[Dict], max_length: int = MAX_HISTORY_LENGTH) -> List[Dict]:
    """Prunes the conversation history to the most recent messages."""
    if len(messages) > max_length:
        return messages[-max_length:]
    return messages

# --- Main function to generate response ---
def get_bondly_response(user_input: str, user_data: Dict, messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Given the user input, user data, and conversation history messages,
    returns the assistant's response and updated messages list.
    """
    # Detect intent and mood
    intent = detect_intent(user_input)
    mood = detect_mood(user_input, user_data)

    # Micro-consent check
    consent_ok, consent_msg = check_micro_consent(user_input, user_data, intent)
    if not consent_ok:
        messages.append({"role": "assistant", "content": consent_msg})
        return consent_msg, messages

    # Update messages with user input
    messages.append({"role": "user", "content": user_input})

    # Build system prompt (cache key: tuple of immutable types)
    goals_tuple = tuple(
        (goal["type"], goal["target_amount"], goal["target_date"], goal.get("saved_amount", 0))
        for goal in user_data.get("financial_goals", [])
    )
    conflict_patterns_str = ','.join(user_data.get("conflict_patterns", [])) if user_data.get("conflict_patterns") else ""
    system_prompt = build_system_prompt(
        first_name=user_data.get("first_name", "User"),
        partner_names=user_data.get("partner_names", ""),
        relationship_status=user_data.get("relationship_status", "single"),
        tone_preference=mood,
        goals_tuple=goals_tuple,
        mood=mood,
        money_personality=user_data.get("money_personality", "balanced"),
        risk_profile=user_data.get("risk_profile", "moderate"),
        investment_timeline=user_data.get("investment_timeline", "long-term"),
        conflict_patterns=conflict_patterns_str
    )

    # Compose messages for the API
    chat_messages = [{"role": "system", "content": system_prompt}] + messages

    # Call OpenAI Chat Completion with retry
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-turbo",
                messages=chat_messages,
                max_tokens=300,
                temperature=0.75,
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.1,
                stop=None
            )
            break
        except APIError as e:
            logger.error(f"OpenAI API error on attempt {attempt}: {e}")
            if attempt == MAX_RETRIES:
                return "Sorry, I'm having trouble connecting right now. Please try again later.", messages
            time.sleep(2 ** attempt)  # exponential backoff

    response_text = response.choices[0].message.content.strip()

    # Add conflict empathy only if current input or intent is conflict/emotion related
    conflict_intents = ["emotional_encouragement"]
    conflict_keywords = ["argue", "fight", "disagree", "conflict", "stress", "frustrate", "upset", "stressed", "anxious", "overwhelmed"]
    if (
        (intent in conflict_intents or any(word in user_input.lower() for word in conflict_keywords))
        and user_data.get("conflict_patterns")
        and not any(phrase in response_text for phrase in CONFLICT_SCRIPTS)
    ):
        empathy_msg = get_conflict_empathy()
        response_text = f"{empathy_msg}\n\n{response_text}"

    # Adjust pronouns for couples if needed
    response_text = adjust_pronouns(response_text, user_data.get("relationship_status", "single"))

    # Append milestone message if applicable
    milestone_msg = check_milestones(user_data)
    if milestone_msg:
        response_text += f"\n\n{milestone_msg}"

    # Append legal disclaimer if needed
    legal_disclaimer = get_legal_disclaimer(intent)
    if legal_disclaimer:
        response_text += f"\n\n{legal_disclaimer}"

    # Append assistant response to conversation history
    messages.append({"role": "assistant", "content": response_text})

    # Prune conversation history again if needed
    messages = prune_conversation_history(messages)

    return response_text, messages
