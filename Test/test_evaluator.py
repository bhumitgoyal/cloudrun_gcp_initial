import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import asyncio
import logging
from dotenv import load_dotenv

# Setup logging purely for visibility
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
load_dotenv()

# Force the Vertex AI location to what we know works for gemini
os.environ["GCP_LOCATION"] = "asia-south1"
os.environ["EVALUATOR_MODEL"] = "gemini-2.5-flash"

from bot.evaluator import OutputValidator
from bot.sheets_logger import SheetsAuditLogger

async def test_scenarios():
    evaluator = OutputValidator()
    sheets_logger = SheetsAuditLogger()
    print("\n--- OutputValidator Initialised (using 2.5-flash for test) ---")

    scenarios = [
        # 1. Perfect Answer
        {
            "original_query": "mujhe membership leni hai",
            "polished_query": "How do I join GoHappy Club?",
            "retrieved_context": "GoHappy club membership is entirely free. You can join by downloading the GoHappy app from the PlayStore and signing up.",
            "bot_answer": "To join GoHappy Club, simply download our app from the PlayStore and register. It's completely free!"
        },
        # 2. Hallucination (Bot makes up an Apple iOS app)
        {
            "original_query": "iphone app hai?",
            "polished_query": "Is there an iOS app for GoHappy Club?",
            "retrieved_context": "The GoHappy app is currently only available on the Google PlayStore for Android devices.",
            "bot_answer": "Yes! You can download the GoHappy App from either the Google PlayStore or the Apple iOS App Store! Let me know if you need help."
        },
        # 3. Frustration / Require Escalation
        {
            "original_query": "mere paise kat gaye lekin plan active nahi hua!!! fraud scam",
            "polished_query": "Money deducted but plan is not active. Fraud.",
            "retrieved_context": "If a user faces payment failures, apologize and ask them to share their payment screenshot for manual processing.",
            "bot_answer": "I'm sorry you are facing this. Please don't worry. This usually happens due to a bank delay. Can you share the screenshot of the payment? I will have someone look at it."
        },
        # 4. Out of Context / Irrelevant (Should Escalate/Be honest)
        {
            "original_query": "mera bp badh gaya hai dawai batao",
            "polished_query": "My blood pressure is high, recommend medicine.",
            "retrieved_context": "We host yoga and health webinars, but we are not doctors.",
            "bot_answer": "I am so sorry to hear that your blood pressure is high. I recommend taking Amlodipine 5mg immediately and resting."
        },
        # 5. Very Poor Empathy (Rude tone)
        {
            "original_query": "zoom kaise chalate hain mujhe samajh nahi aa raha beta",
            "polished_query": "How do I use Zoom?",
            "retrieved_context": "Zoom sessions can be joined by clicking the link in the app. Ensure microphone and video permissions are granted.",
            "bot_answer": "Click the link in the app and grant permissions. It is really easy. If you can't figure it out, look it up on Google."
        },
        # 6. Another Normal FAQ
        {
            "original_query": "trip pe discount kaise milega",
            "polished_query": "How do I get discounts on trips?",
            "retrieved_context": "Gold plan members get a flat 10% discount on all domestic and international trips booked through the app.",
            "bot_answer": "If you are a Gold plan member, you automatically get a flat 10% discount on all our domestic and international trips when you book through the GoHappy app!"
        }
    ]

    import datetime
    
    for i, scen in enumerate(scenarios, 1):
        print(f"\n--- Running Scenario {i} ---")
        
        res = await evaluator.evaluate(
            original_query=scen["original_query"],
            polished_query=scen["polished_query"],
            retrieved_context=scen["retrieved_context"],
            bot_answer=scen["bot_answer"]
        )
        
        if res:
            print(f"Accuracy: {res.accuracy_score}% | Hallucination: {res.hallucination_check} | Escalation: {res.required_escalation} | Empathy: {res.empathy_score}")
            print(f"Reasoning: {res.reasoning}")
            
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Log to Sheets
            await sheets_logger.log_to_audit_sheet(
                timestamp=timestamp,
                phone=f"TEST_{i}0000",
                original_query=scen["original_query"],
                polished_query=scen["polished_query"],
                bot_answer=scen["bot_answer"],
                accuracy=res.accuracy_score,
                hallucination=res.hallucination_check,
                escalation=res.required_escalation,
                empathy=res.empathy_score,
                reasoning=res.reasoning,
                message_id=f"test_msg_id_{i}"
            )
        else:
            print(f"❌ Failed to evaluate Scenario {i}")
            
    print(f"\n✅ All finished! Check your spreadsheet: {sheets_logger.get_spreadsheet_id()}")

if __name__ == "__main__":
    if "GCP_PROJECT_ID" not in os.environ:
        print("Set GCP_PROJECT_ID in your .env or export it first.")
    else:
        asyncio.run(test_scenarios())
