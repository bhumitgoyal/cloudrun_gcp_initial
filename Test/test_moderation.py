import os
import asyncio
import logging
from bot.moderation import HinglishModerator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_mod")

async def test_moderation():
    moderator = HinglishModerator()
    
    test_cases = [
        "yaar bc class kab hai bata do",
        "tum log sab ke sab chutiye ho scammer",
        "mera link kyu nahi chal raha bsdk",
        "gand marao",
        "bhai kya mast trip thi bc maza aa gaya",
        "Good morning beta"
    ]
    
    for case in test_cases:
        logger.info(f"---")
        logger.info(f"Input: {case}")
        result = await moderator.analyze_message(case)
        logger.info(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_moderation())
