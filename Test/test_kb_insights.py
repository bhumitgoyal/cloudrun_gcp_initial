import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import asyncio
import logging
from bot.sheets_logger import SheetsAuditLogger
from bot.kb_insights import KBInsightsGenerator
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

async def run():
    sheets_logger = SheetsAuditLogger()
    kb_insights = KBInsightsGenerator(sheets_logger)
    result = await kb_insights.generate_insights()
    print("\nFINAL RESULT:\n", result)

if __name__ == "__main__":
    asyncio.run(run())
