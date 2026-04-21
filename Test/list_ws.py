import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from bot.sheets_logger import SheetsAuditLogger
from dotenv import load_dotenv

load_dotenv()

def list_ws():
    logger = SheetsAuditLogger()
    logger._ensure_initialised()
    client = logger._client
    spreadsheet_id = logger.get_spreadsheet_id()
    sheet = client.open_by_key(spreadsheet_id)
    worksheets = sheet.worksheets()
    for ws in worksheets:
        print(f"Title: '{ws.title}', ID: {ws.id}")

if __name__ == "__main__":
    list_ws()
