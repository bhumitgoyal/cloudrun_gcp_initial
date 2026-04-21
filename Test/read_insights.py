import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from bot.sheets_logger import SheetsAuditLogger
from dotenv import load_dotenv

load_dotenv()

def read_insights():
    logger = SheetsAuditLogger()
    logger._ensure_initialised()
    client = logger._client
    spreadsheet_id = logger.get_spreadsheet_id()
    sheet = client.open_by_key(spreadsheet_id)
    ws = sheet.worksheet("KB Insights")
    rows = ws.get_all_values()
    print("TOTAL ROWS:", len(rows))
    if len(rows) > 1:
        print("\n--- TIMESTAMP ---")
        print(rows[-1][0])
        print("\n--- QUERIES ANALYZED ---")
        print(rows[-1][1])
        print("\n--- RECOMMENDATIONS ---")
        print(rows[-1][2])
    else:
        print("No insights found.")

if __name__ == "__main__":
    read_insights()
