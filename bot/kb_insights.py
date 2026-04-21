import os
import logging
import datetime
from typing import Optional
from gspread import Spreadsheet, Worksheet

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from bot.llm import SYSTEM_PROMPT
from bot.sheets_logger import SheetsAuditLogger

logger = logging.getLogger("gohappy.kb_insights")

class KBInsightsGenerator:
    """
    Generates Knowledge Base Insights using an LLM.
    Triggered via WhatsApp by an admin.
    """
    
    def __init__(self, sheets_logger: SheetsAuditLogger):
        self.sheets_logger = sheets_logger
        project = os.environ.get("GCP_PROJECT_ID", "")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        if project:
            vertexai.init(project=project, location=location)

        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-002")
        self.model = GenerativeModel(
            model_name=model_name,
            system_instruction="You are an expert conversational AI data analyst and knowledge base architect for GoHappy Club."
        )
        self.generation_config = GenerationConfig(
            temperature=0.4,
            max_output_tokens=2048,
        )

    async def generate_insights(self) -> str:
        """
        1. Access the main audit sheet.
        2. Identify unprocessed rows.
        3. Pass them to Gemini along with the System Prompt and README context.
        4. Create/append to a "KB Insights" worksheet.
        5. Mark rows as Processed.
        Returns a string status for the admin.
        """
        try:
            self.sheets_logger._ensure_initialised()
            client = self.sheets_logger._client
            spreadsheet_id = self.sheets_logger.get_spreadsheet_id()
            
            if not spreadsheet_id:
                return "❌ No Audit Spreadsheet found. Has the bot logged any messages yet?"
            
            spreadsheet: Spreadsheet = client.open_by_key(spreadsheet_id)
            main_sheet: Worksheet = spreadsheet.sheet1
            
            # 1. Ensure "Processed?" column exists
            header_row = main_sheet.row_values(1)
            processed_col_index = None
            for idx, col in enumerate(header_row):
                if col.strip().lower() in ["processed?", "processed"]:
                    processed_col_index = idx + 1
                    break
            
            if not processed_col_index:
                processed_col_index = len(header_row) + 1
                main_sheet.update_cell(1, processed_col_index, "Processed?")
                
            # 2. Fetch all unprocessed rows
            all_records = main_sheet.get_all_records(expected_headers=header_row)
            # get_all_records maps headers to column values.
            # But get_all_records fails if headers are empty or non-unique.
            # Usually safer to use get_all_values()
            all_values = main_sheet.get_all_values()
            if len(all_values) <= 1:
                return "ℹ️ No queries found in the spreadsheet to analyze."

            headers = all_values[0]
            unprocessed_rows = []
            unprocessed_row_indices = [] # 1-based index including header
            
            for i in range(1, len(all_values)):
                row = all_values[i]
                # Pad row to match header length + Processed column
                while len(row) < processed_col_index:
                    row.append("")
                
                processed_val = row[processed_col_index - 1].strip()
                if not processed_val:
                    unprocessed_rows.append(row)
                    unprocessed_row_indices.append(i + 1) # +1 for 1-based, +1 for header
                    
            if not unprocessed_rows:
                return "✅ No new un-processed queries to analyze! You're fully caught up."

            # 3. Format the data for the LLM
            formatted_queries = ""
            for idx, row in enumerate(unprocessed_rows):
                phone = row[1] if len(row) > 1 else "Unknown"
                query = row[2] if len(row) > 2 else ""
                answer = row[4] if len(row) > 4 else ""
                accuracy = row[5] if len(row) > 5 else ""
                reasoning = row[9] if len(row) > 9 else ""
                formatted_queries += f"Row {idx+1}:\nQuery: {query}\nBot Answer: {answer}\nAccuracy: {accuracy}\nEvaluator Reasoning: {reasoning}\n---\n"
                
            # Attempt to read README.md
            readme_context = ""
            try:
                if os.path.exists("README.md"):
                    with open("README.md", "r", encoding="utf-8") as f:
                        readme_context = f.read()
            except Exception as e:
                logger.warning("Could not read README.md: %s", e)
                
            llm_prompt = f"""
I want you to analyze the following recent customer queries that our GoHappy Club chatbot handled. 
Some of them may have been inaccurate, required escalation, or had hallucinations. Your goal is to identify common patterns, missing facts, and missing instructions.

Based on the conversations, answer this question:
"What EXACT paragraphs, facts, or policies should we add to our knowledge base to prevent the bot from failing on these types of queries next time?"

━━━━━━━━━━━━━━━━━
CONTEXT - OUR SYSTEM PROMPT:
{SYSTEM_PROMPT}

━━━━━━━━━━━━━━━━━
CONTEXT - OUR README:
{readme_context[:4000]} # Trim to fit limit if needed

━━━━━━━━━━━━━━━━━
RECENT CHATBOT QUERIES:
{formatted_queries}

━━━━━━━━━━━━━━━━━
OUTPUT FORMAT:
Provide your insights as a professional, clearly formatted report.
1. Summary of Gaps (What is missing)
2. Suggested Additions to Knowledge Base (Write the exact text we should paste into our corpus to fix this)
"""
            # Call Gemini
            response = await self.model.generate_content_async(
                llm_prompt,
                generation_config=self.generation_config,
            )
            raw_insights = response.text.strip()
            
            # 4. Save Insights to a new Worksheet
            insights_sheet_name = "KB Insights"
            try:
                insights_sheet = spreadsheet.worksheet(insights_sheet_name)
            except Exception:
                insights_sheet = spreadsheet.add_worksheet(title=insights_sheet_name, rows="100", cols="5")
                insights_sheet.append_row(["Timestamp", "New Queries Analyzed", "Insights & Recommendations"])
                
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            insights_sheet.append_row([timestamp, len(unprocessed_rows), raw_insights], value_input_option="USER_ENTERED")
            
            # 5. Mark rows as Processed
            # We can batch update to save API limits
            cells_to_update = []
            for row_idx in unprocessed_row_indices:
                cells_to_update.append({
                    'range': f"{self._col_num_to_letter(processed_col_index)}{row_idx}",
                    'values': [["viewed"]]
                })
            
            main_sheet.batch_update(cells_to_update)
            
            logger.info("Successfully analyzed %d queries and added insights.", len(unprocessed_rows))
            return f"✅ Successfully analyzed {len(unprocessed_rows)} new queries!\nInsights have been saved to the 'KB Insights' tab in your Google Sheet."

        except Exception as exc:
            logger.error("Failed to generate KB insights: %s", exc, exc_info=True)
            return f"❌ Failed to generate insights. Error: {str(exc)}"

    @staticmethod
    def _col_num_to_letter(col_num: int) -> str:
        string = ""
        while col_num > 0:
            col_num, remainder = divmod(col_num - 1, 26)
            string = chr(65 + remainder) + string
        return string
