import asyncio
import logging
from typing import List, Optional
from telethon import TelegramClient, events
from telethon.tl.types import User
from config import TelegramConfig
import os
from inference import predict_message_class

# --- Google Sheets Integration ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- Google Sheets Config ---
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID', 'YOUR_SPREADSHEET_ID')  # Set this in your environment or config
GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH', 'google-credentials.json')  # Path to your service account json
SHEET_ROW_LIMIT = int(os.getenv('SHEET_ROW_LIMIT', 900000))  # Max rows per sheet, ensure int


# Helper class for Google Sheets logging

class GoogleSheetLogger:
    def __init__(self, sheet_id, creds_path, row_limit=900000):
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
        ]
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            self.client = gspread.authorize(creds)
            self.sheet = self.client.open_by_key(sheet_id)
            # Ensure row_limit is always int
            self.row_limit = int(row_limit)
            logging.info(f"GoogleSheetLogger: row_limit type is {type(self.row_limit)}")
            self.worksheet = self._get_or_create_worksheet()
            logging.info(f"GoogleSheetLogger: Connected to sheet {sheet_id} with row limit {self.row_limit}")
        except Exception as e:
            logging.error(f"GoogleSheetLogger: Failed to initialize: {e}", exc_info=True)
            raise

    def _get_or_create_worksheet(self):
        try:
            worksheets = self.sheet.worksheets()
            for ws in worksheets[::-1]:  # Check last sheet first
                ws_row_count = int(ws.row_count)
                if ws_row_count < self.row_limit or ws_row_count - len(ws.get_all_values()) > 1:
                    return ws
            # Create new worksheet
            new_index = len(worksheets) + 1
            ws = self.sheet.add_worksheet(title=f"Messages_{new_index}", rows=str(self.row_limit), cols="10")
            ws.append_row(["Timestamp", "Sender", "Sender ID", "Group", "Group ID", "Message", "Keyword Match", "Model Prediction"])
            logging.info(f"GoogleSheetLogger: Created new worksheet Messages_{new_index}")
            return ws
        except Exception as e:
            logging.error(f"GoogleSheetLogger: Failed to get or create worksheet: {e}", exc_info=True)
            raise

    def log_message(self, timestamp, sender, sender_id, group, group_id, message, keyword_match, model_prediction):
        try:
            ws = self.worksheet
            # Check if we need to switch to a new worksheet
            if len(ws.get_all_values()) >= self.row_limit:
                ws = self._get_or_create_worksheet()
                self.worksheet = ws
            # If the worksheet is empty, add headers first
            if len(ws.get_all_values()) == 0:
                ws.append_row(["Timestamp", "Sender", "Sender ID", "Group", "Group ID", "Message", "Keyword Match", "Model Prediction"])
                logging.info(f"GoogleSheetLogger: Added headers to worksheet {ws.title}")
            ws.append_row([
                timestamp, sender, sender_id, group, group_id, message, str(keyword_match), model_prediction
            ])
            logging.info(f"GoogleSheetLogger: Logged message for sender {sender} in group {group}")
        except Exception as e:
            logging.error(f"GoogleSheetLogger: Failed to log message: {e}", exc_info=True)

# Initialize logger (global, so it persists across handler calls)
sheet_logger = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_sender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Telegram credentials ---
API_ID = TelegramConfig.API_ID
API_HASH = TelegramConfig.API_HASH
SESSION_NAME = TelegramConfig.SESSION_NAME

# --- Monitoring settings ---
KEYWORDS = TelegramConfig.KEYWORDS
FORWARD_GROUP_ID = TelegramConfig.FORWARD_GROUP_ID

async def main():
    try:
        client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        await client.start()
        logger.info("Client started and listening for messages...")

        global sheet_logger
        if sheet_logger is None:
            try:
                sheet_logger = GoogleSheetLogger(GOOGLE_SHEET_ID, GOOGLE_CREDENTIALS_PATH, SHEET_ROW_LIMIT)
                logger.info("GoogleSheetLogger initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize GoogleSheetLogger: {e}", exc_info=True)
                sheet_logger = None

        @client.on(events.NewMessage)
        async def handler(event):
            try:
                # Only process group/supergroup messages
                if not (event.is_group or event.is_channel):
                    return

                message_text = event.message.message or ""
                keyword_match = any(keyword.lower() in message_text.lower() for keyword in KEYWORDS)
                model_class = predict_message_class(message_text, model_name='logistic_regression')
                sender = await event.get_sender()
                sender_name = getattr(sender, 'username', None) or getattr(sender, 'first_name', '')
                group = await event.get_chat()
                group_name = getattr(group, 'title', 'Unknown Group')
                group_id = event.chat_id

                # Log every message to Google Sheet
                if sheet_logger:
                    try:
                        sheet_logger.log_message(
                            datetime.utcnow().isoformat(),
                            sender_name,
                            sender.id,
                            group_name,
                            group_id,
                            message_text,
                            keyword_match,
                            model_class
                        )
                    except Exception as e:
                        logger.error(f"Failed to log message to Google Sheet: {e}", exc_info=True)

                if keyword_match or model_class == 'referral':
                    details = f"Message flagged!\n"
                    details += f"From: {sender_name} (ID: {sender.id})\n"
                    details += f"Group: {group_name} (ID: {group_id})\n"
                    details += f"Message: {message_text}\n"
                    details += f"Keyword match: {keyword_match}\n"
                    details += f"Model prediction: {model_class}"

                    # Forward the original message
                    try:
                        await client.send_message(FORWARD_GROUP_ID, details)
                        await event.forward_to(FORWARD_GROUP_ID)
                        logger.info(f"Forwarded message from {sender_name} in {group_name}")
                    except Exception as e:
                        logger.error(f"Failed to forward message: {e}", exc_info=True)
            except Exception as handler_error:
                logger.error(f"Error in message handler: {handler_error}", exc_info=True)

        await client.run_until_disconnected()
    except Exception as main_error:
        logger.error(f"Fatal error in main: {main_error}", exc_info=True)

def list_group_ids():
    """Lists all group and channel names with their IDs."""
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    async def run():
        await client.start()
        logger.info("Listing all groups and channels:")
        async for dialog in client.iter_dialogs():
            if dialog.is_group or dialog.is_channel:
                logger.info(f"{dialog.name}: {dialog.id}")
        await client.disconnect()
    asyncio.run(run())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list_groups":
        list_group_ids()
    else:
        asyncio.run(main())
