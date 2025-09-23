import asyncio
import logging
from typing import List, Optional
from telethon import TelegramClient, events
from telethon.tl.types import User
from config import TelegramConfig
import os
from inference import predict_message_class

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
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    logger.info("Client started and listening for messages...")

    @client.on(events.NewMessage)
    async def handler(event):
        # Only process group/supergroup messages
        if not (event.is_group or event.is_channel):
            return

        message_text = event.message.message or ""
        keyword_match = any(keyword.lower() in message_text.lower() for keyword in KEYWORDS)
        model_class = predict_message_class(message_text, model_name='logistic_regression')
        if keyword_match or model_class == 'referral':
            sender = await event.get_sender()
            sender_name = getattr(sender, 'username', None) or getattr(sender, 'first_name', '')
            group = await event.get_chat()
            group_name = getattr(group, 'title', 'Unknown Group')
            group_id = event.chat_id

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
                logger.error(f"Failed to forward message: {e}")

    await client.run_until_disconnected()

def list_group_ids():
    """Lists all group and channel names with their IDs."""
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    async def run():
        await client.start()
        print("Listing all groups and channels:")
        async for dialog in client.iter_dialogs():
            if dialog.is_group or dialog.is_channel:
                print(f"{dialog.name}: {dialog.id}")
        await client.disconnect()
    asyncio.run(run())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list_groups":
        list_group_ids()
    else:
        asyncio.run(main())
