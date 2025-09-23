import os
from dotenv import load_dotenv

load_dotenv()

class TelegramConfig:
    API_ID = os.getenv('TELEGRAM_API_ID')
    API_HASH = os.getenv('TELEGRAM_API_HASH')
    SESSION_NAME = os.getenv('TELEGRAM_SESSION', 'telegram_session.session')
    KEYWORDS = os.getenv('KEYWORDS', '').split(',')
    FORWARD_GROUP_ID = int(os.getenv('FORWARD_GROUP_ID', 0))
