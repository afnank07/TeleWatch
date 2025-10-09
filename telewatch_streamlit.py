import streamlit as st


import subprocess
import time
import os
import sys
from config import TelegramConfig

# --- Telegram credentials ---
API_ID = TelegramConfig.API_ID
API_HASH = TelegramConfig.API_HASH
SESSION_NAME = TelegramConfig.SESSION_NAME
KEYWORDS = TelegramConfig.KEYWORDS
FORWARD_GROUP_ID = TelegramConfig.FORWARD_GROUP_ID


# Path to log file
LOG_FILE = 'telegram_sender.log'

# Start the bot as a subprocess
def start_bot():
    if 'bot_process' not in st.session_state or st.session_state['bot_process'] is None or st.session_state['bot_process'].poll() is not None:
        # Use the current Python interpreter (venv) to launch main.py
        st.session_state['bot_process'] = subprocess.Popen([sys.executable, 'main.py'])
        st.success('Bot started!')
    else:
        st.info('Bot is already running.')


st.title('TeleWatch Bot Messages')
if st.button('Start Bot'):
    start_bot()

st.write('### Live Log Output:')
log_placeholder = st.empty()

# Continuously update log output
def read_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Log file not found.'

while True:
    log_placeholder.text(read_log())
    time.sleep(2)