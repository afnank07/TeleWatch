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

# Async Telegram client in a thread

class TelegramStreamHandler:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.started = False

    def start(self):
        if not self.started:
            self.thread.start()
            self.started = True

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())

    async def main(self):
        # Create TelegramClient inside the thread after setting the event loop
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        await self.client.start()
        @self.client.on(events.NewMessage)
        async def handler(event):
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
                details = f"**RECEIVED:**\nFrom: {sender_name} (ID: {sender.id})\nGroup: {group_name} (ID: {group_id})\nMessage: {message_text}\nKeyword match: {keyword_match}\nModel prediction: {model_class}"
                st.session_state['messages'].append(details)
                try:
                    await self.client.send_message(FORWARD_GROUP_ID, details)
                    await event.forward_to(FORWARD_GROUP_ID)
                    st.session_state['messages'].append(f"**SENT:** Forwarded message from {sender_name} in {group_name}")
                except Exception as e:
                    st.session_state['messages'].append(f"**ERROR:** Failed to forward message: {e}")
        await self.client.run_until_disconnected()

# Start Telegram handler
if 'tg_handler' not in st.session_state:
    st.session_state['tg_handler'] = TelegramStreamHandler()
    st.session_state['tg_handler'].start()
