
# TeleWatch

TeleWatch is a lightweight monitoring and alerting tool for Telegram groups and channels. It uses Telethon to listen for messages, classifies them using a local Sentence-BERT model and scikit-learn classifiers, and provides a Streamlit dashboard for live monitoring and log viewing.

---

## Features
- Monitors Telegram groups/channels for specific keywords or referral messages
- Uses a local Sentence-BERT model for message classification
- Forwards flagged messages to a specified group/channel
- Logs all activity to `telegram_sender.log`
- Streamlit dashboard for live log viewing and bot control

---

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd TeleWatch
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Edit the `.env` file with your Telegram API credentials and settings:
```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION=telegram_session
KEYWORDS=keyword1,keyword2
FORWARD_GROUP_ID=-100xxxxxxxxxx
SBERT_MODEL_PATH=./models/sentence-bert/
```

**Note:** Ensure the `models/sentence-bert/` directory contains all required Sentence-BERT files. You can use a HuggingFace model by setting `SBERT_MODEL_PATH=all-MiniLM-L6-v2`.

### 4. Prepare models
Place your trained classifier models (`logistic_regression.joblib`, etc.) in the `models/` directory.

---

## Usage


### Run the Telegram Monitor (main.py)
```bash
python main.py
```
This will start listening for messages in your configured Telegram groups/channels and log activity to `telegram_sender.log`.

**Options:**
- To list all accessible group/channel IDs (for setting `FORWARD_GROUP_ID`):
  ```bash
  python main.py list_groups
  ```

---

### Train and Evaluate Classifiers (classification.py)
The `classification.py` script allows you to train, cross-validate, and save classifiers for message classification.

**Steps:**
1. Place your labeled dataset in `inputs/dataset.xlsx` with columns `Message` and `class` (`referral` or `not referral`).
2. Run:
	```bash
	python classification.py
	```
3. The script will:
	- Encode messages using Sentence-BERT
	- Train Logistic Regression, SVM (Linear), and Random Forest classifiers with 5-fold cross-validation
	- Print precision, recall, F1, and accuracy for each classifier
	- Save the trained models to `models/` as `.joblib` files

**Example output:**
```
Classifier: Logistic Regression
Precision: 0.92
Recall: 0.90
F1 Score: 0.91
Accuracy: 0.91
...
Saved logistic_regression model to models/logistic_regression.joblib
```

**Inference Example:**
You can use the provided function in `classification.py` to test a message:
```python
from classification import load_model_and_predict
load_model_and_predict("Your message text here", model_name='logistic_regression')
```

---

### Run the Streamlit Dashboard
```bash
streamlit run telewatch_streamlit.py --server.port 8501 --server.address 0.0.0.0
```
Open your browser to `http://localhost:8501` to view the dashboard, start the bot, and see live logs.

---

## Deployment

### On a VPS (Ubuntu example)
1. Set up your environment and install dependencies as above.
2. (Recommended) Use a process manager like `tmux`, `screen`, or `systemd` to keep the app running.
3. Secure your server (see Security section below).
4. (Optional) Use Nginx as a reverse proxy and enable HTTPS.

### On Render or Cloud
1. Push your code to a GitHub repo.
2. Set the start command to:
	 ```
	 streamlit run telewatch_streamlit.py --server.port $PORT --server.address 0.0.0.0
	 ```
3. Add your secrets as environment variables in the Render dashboard.

---

## Security Best Practices
- Never commit secrets or API keys to your repo.
- Use environment variables or a `.env` file with restricted permissions (`chmod 600 .env`).
- Enable a firewall and only open necessary ports (e.g., 8501 for Streamlit, 22 for SSH).
- Use SSH keys, disable root login, and keep your system updated.
- Protect your Streamlit app with a password or behind a reverse proxy.
- Always use HTTPS for public access.

---

## Troubleshooting
- **SBERT model not found:**
	- Ensure `SBERT_MODEL_PATH` points to the correct directory or use a HuggingFace model name.
	- Check file permissions and paths (absolute vs. relative).
- **Invalid Peer error:**
	- Make sure `FORWARD_GROUP_ID` is correct and your bot/account is a member of the group/channel.
	- Use the `list_group_ids()` utility in `main.py` to print available group/channel IDs.
- **502 Bad Gateway on Render:**
	- Check logs for Python errors or port issues.
	- Make sure you are using the correct start command and all dependencies are installed.

---

## License
MIT License
