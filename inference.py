# Batch evaluation utility
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Inference utility for message classification
import os
from sentence_transformers import SentenceTransformer
import joblib
from load_dotenv import load_dotenv
load_dotenv()


# --- Model cache and config ---
MODEL_PATH = os.environ.get('SBERT_MODEL_PATH')  # Set to local path or model name
print(f"Using SBERT model path: {MODEL_PATH}")
# MODEL_PATH = "./models/sentence-bert/"  # Set to local path or model name
_sbert_model = None
_clf_cache = {}

def get_sbert_model():
	global _sbert_model
	if _sbert_model is None:
		_sbert_model = SentenceTransformer(MODEL_PATH)
	return _sbert_model

def get_classifier(model_name):
	global _clf_cache
	if model_name not in _clf_cache:
		model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.joblib')
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"Model file not found: {model_path}")
		_clf_cache[model_name] = joblib.load(model_path)
	return _clf_cache[model_name]

def predict_message_class(message, model_name='logistic_regression'):
	"""
	Loads a trained model and SBERT, encodes the message, and returns the predicted class label.
	model_name: 'logistic_regression', 'svm_(linear)', or 'random_forest'
	Returns: 'referral' or 'not referral'
	"""
	sbert = get_sbert_model()
	clf = get_classifier(model_name)
	emb = sbert.encode([message])
	pred = clf.predict(emb)[0]
	return 'referral' if pred == 1 else 'not referral'

def evaluate_on_excel(test_file='test.xlsx', model_name='logistic_regression', message_col='Message', label_col='class'):
	"""
	Runs inference on all messages in test_file and prints precision, recall, accuracy, and F1 score.
	Assumes label_col contains 'referral' or 'not referral'.
	"""
	df = pd.read_excel(test_file)
	messages = df[message_col].astype(str).tolist()
	true_labels = df[label_col].map(lambda x: 1 if x == 'referral' else 0).values
	pred_labels = []
	pred_classes = []
	for msg in tqdm(messages, desc="Inference", leave=True):
		pred = predict_message_class(msg, model_name=model_name)
		pred_labels.append(1 if pred == 'referral' else 0)
		pred_classes.append(pred)
	# Add predicted class to DataFrame
	df['predicted_class'] = pred_classes
	# Save to new Excel file
	out_file = f"predictions_{model_name}.xlsx"
	# df.to_excel(out_file, index=False)
	print(f"Saved predictions to {out_file}")
	print(f"Results for {model_name} on {test_file}:")
	print(f"Precision: {precision_score(true_labels, pred_labels):.3f}")
	print(f"Recall: {recall_score(true_labels, pred_labels):.3f}")
	print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.3f}")
	print(f"F1 Score: {f1_score(true_labels, pred_labels):.3f}")
	print(classification_report(true_labels, pred_labels, target_names=['not referral', 'referral']))

# Run batch evaluation if executed directly
# Run batch evaluation if executed directly

# ---
# To host the model on your own VM, download these files from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2:
#   config.json
#   sentence_bert_config.json
#   tokenizer.json
#   tokenizer_config.json
#   vocab.txt
#   special_tokens_map.json
#   model.safetensors (or pytorch_model.bin)
# Place them in a directory, e.g. /path/to/your/model/
# Set the environment variable SBERT_MODEL_PATH to this directory, or edit MODEL_PATH above.
# Example: os.environ['SBERT_MODEL_PATH'] = '/path/to/your/model/'

if __name__ == "__main__":
	evaluate_on_excel(test_file='./inputs/test.xlsx', model_name='logistic_regression')
	# evaluate_on_excel(test_file='./inputs/test.xlsx', model_name='random_forest')
	# evaluate_on_excel(test_file='./inputs/test.xlsx', model_name='svm_(linear)')