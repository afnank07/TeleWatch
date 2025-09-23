# Import required libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from tqdm import tqdm
import joblib

# Load dataset
df = pd.read_excel('./inputs/dataset.xlsx')
# Expecting columns: 'message', 'label' (label: 'referral' or 'not referral')
messages = df['Message'].astype(str).tolist()
labels = df['class'].map(lambda x: 1 if x == 'referral' else 0).values

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(messages, show_progress_bar=True)

# Define classifiers
classifiers = {
	'Logistic Regression': LogisticRegression(max_iter=1000),
	'SVM (Linear)': SVC(kernel='linear', probability=True),
	'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Cross-validation setup
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

results = {}
print("Starting cross-validation for classifiers...")
for clf_name, clf in tqdm(classifiers.items(), desc="Classifiers", leave=True):
	precisions, recalls, f1s, accs = [], [], [], []
	print(f"\nTraining {clf_name}...")
	for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(embeddings, labels), total=k, desc=f"{clf_name} folds", leave=False)):
		print(f"  Fold {fold+1}/{k}")
		X_train, X_test = embeddings[train_idx], embeddings[test_idx]
		y_train, y_test = labels[train_idx], labels[test_idx]
		# Check if both classes are present in training set
		if len(np.unique(y_train)) < 2:
			print(f"    Skipping fold {fold+1}: only one class present in training set.")
			continue
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		precisions.append(precision_score(y_test, y_pred))
		recalls.append(recall_score(y_test, y_pred))
		f1s.append(f1_score(y_test, y_pred))
		accs.append(accuracy_score(y_test, y_pred))
	results[clf_name] = {
		'precision': np.mean(precisions),
		'recall': np.mean(recalls),
		'f1': np.mean(f1s),
		'accuracy': np.mean(accs)
	}
	# Save the trained model on the full dataset for later inference
	if len(np.unique(labels)) >= 2:
		clf.fit(embeddings, labels)
		joblib.dump(clf, f'models/{clf_name.replace(" ", "_").lower()}.joblib')
		print(f"Saved {clf_name} model to models/{clf_name.replace(' ', '_').lower()}.joblib")

# Print results
for clf_name, metrics in results.items():
	print(f"\nClassifier: {clf_name}")
	print(f"Precision: {metrics['precision']:.3f}")
	print(f"Recall: {metrics['recall']:.3f}")
	print(f"F1 Score: {metrics['f1']:.3f}")
	print(f"Accuracy: {metrics['accuracy']:.3f}")


# Optional: Print detailed classification report for best classifier
best_clf = max(results, key=lambda k: results[k]['f1'])
print(f"\nBest classifier by F1: {best_clf}")
clf = classifiers[best_clf]
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(embeddings, labels):
	X_train, X_test = embeddings[train_idx], embeddings[test_idx]
	y_train, y_test = labels[train_idx], labels[test_idx]
	if len(np.unique(y_train)) < 2:
		print("Skipping final report: only one class present in training set.")
		continue
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred, target_names=['not referral', 'referral']))
	break  # Only show for first fold

# --- Inference code example ---
def load_model_and_predict(message, model_name='logistic_regression'):
	"""
	Loads a trained model and SBERT, encodes the message, and returns the prediction.
	model_name: 'logistic_regression', 'svm_(linear)', or 'random_forest'
	"""
	# Load SBERT model
	sbert = SentenceTransformer('all-MiniLM-L6-v2')
	# Load classifier
	clf = joblib.load(f'models/{model_name}.joblib')
	# Encode message
	emb = sbert.encode([message])
	# Predict
	pred = clf.predict(emb)[0]
	label = 'referral' if pred == 1 else 'not referral'
	print(f"Prediction: {label}")
	return label

# Example usage:
# load_model_and_predict("Your message text here", model_name='logistic_regression')
