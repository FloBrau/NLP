import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

true = pd.read_csv('group58_stage1.csv')
preds = pd.read_csv('group58_stage3.csv')

true.columns = true.columns.str.strip()
preds.columns = preds.columns.str.strip()
print(true.columns)

true_labels = true['real_news']
pred_labels = preds['real_news']

true_labels = true_labels.astype(str).str.strip().str.lower()
pred_labels = pred_labels.astype(str).str.strip().str.lower()

print(true_labels.unique())
print(pred_labels.unique())

cm = confusion_matrix(true_labels, pred_labels, labels=['no', 'yes'])
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, pos_label='yes')
recall = recall_score(true_labels, pred_labels, pos_label='yes')
f1score = f1_score(true_labels, pred_labels, pos_label='yes')

print(cm)
print(f"Accuracy:  {accuracy}")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1 Score:  {f1score}")