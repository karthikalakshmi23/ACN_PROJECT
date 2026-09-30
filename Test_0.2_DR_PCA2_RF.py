# Test_0.2_DR_PCA2_RF.py
#
# Test size 0.2  |  PCA, 2 components  |  Random Forest (sheet #7)
#
# This is SGDC_60_40.py with the changes marked CHANGE below and nothing else.
#
# Taken from the sheet unchanged.

import os
from pathlib import Path

# CHANGE 1: run from the project folder, so the CSV names below resolve, and
# read them as latin-1. The Web Attack labels contain byte 0x96, which the
# default utf-8 decoder rejects with UnicodeDecodeError.
os.chdir(Path(__file__).resolve().parent.parent)
import numpy as np
import pandas as pd

df1 = pd.read_csv(r'C:\Users\Karthika\OneDrive\Desktop\ACN_PCA_75_Runs\Scripts\Tuesday-WorkingHours.pcap_ISCX (1).csv', low_memory=True, encoding='latin-1')
df2 = pd.read_csv(r'C:\Users\Karthika\OneDrive\Desktop\ACN_PCA_75_Runs\Scripts\Wednesday-workingHours.pcap_ISCX.csv', low_memory=True, encoding='latin-1')
df3 = pd.read_csv(r'C:\Users\Karthika\OneDrive\Desktop\ACN_PCA_75_Runs\Scripts\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True, encoding='latin-1')

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

# CHANGE 2: a stratified sample of 100000 flows, the same for every run.
# The full three captures hold about 1.3 million flows, which makes 75 runs
# impractical. At least 50 flows of every class are kept.
label = dataset.columns[-1]
dataset = pd.concat(
    [g.sample(n=min(len(g), max(50, round(100000 * len(g) / len(dataset)))), random_state=0)
     for _, g in dataset.groupby(label)],
    ignore_index=True
)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,   # CHANGE 3: test size
    random_state=0,
    stratify=y
)

# CHANGE 4: PCA, from PCA_LDA_KErnel_PCA_CODE.pdf (fitted on training data only).
# PCA follows the directions of largest variance, so the features have to be on
# a common scale first, otherwise the components just track whichever column has
# the biggest units. The scaler is fitted on the training split only.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)          # no labels: PCA is unsupervised
X_test = pca.transform(X_test)
print("\nVariance explained by 2 components : {:.4f}".format(pca.explained_variance_ratio_.sum()))

# CHANGE 5: Random Forest (sheet #7) in place of SGDClassifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=200,
    criterion='entropy',
    random_state=0
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nPrecision : {:.4f}".format(precision_score(y_test, y_pred, average='weighted')))
print("\nRecall : {:.4f}".format(recall_score(y_test, y_pred, average='weighted')))
print("\nF1 Score : {:.4f}".format(f1_score(y_test, y_pred, average='weighted')))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# CHANGE 6: save the confusion matrix as Test_0.2_DR_PCA2_RF.jpg and record the result
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

here = Path(__file__).resolve().parent
names = [str(c) for c in labelencoder_y.classes_]
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues', norm=LogNorm(vmin=1))
ax.set_xticks(range(len(names)), names, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(names)), names, fontsize=7)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Test_0.2_DR_PCA2_RF   accuracy {:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=10)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=6,
                    color='white' if cm[i, j] > cm.max() / 3 else 'black')
fig.colorbar(im, ax=ax, fraction=0.04, label='flows (log scale)')
fig.tight_layout()
fig.savefig(here / 'Test_0.2_DR_PCA2_RF.jpg', dpi=150)
plt.close(fig)

row = pd.DataFrame([{
    'Test': 0.2, 'DR': 'PCA2', 'Clas': 'RF',
    'Acc': round(accuracy_score(y_test, y_pred), 4),
    'Pre': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'Rec': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'F1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'script': 'Test_0.2_DR_PCA2_RF',
}])
out = here.parent / 'results' / '75_runs.csv'
out.parent.mkdir(exist_ok=True)
if out.exists():
    old = pd.read_csv(out)
    row = pd.concat([old[old['script'] != 'Test_0.2_DR_PCA2_RF'], row], ignore_index=True)
row.to_csv(out, index=False)
