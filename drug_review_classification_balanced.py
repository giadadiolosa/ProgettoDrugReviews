'''
drug_review_summary_classification.py
Progetto: Classificazione sentiment (negativa / neutra / positiva)
Dataset: Drug Review (Druglib.com)
Output: crea report, matrice di confusione, modello e 3 CSV riassuntivi
'''

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from tabulate import tabulate 

# --- Percorsi dei file ---
TRAIN_FILE = "drugLibTrain_raw.tsv"
TEST_FILE = "drugLibTest_raw.tsv"
OUTPUT_DIR = "outputsBalanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Caricamento dati ---
def load_data(train_file, test_file):
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("I file TSV non sono stati trovati nella cartella corrente!")
    df_train = pd.read_csv(train_file, sep="\t", header=0, encoding='utf-8', dtype=str)
    df_test = pd.read_csv(test_file, sep="\t", header=0, encoding='utf-8', dtype=str)
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

# --- Preprocessamento ---
def preprocess(df):
    df['text'] = df[['benefitsReview','sideEffectsReview','commentsReview']].fillna('').agg(' '.join, axis=1)

    def to_label(r):
        try:
            r = float(r)
        except:
            return None
        if r <= 3:
            return 'negativa'
        elif r <= 7:
            return 'neutra'
        else:
            return 'positiva'

    df['sentiment'] = df['rating'].apply(to_label)
    df = df.dropna(subset=['sentiment', 'text'])
    return df[['text','sentiment']]

# --- Training e valutazione ---
def train_and_evaluate(df):
    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', max_df=0.9, min_df=3),
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
        
    )
    
    print("Addestramento del modello...")
    model.fit(X_train, y_train)
    print("Addestramento completato.")
    
    y_pred = model.predict(X_test)

    # --- Classification report ---
    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR,'classification_report.txt'),'w',encoding='utf-8') as f:
        f.write(report)
    print("Classification report salvato.")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=['negativa','neutra','positiva'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negativa','neutra','positiva'])
    disp.plot()
    plt.savefig(os.path.join(OUTPUT_DIR,'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix salvata.")

    # --- Creazione CSV riassuntivi ---
    preds_df = pd.DataFrame({
        'short_text': X_test.str[:100],  
        'true': y_test.values,
        'predicted': y_pred
    })
    preds_df.to_csv(os.path.join(OUTPUT_DIR,'predictions_summary.csv'), index=False)

    summary = pd.DataFrame({
        'counts_true': y_test.value_counts(),
        'counts_predicted': pd.Series(y_pred).value_counts()
    })
    summary.to_csv(os.path.join(OUTPUT_DIR,'summary_counts.csv'))

    df_examples = preds_df.groupby('predicted').head(10)
    df_examples.to_csv(os.path.join(OUTPUT_DIR,'examples.csv'), index=False)

    # --- SALVATAGGIO MODELLO ---
    with open(os.path.join(OUTPUT_DIR,'trained_model.pkl'),'wb') as f:
        pickle.dump(model,f)
    print("Modello salvato.")

    # --- TABELLA BELLA E LEGGIBILE ---
    print("\n📊 ANTEPRIMA PREDIZIONI (prime 10 righe):\n")
    print(tabulate(preds_df.head(10), headers="keys", tablefmt="grid", showindex=False))

    print("\nTutti i file di output sono nella cartella 'outputs'.")


# --- Main ---
def main():
    df = load_data(TRAIN_FILE, TEST_FILE)
    df_clean = preprocess(df)
    train_and_evaluate(df_clean)

if __name__ == "__main__":
    main()
