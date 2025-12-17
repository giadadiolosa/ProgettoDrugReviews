import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------
# 1. Caricamento del Modello
# ----------------------------------------------------------------------
try:
    # Carica il modello in memoria. Sostituisci 'trained_model.pkl' se necessario.
    pipeline_caricata = joblib.load('trained_model.pkl')
    print("✅ Modello (Pipeline) caricato con successo!")

except FileNotFoundError:
    print("ERRORE: Assicurati che il file 'trained_model.pkl' sia nella directory corretta.")
    exit()

# ----------------------------------------------------------------------
# 2. Verifica e Estrazione Componenti
# ----------------------------------------------------------------------

# Una pipeline ha due componenti principali (TF-IDF e Random Forest)
if isinstance(pipeline_caricata, Pipeline) and len(pipeline_caricata.steps) >= 2:
    
    # Estrae il TfidfVectorizer (step di trasformazione)
    tfidf_vectorizer = pipeline_caricata.named_steps.get('tfidfvectorizer') 
    
    # Estrae il RandomForestClassifier (step di classificazione)
    rf_classifier = pipeline_caricata.named_steps.get('randomforestclassifier') 
    
    if tfidf_vectorizer is None or rf_classifier is None:
        # Se i nomi non sono 'tfidfvectorizer' e 'randomforestclassifier', prova con gli indici
        tfidf_vectorizer = pipeline_caricata.steps[0][1]
        rf_classifier = pipeline_caricata.steps[-1][1]
        
    print("\n---------------------------------------------------------")
    print("Dettagli del Modello Caricato:")
    print("---------------------------------------------------------")
    
    # ----------------------------------------------------------------------
    # 3. Analisi del TF-IDF (Le variabili/Features)
    # ----------------------------------------------------------------------
    
    # Ottieni la lista delle parole (features) che il modello ha imparato
    features = tfidf_vectorizer.get_feature_names_out()
    
    print(f"Tipo di Trasformatore: {type(tfidf_vectorizer).__name__}")
    print(f"Dimensione del Vocabolario (Numero di Features): {len(features):,}")
    print(f"Esempi di Features: {features[::int(len(features)/10)][:10]}") # Stampa 10 esempi
    
    # ----------------------------------------------------------------------
    # 4. Analisi del Random Forest (L'Importanza delle Variabili)
    # ----------------------------------------------------------------------
    
    print(f"\nTipo di Classificatore: {type(rf_classifier).__name__}")
    print(f"Numero di Alberi nella Foresta: {rf_classifier.n_estimators}")
    print(f"Profondità Massima (Max Depth): {rf_classifier.max_depth}")
    
    # Estrae i punteggi di importanza dal Random Forest
    importances = rf_classifier.feature_importances_
    
    # Crea un DataFrame per analizzare le parole più importanti
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Stampa le 10 parole (features TF-IDF) più importanti per la classificazione
    print("\n---------------------------------------------------------")
    print("Le 10 Features (Parole TF-IDF) più Importanti:")
    print("---------------------------------------------------------")
    print(feature_importance_df.head(10).to_string(index=False))
    
else:
    print("\n⚠️ ATTENZIONE: Il file .pkl caricato non sembra essere una pipeline scikit-learn standard o ha un formato inatteso.")
    print("È possibile che contenga solo il Random Forest o un altro oggetto non supportato da questa analisi.")
