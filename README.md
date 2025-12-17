# Progetto Drug Reviews
## Introduzione
Abbiamo analizzato il Drug Reviews Dataset (https://www.google.com/search?q=Druglib.com) che contiene recensioni da parte di pazienti su farmaci specifici, utilizzato per studiare la loro efficacia e i loro effetti collaterali, oltre a permettere di fare la classificazione del sentiment (Positivo, Neutro o Negativo).
### Il dataset è suddiviso in:
-> un set di traning (75%), chiamato "drugLibTrain_raw.tsv"
-> un set di test (25%), chiamato "drugLibTest_raw.tsv"
### Modello di Machine Learning utilizzato
E' stato utilizzato Random Forest, assieme ad un metodo di rappresentazione del testo TF-IDF
### Librerie da installare:
- pandas
- scikit-learn
- matplotlib
- tabulate
### Su terminale:
`python -m pip install pandas scikit-learn matplotlib tabulate`
### Contenuti della cartella "progetto"
- drugLibTrain_raw.tsv, drugLibTest_raw.tsv -> file scaricati da UCI Machine Learning Repository
- drug_review_classification.py -> codice del progetto
- drug_review_classification_balanced.py -> codice del progetto con aggiunto:         `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')`
#### Cartella "outputs":
  1. predictions_summary.csv -> contiene le recensioni, assieme alla classificazione reale e predetta dal modello
  2. examples.csv -> ha il medesimo contenuto ma con solo 30 recensioni (per vederne un esempio)
  3. summary_counts.csv -> ha un riassunto delle classificazioni reali e predette
  4. confusion_matrix.png -> è un'immagine prodotta dal codice che rappresenta la matrice di confusione
  5. classification_report -> file dove si può osservare quanto è stato preciso il nostro modello
#### Cartella "outputsBalanced": 
contiene tutti i file precedenti, prodotti all'esecuzione del file "drug_review_classification_balanced.py"
#### Cartella "studio random forest":
contiene un file, sempre prodotto dal nostro codice, che indica il modo in cui ha lavorato il nostro modello (trained_model.pkl) assieme ad un file python realizzato per visualizzarlo (trained-randomforest.py)


Nota: i file .DS_Store e .gitattributes sono da ignorare








