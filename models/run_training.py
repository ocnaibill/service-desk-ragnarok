import sys
from pathlib import Path

# src/models/ é onde estão vectorizer.py e trainer.py
# run_training.py está em service-desk-ragnarok/models/
# parents[0] = service-desk-ragnarok/models/
# parents[1] = service-desk-ragnarok/
SRC_MODELS = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(SRC_MODELS))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split

from vectorizer import vectorize_data
from trainer import train_linear_model

# BASE_DIR = service-desk-ragnarok/
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "arquivos_lematizados" / "dados_tratados.csv"

def main():
    print("Carregando dados lematizados...")
    df = pd.read_csv(DATA_PATH)

    X_sparse, vectorizer = vectorize_data(df, "text_final")
    y = df["target_category"].values

    # Salva na mesma pasta que report.py espera: service-desk-ragnarok/models/
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / "TfidfVectorizer.joblib")
    print(f"Vetorizador salvo em: {models_dir / 'TfidfVectorizer.joblib'}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse, y, test_size=0.2, random_state=42, stratify=y
    )

    modelos = [
        LogisticRegression(max_iter=1000),
        RidgeClassifier(),
        SGDClassifier(max_iter=1000, random_state=42),
    ]

    for modelo in modelos:
        nome = modelo.__class__.__name__
        print(f"\nTreinando {nome}...")
        modelo_treinado = train_linear_model(modelo, X_train, y_train, models_dir=models_dir)
        acuracia = modelo_treinado.score(X_test, y_test)
        print(f"  Acurácia em teste: {acuracia:.4f}")
        print(f"  Artefato salvo em: {models_dir / f'{nome}.joblib'}")

    print("\nTreinamento concluído.")

if __name__ == "__main__":
    main()