import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split

from models.vectorizer import vectorize_data
from models.trainer import train_linear_model

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "arquivos_lematizados" / "dados_tratados.csv"

def main():
    print("Carregando dados lematizados...")
    df = pd.read_csv(DATA_PATH)

    X_sparse, vectorizer = vectorize_data(df, "text_final")
    y = df["target_category"].values

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
        modelo_treinado = train_linear_model(modelo, X_train, y_train)
        acuracia = modelo_treinado.score(X_test, y_test)
        print(f"  Acurácia em teste: {acuracia:.4f}")
        print(f"  Artefato salvo em: models/{nome}.joblib")

    print("\nTreinamento concluído.")

if __name__ == "__main__":
    main()
