import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, label_names=None):
    y_pred = model.predict(X_test)

    model_name = model.__class__.__name__

    # --- Métricas textuais ---
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)

    print(f"Modelo: {model_name}")
    print(f"Acurácia: {accuracy:.4f}\n")
    print("Relatório de Classificação:")
    print(report)

    # --- Diretório de saída ---
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Salvar relatório em texto ---
    report_path = reports_dir / f"{model_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Acurácia: {accuracy:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(report)

    # --- Matriz de Confusão ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Matriz de Confusão — {model_name}")
    plt.tight_layout()

    matrix_path = reports_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.close()

    print(f"Relatório salvo em:          {report_path}")
    print(f"Matriz de confusão salva em: {matrix_path}")

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def check_class_balance(df: pd.DataFrame, target_col: str) -> dict:
    """
    Calcula a frequência absoluta e relativa de cada categoria no dataset
    e identifica se o modelo pode ser viciado em uma classe majoritária.
    """
    total = len(df)
    contagem = df[target_col].value_counts()
    percentuais = df[target_col].value_counts(normalize=True) * 100

    resultado = {}
    print("\n" + "="*50)
    print("Diagnóstico de Viés (Balanceamento de Classes)")
    print("="*50)
    for categoria in contagem.index:
        absoluto = contagem[categoria]
        relativo = percentuais[categoria]
        resultado[categoria] = {"absoluto": absoluto, "percentual": relativo}
        print(f"  {categoria:<20} {absoluto:>5} amostras  ({relativo:.1f}%)")

    # Identifica possível viés
    max_pct = percentuais.max()
    min_pct = percentuais.min()
    razao = max_pct / min_pct

    print(f"\n  Classe majoritária: {percentuais.idxmax()} ({max_pct:.1f}%)")
    print(f"  Classe minoritária: {percentuais.idxmin()} ({min_pct:.1f}%)")
    print(f"  Razão maior/menor:  {razao:.1f}x")

    if razao >= 5:
        print("  ⚠️  DESBALANCEAMENTO SEVERO — alto risco de viés.")
    elif razao >= 2:
        print("  ⚠️  DESBALANCEAMENTO MODERADO — monitorar métricas por classe.")
    else:
        print("  ✅  Dataset razoavelmente balanceado.")

    print("="*50)
    return resultado


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    # --- Carregar dados ---
    data_path = base_dir / "data" / "arquivos_lematizados" / "dados_tratados.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dados não encontrados em '{data_path}'.")

    df = pd.read_csv(data_path)

    # --- Diagnóstico de viés ---
    check_class_balance(df, "target_category")

    # --- Carregar vetorizador ---
    vectorizer_path = base_dir / "models" / "TfidfVectorizer.joblib"
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vetorizador não encontrado em '{vectorizer_path}'.\n"
            "Adicione 'joblib.dump(vectorizer, BASE_DIR / \"models\" / \"TfidfVectorizer.joblib\")' "
            "no run_training.py após o vectorize_data."
        )

    vectorizer = joblib.load(vectorizer_path)

    # --- Reproduzir o mesmo split do run_training.py ---
    X_sparse = vectorizer.transform(df["text_final"].astype(str))
    y = df["target_category"].values

    _, X_test, _, y_test = train_test_split(
        X_sparse, y, test_size=0.2, random_state=42, stratify=y
    )

    label_names = sorted(set(y_test))

    # --- Avaliar todos os modelos salvos em models/ ---
    models_dir = base_dir / "models"
    model_files = [
        p for p in models_dir.glob("*.joblib")
        if p.stem != "TfidfVectorizer"
    ]

    if not model_files:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado em '{models_dir}'.\n"
            "Execute 'python src/models/run_training.py' antes."
        )

    for model_path in sorted(model_files):
        print(f"\n{'='*50}")
        print(f"Avaliando: {model_path.name}")
        print('='*50)
        model = joblib.load(model_path)
        evaluate_model(model, X_test, y_test, label_names=label_names)

    print("\nAvaliação concluída.")