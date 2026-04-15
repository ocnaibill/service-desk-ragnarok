import pandas as pd
import spacy
from pathlib import Path

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    print("Erro: O modelo do spaCy não foi encontrado.")
    exit()

def preprocess_pipeline(cleaned_text: str) -> list:
    """
    Lematização Seletiva: Lematiza verbos para o infinitivo, 
    mas mantém substantivos no plural para preservar termos como "Banco de Dados".
    """
    if not isinstance(cleaned_text, str):
        return []
    
    doc = nlp(cleaned_text)
    lemmatized_tokens = []
    
    for token in doc:
        # Regra de Ouro: Se for verbo, lematiza (ex: relata -> relatar)
        if token.pos_ == "VERB":
            lemmatized_tokens.append(token.lemma_.lower())
        else:
            # Se não for verbo, mantém a palavra original! (ex: dados -> dados)
            lemmatized_tokens.append(token.text.lower())
            
    return lemmatized_tokens

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_path = base_dir / "data" / "dados_processados" / "chamados_higienizados.csv"
    output_path = base_dir / "data" / "arquivos_lematizados" / "dados_tratados.csv"

    print("Lendo os dados higienizados do Cotonete...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {input_path}.")
        return

    col_id = 'id'
    col_text = 'texto_final_higienizado'
    col_target = 'categoria'
    col_priority = 'prioridade'

    print("Aplicando Lematização Seletiva (só verbos)...")
    
    df['tokens'] = df[col_text].apply(preprocess_pipeline)
    df['text_final'] = df['tokens'].apply(lambda x: " ".join(x))

    # Hack rápido de NLP para juntar as preposições que o spaCy separa agressivamente
    # (Transforma "a o banco" de volta para "ao banco")
    replaces = {
        r'\ba o\b': 'ao', r'\ba os\b': 'aos',
        r'\bde o\b': 'do', r'\bde os\b': 'dos',
        r'\bem o\b': 'no', r'\bem os\b': 'nos',
        r'\bpor o\b': 'pelo', r'\bpor os\b': 'pelos'
    }
    for old, new in replaces.items():
        df['text_final'] = df['text_final'].str.replace(old, new, regex=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_final = df[[col_id, 'text_final', col_target, col_priority]].copy()
    df_final.rename(columns={col_target: 'target_category'}, inplace=True)

    print("Salvando o arquivo final de altíssima qualidade...")
    df_final.to_csv(output_path, index=False)
    print(f"Sucesso! Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()