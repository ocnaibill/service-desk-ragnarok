import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path

# Baixa os pacotes do NLTK silenciosamente
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Inicializa o lematizador
lemmatizer = WordNetLemmatizer()

def preprocess_pipeline(cleaned_text: str) -> list:
    """
    Parâmetros: String vinda da função anterior.
    Retorno: Lista de tokens lematizados.
    """
    if not isinstance(cleaned_text, str):
        return []
    
    tokens = word_tokenize(cleaned_text, language='portuguese')
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_path = base_dir / "data" / "dados_processados" / "chamados_higienizados.csv"
    output_dir = base_dir / "data" / "arquivos_lematizados"
    
    # Definindo os dois caminhos de saída
    output_path_1 = output_dir / "processed.csv"
    output_path_2 = output_dir / "dados_tratados.csv"

    print("Lendo os dados higienizados do Cotonete...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {input_path}.")
        return

    # CORREÇÃO: Pegando exatamente as colunas com os nomes que o Cotonete usou
    col_id = 'id'
    col_text = 'texto_final_higienizado' # Aqui está o texto limpo completo!
    col_target = 'categoria' # Aqui está a categoria real (ex: Rede, Acesso, etc)

    print("Aplicando o pipeline de lematização no texto completo...")
    
    # Aplica a lematização no texto correto
    df['tokens'] = df[col_text].apply(preprocess_pipeline)
    df['text_final'] = df['tokens'].apply(lambda x: " ".join(x))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepara o DataFrame final apenas com o que a tarefa exige
    df_final = df[[col_id, 'text_final', col_target]].copy()
    
    # Renomeia as colunas para o padrão exigido na Ação Adicional
    df_final.columns = ['id', 'text_final', 'target_category']

    # Salva o resultado
    print("Salvando os arquivos finais corrigidos...")
    df_final.to_csv(output_path_1, index=False)
    print(f"- Salvo com sucesso: {output_path_1}")
    
    df_final.to_csv(output_path_2, index=False)
    print(f"- Salvo com sucesso: {output_path_2}")
    
    print("Sucesso total! A base está pronta para a Parte 2.")

if __name__ == "__main__":
    main()