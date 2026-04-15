#Manos, para funcionar precisa dar um "python -m pip install --upgrade pip setuptools wheel", um "python -m pip install spacy" e um "python -m spacy download pt_core_news_sm"
# Se não der esses pip install que citei, até funciona. Mas a retirada das preposições fica mais imprecisa

from __future__ import annotations
from pathlib import Path
import csv
import re
import unicodedata

try:
    import spacy
except ImportError:
    spacy = None  

BASE_DIR = Path(__file__).resolve().parents[2]
CAMINHO_ENTRADA = BASE_DIR / "data" / "raw" / "raw.txt"
CAMINHO_SAIDA = (
    BASE_DIR / "data" / "dados_processados" / "chamados_higienizados.csv"
)

def carregar_modelo_spacy() -> "spacy.Language | None":
    if spacy is None:
        return None
    for nome_modelo in ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg", "pt"]:
        try:
            return spacy.load(nome_modelo) 
        except Exception:
            continue
    return None

NLP_PORTUGUES = carregar_modelo_spacy()
PREPOSICOES_FALLBACK = {
    "ante",
    "apos",
    "após",
    "ate",
    "até",
    "com",
    "contra",
    "de",
    "desde",
    "em",
    "entre",
    "para",
    "perante",
    "por",
    "sem",
    "sob",
    "sobre",
    "tras",
    "trás",
}

def limpar_texto_bruto(texto_bruto: str) -> str:
    """
    Aplica sanitização básica ao texto após a remoção de preposições.

    Transforma o texto em minúsculo, remove quebras de linha, retira
    acentos, pontuação e caracteres especiais e normaliza espaços.
    """
    if texto_bruto is None:
        return ""

    texto_limpo = texto_bruto.lower()
    texto_limpo = texto_limpo.replace("\n", " ").replace("\r", " ")
    texto_limpo = unicodedata.normalize("NFD", texto_limpo)
    texto_limpo = "".join(
        c for c in texto_limpo if unicodedata.category(c) != "Mn"
    )
    texto_limpo = re.sub(r"[^\w\s]", " ", texto_limpo)
    texto_limpo = texto_limpo.replace("_", " ")
    texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()

    return texto_limpo

def remover_preposicoes(texto_original: str) -> str:
    if NLP_PORTUGUES is not None:
        doc = NLP_PORTUGUES(texto_original)
        tokens = [token.text for token in doc if token.pos_ != "ADP"]
        return " ".join(tokens)

    tokens_originais = texto_original.split()
    tokens_filtrados = [
        token
        for token in tokens_originais
        if token.lower() not in PREPOSICOES_FALLBACK
    ]
    return " ".join(tokens_filtrados)

def higienizar_arquivo(
    caminho_entrada: Path = CAMINHO_ENTRADA,
    caminho_saida: Path = CAMINHO_SAIDA,
) -> None:
    """
    Lê o arquivo de entrada e gera um CSV higienizado com remoção de
    preposições. As colunas de saída são:
    id, titulo_higienizado, descricao_higienizada, texto_final_higienizado,
    categoria, prioridade.
    """
    if not caminho_entrada.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {caminho_entrada}"
        )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    total_linhas = 0
    total_validas = 0
    total_invalidas = 0

    with caminho_entrada.open("r", encoding="utf-8", newline="") as arquivo_in, \
         caminho_saida.open("w", encoding="utf-8", newline="") as arquivo_out:

        escritor = csv.DictWriter(
            arquivo_out,
            fieldnames=[
                "id",
                "titulo_higienizado",
                "descricao_higienizada",
                "texto_final_higienizado",
                "categoria",
                "prioridade",
            ],
        )
        escritor.writeheader()

        for numero_linha, linha in enumerate(arquivo_in, start=1):
            total_linhas += 1
            linha = linha.strip()

            if not linha:
                continue

            partes = linha.split("|")
            if len(partes) != 5:
                total_invalidas += 1
                print(
                    f"[AVISO] Linha {numero_linha} ignorada: "
                    f"esperava 5 campos, mas vieram {len(partes)}."
                )
                continue

            id_chamado, titulo, descricao, categoria, prioridade = partes

            titulo_sem_preposicoes = remover_preposicoes(titulo)
            descricao_sem_preposicoes = remover_preposicoes(descricao)

            titulo_higienizado = limpar_texto_bruto(titulo_sem_preposicoes)
            descricao_higienizada = limpar_texto_bruto(descricao_sem_preposicoes)

            texto_final_higienizado = (
                f"{titulo_higienizado} {descricao_higienizada}"
            ).strip()

            escritor.writerow(
                {
                    "id": id_chamado,
                    "titulo_higienizado": titulo_higienizado,
                    "descricao_higienizada": descricao_higienizada,
                    "texto_final_higienizado": texto_final_higienizado,
                    "categoria": categoria,
                    "prioridade": prioridade,
                }
            )
            total_validas += 1

    print("Higienização com remoção de preposições finalizada.")
    print(f"Linhas lidas: {total_linhas}")
    print(f"Linhas válidas processadas: {total_validas}")
    print(f"Linhas inválidas ignoradas: {total_invalidas}")
    print(f"Arquivo gerado: {caminho_saida}")

if __name__ == "__main__":
    higienizar_arquivo()
