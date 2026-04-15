from pathlib import Path
import csv
import re
import unicodedata


CAMINHO_ENTRADA = Path("data/raw/raw.txt")
CAMINHO_SAIDA = Path("data/dados_processados/chamados_higienizados.csv")


def limpar_texto(texto_bruto: str) -> str:
    """
    Recebe um texto bruto e devolve um texto:
    - em minúsculo
    - sem acentos
    - sem pontuação
    - sem caracteres especiais
    - sem quebras de linha
    - com espaços normalizados
    """
    if texto_bruto is None:
        return ""

    texto_limpo = texto_bruto.lower()

    texto_limpo = texto_limpo.replace("\n", " ").replace("\r", " ")

    texto_limpo = unicodedata.normalize("NFD", texto_limpo)
    texto_limpo = "".join(
        caractere
        for caractere in texto_limpo
        if unicodedata.category(caractere) != "Mn"
    )

    texto_limpo = re.sub(r"[^\w\s]", " ", texto_limpo)
    texto_limpo = texto_limpo.replace("_", " ")
    texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()

    return texto_limpo


def higienizar_arquivo(
    caminho_entrada: Path = CAMINHO_ENTRADA,
    caminho_saida: Path = CAMINHO_SAIDA
) -> None:
    """
    Lê o raw.txt no formato:
    ID|TITULO|DESCRICAO|CATEGORIA|PRIORIDADE

    E gera um CSV novo com as colunas:
    id, titulo_higienizado, descricao_higienizada, texto_final_higienizado,
    categoria, prioridade
    """
    if not caminho_entrada.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {caminho_entrada}"
        )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    total_linhas = 0
    total_validas = 0
    total_invalidas = 0

    with caminho_entrada.open("r", encoding="utf-8", newline="") as arquivo_entrada, \
         caminho_saida.open("w", encoding="utf-8", newline="") as arquivo_saida:

        escritor_csv = csv.DictWriter(
            arquivo_saida,
            fieldnames=[
                "id",
                "titulo_higienizado",
                "descricao_higienizada",
                "texto_final_higienizado",
                "categoria",
                "prioridade",
            ],
        )
        escritor_csv.writeheader()

        for numero_linha, linha in enumerate(arquivo_entrada, start=1):
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

            titulo_higienizado = limpar_texto(titulo)
            descricao_higienizada = limpar_texto(descricao)

            texto_final_higienizado = (
                f"{titulo_higienizado} {descricao_higienizada}"
            ).strip()

            escritor_csv.writerow(
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

    print("Higienização finalizada com sucesso.")
    print(f"Linhas lidas: {total_linhas}")
    print(f"Linhas válidas processadas: {total_validas}")
    print(f"Linhas inválidas ignoradas: {total_invalidas}")
    print(f"Arquivo gerado: {caminho_saida}")


if __name__ == "__main__":
    higienizar_arquivo()
