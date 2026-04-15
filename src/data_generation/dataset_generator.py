import ollama
import csv #poderá ser usado futuramente.
import random
import time
import sys
import json
from pathlib import Path

MODELO = "phi3:3.8b-mini-4k-instruct-q4_K_M"
TOTAL_CHAMADOS = 10000
ARQUIVO_RAW = "raw.txt"

CATEGORIAS={
    "Acesso": 3000,
    "Hardware": 2500,
    "Rede": 1500,
    "Comunicação": 1500,
    "Infraestrutura": 1000,
    "Banco de Dados": 500
}

PRIORIDADES ={
    "Alta": 0.10,
    "Media": 0.30,
    "Baixa": 0.60
}

EXEMPLOS = {
    "Acesso": "Usuário não consegue logar|O funcionário informa que esqueceu a senha do sistema e já tentou recuperar, mas o e-mail não chega.",
    "Hardware": "Computador não liga|Ao apertar o botão power, a máquina não dá nenhum sinal de vida. Já testei outra tomada.",
    "Rede": "Wi-Fi cai constantemente|O sinal do Wi-Fi some várias vezes ao dia, principalmente no período da tarde.",
    "Comunicação": "E-mail não envia anexos|Quando tento enviar um arquivo PDF pelo Outlook, dá erro de tamanho excedido.",
    "Infraestrutura": "Sistema lento|O ERP da empresa está respondendo muito devagar, levando minutos para abrir uma tela.",
    "Banco de Dados": "Consulta travando|A query que gera o relatório mensal está demorando mais de 1 hora e às vezes nem finaliza."
}


def gerar_chamado(categoria, prioridade):
    ex_tit, ex_desc = EXEMPLOS[categoria].split("|", 1)
    prompt = f"""Você é um atendente de Service Desk. Crie um chamado de suporte técnico REALISTA e CONCISO.
Categoria: {categoria}
Prioridade: {prioridade}

Responda APENAS com um objeto JSON válido contendo as chaves "titulo" (máx 10 palavras) e "descricao" (20 a 60 palavras). Não adicione nenhum texto antes ou depois do JSON.

Exemplo de formato esperado:
{{
  "titulo": "{ex_tit}",
  "descricao": "{ex_desc}"
}}

Agora gere um chamado inédito em JSON para a categoria {categoria} com prioridade {prioridade}:"""
    
    try:
        response = ollama.generate(
            model=MODELO, 
            prompt=prompt, 
            format="json", 
            options={"temperature": 0.7}
        )
        
        texto = response['response'].strip()
        
        dados = json.loads(texto)
        
        titulo = dados.get("titulo", "Chamado Sem Título")
        descricao = dados.get("descricao", "Descrição não fornecida.")
        
        titulo = titulo.replace('\n', ' ').replace('|', ' ')
        descricao = descricao.replace('\n', ' ').replace('|', ' ')
        
        return titulo.strip(), descricao.strip()
        
    except json.JSONDecodeError:
        print(f"Erro de JSON retornado pelo modelo: {texto}. Usando fallback.")
        fallback_desc = f"Problema relatado na categoria {categoria} com prioridade {prioridade}. Necessita análise técnica."
        return f"Chamado {categoria}", fallback_desc
    except Exception as e:
        print(f"Erro ao gerar chamado: {e}. Usando fallback.")
        fallback_desc = f"Problema relatado na categoria {categoria} com prioridade {prioridade}. Necessita análise técnica."
        return f"Chamado {categoria}", fallback_desc
    
def salvar_chamado(id_chamado, titulo, descricao, categoria, prioridade):
    """Append no arquivo raw.txt no formato correto"""
    with open(ARQUIVO_RAW, "a", encoding="utf-8") as f:
        f.write(f"{id_chamado}|{titulo}|{descricao}|{categoria}|{prioridade}\n")

def main():
    try:
        ollama.list()
    except Exception:
        print("Ollama não está rodando. Inicie com 'ollama serve' e instale o modelo:")
        print(f"  ollama pull {MODELO}")
        sys.exit(1)
    
    chamados_ja_gerados = {cat: 0 for cat in CATEGORIAS.keys()}
    ultimo_id = 0
    
    arquivo_path = Path(ARQUIVO_RAW)
    
    if arquivo_path.exists():
        print("Arquivo existente encontrado. Calculando progresso...")
        with open(arquivo_path, "r", encoding="utf-8") as f:
            linhas = f.readlines()
            ultimo_id = len(linhas)
            for linha in linhas:
                partes = linha.strip().split('|')
                if len(partes) >= 4:
                    categoria_lida = partes[3]
                    if categoria_lida in chamados_ja_gerados:
                        chamados_ja_gerados[categoria_lida] += 1
                        
        print(f"Já temos {ultimo_id} chamados no arquivo. Retomando a geração...")
    else:
        print("Nenhum arquivo anterior encontrado. Começando do zero...")

    chamados_para_gerar = []
    for cat, qtd_total in CATEGORIAS.items():
        qtd_faltante = qtd_total - chamados_ja_gerados[cat]
        
        if qtd_faltante > 0:
            for _ in range(qtd_faltante):
                rand = random.random()
                if rand < PRIORIDADES["Alta"]:
                    prior = "Alta"
                elif rand < PRIORIDADES["Alta"] + PRIORIDADES["Media"]:
                    prior = "Media"
                else:
                    prior = "Baixa"
                chamados_para_gerar.append((cat, prior))
    
    if not chamados_para_gerar:
        print("\n✅ Todos os chamados já foram gerados! O dataset está completo.")
        return

    random.shuffle(chamados_para_gerar)
    
    for idx, (categoria, prioridade) in enumerate(chamados_para_gerar, start=ultimo_id + 1):
        print(f"Gerando {idx}/{TOTAL_CHAMADOS}: {categoria} - {prioridade}")
        titulo, descricao = gerar_chamado(categoria, prioridade)
        salvar_chamado(idx, titulo, descricao, categoria, prioridade)
        time.sleep(0.5)
    
    print(f"\n✅ Geração concluída! Arquivo salvo em {ARQUIVO_RAW}")

if __name__ == "__main__":
    main()