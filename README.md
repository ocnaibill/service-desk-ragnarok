# Classificador de Chamados com IA e NLP

Este projeto consiste em um pipeline de Engenharia de Software voltado para a classificação automatizada de textos (chamados técnicos). O sistema realiza desde a geração de dados sintéticos até o treinamento modular de modelos lineares e avaliação de métricas de performance.

## 📋 Sumário
1. Requisitos Mínimos
2. Configuração de Ambiente Virtual
3. Instalação de Dependências
4. Como Rodar o Projeto

---

## 💻 Requisitos Mínimos

Antes de iniciar, certifique-se de ter instalado em sua máquina:
* **Python 3.13** ou superior.
* **Pip** (Gerenciador de pacotes do Python).
* **Git** (Para versionamento).

---

## 🌐 Configuração de Ambiente Virtual (venv)

O uso de um ambiente virtual isola as dependências deste projeto, garantindo que as bibliotecas não conflitem com outras instalações no seu sistema.

1. No terminal, acesse a pasta raiz do projeto:
   ```bash
   cd nome-do-seu-repositorio
   ```

2. Crie o ambiente virtual:
   * **Windows:**
     ```bash
     python -m venv venv
     ```

3. Ative o ambiente:
   * **Windows (PowerShell):**
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
---

## 📦 Instalando Dependências

Com o ambiente virtual **ativado**, execute o comando abaixo para instalar todas as bibliotecas necessárias (Pandas, Scikit-Learn, NLTK, Joblib):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Dica:** O projeto utiliza o `NLTK` para lematização. Na primeira execução, o script baixará automaticamente os pacotes necessários.

---

## 🚀 Como Rodar o Projeto

O projeto foi construído seguindo princípios de modularidade. Você pode executar o fluxo completo ou etapas específicas:

### 1. Executar o Pipeline Completo
Para gerar os dados, limpar, preprocessar, treinar o modelo e gerar as métricas de uma só vez:

Observação: Certifique-se que está na pasta raiz do projeto
```bash
python src/main.py
```

### 2. Executar Etapas Individuais (Scripts Modulares)
Caso queira rodar apenas partes específicas do processo:

Observação: Certifique-se que está na pasta raiz do projeto
* **Gerar/Limpar Dados:**
  ```bash
  python data/preprocess.py
  ```
* **Treinar Modelo:**
  ```bash
  python model/train.py
  ```
* **Avaliar Resultados (Matriz de Confusão):**
  ```bash
  python reports/report.py
  ```

---
