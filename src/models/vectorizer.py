from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def vectorize_data(dataframe, coluna):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(dataframe[coluna].astype(str))

    ### Parte para descomentar caso o Iguinho queira ver o funcionamento da vetorização
    # X_amostra = X[:5].toarray()
    # df_tfidf = pd.DataFrame(X_amostra, columns=vectorizer.get_feature_names_out())
    # df_tfidf = df_tfidf.loc[:, (df_tfidf != 0).any(axis=0)]
    # print(df_tfidf)
    # print(X[0])
    # feature_names = vectorizer.get_feature_names_out()
    # indice = 675
    # print(f"A palavra no índice {indice} é: {feature_names[indice]}")

    return X, vectorizer
