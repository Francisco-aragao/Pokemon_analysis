import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('./Data/tournaments.csv')
df.replace(np.nan, 0, inplace=True)

df = df[df['region_tournament'] != 0]
# eliminando cartas do tipo treinador
df = df[df['type_card'] != 'Trainer']
regions = df['region_tournament'].unique()

# Agrupando cartas de cada jogador no dataset
def agrupa_jogador_cartas(df):
    dict_jogador_cartas = {}
    for _, row in df.iterrows():
        if row['name_player'] not in dict_jogador_cartas:
            dict_jogador_cartas[row['name_player']] = []
        dict_jogador_cartas[row['name_player']].append(row['name_card'])
    return len(dict_jogador_cartas), dict_jogador_cartas

for region in regions:
    print(f"region: {region}")
    df_region = df[(df['region_tournament'] == region) & (df['year_tournament'] == 2019)]
    jogadores = df_region['name_player'].unique()
    cartas = df_region['name_card'].unique()
    quantidade_jogadores, dict_jogador_cartas = agrupa_jogador_cartas(df_region)
    # Cria um dataframe que as colunas sao todas as cartas existentes no dataset
    df_jogadores_cartas = pd.DataFrame(columns=cartas)
    for jogador in jogadores:
        cartas_jogador = dict_jogador_cartas[jogador]
        linha = []
        for carta in cartas:
            if carta in cartas_jogador:
                linha.append(True)
            else:
                linha.append(False)
        df_jogadores_cartas.loc[len(df_jogadores_cartas)] = linha
    # Apriori
    frequent_itemsets = apriori(df_jogadores_cartas, min_support=0.2, use_colnames=True)
    frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(f"Total de jogadores: {quantidade_jogadores}")
    print(frequent_itemsets)
    print(rules)
    print("\n\n")
