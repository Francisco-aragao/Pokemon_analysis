import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

df = pd.read_csv('./Data/tournaments.csv')
df = df[df['region_tournament'].notna()]

# eliminando cartas do tipo treinador
df = df[df['type_card'] != 'Trainer']
regions = df['region_tournament'].unique()

# Agrupando cartas de cada deck no dataset
def agrupa_deck_cartas(df_region):
    #filtra torneios da regiao
    torneios = df_region['id_tournament'].unique()

    lista_decks = []
    for torneio in torneios:
        df_torneio = df_region[df_region['id_tournament'] == torneio]
        jogadores = df_torneio['id_player'].unique()
        for jogador in jogadores:
            df_jogador = df_torneio[df_torneio['id_player'] == jogador]
            cartas_jogador = df_jogador['name_card'].unique()
            lista_decks.append(cartas_jogador)
    return lista_decks

for region in regions:
    print(f"region: {region}")
    df_region = df[(df['region_tournament'] == region) & (df['year_tournament'] == 2019)]
    lista_decks = agrupa_deck_cartas(df_region)

    # Cria um dataframe que as colunas sao todas as cartas existentes no dataset
    cartas = df_region['name_card'].unique()

    # Lista para armazenar os dados dos decks
    data = []
    for deck in lista_decks:
        pokemons_no_deck = {carta: (carta in deck) for carta in cartas}
        data.append(pokemons_no_deck)
    
    # Converte a lista de dicionários em um DataFrame
    df_decks_cartas = pd.DataFrame(data, columns=cartas)

    # FP-Growth
    frequent_itemsets = fpgrowth(df_decks_cartas, min_support=0.2, use_colnames=True)
    frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values(by='lift', ascending=False).drop(['antecedent support', 'consequent support', 'leverage', 'conviction', 'zhangs_metric'], axis=1)
    print(f"Total de decks: {len(lista_decks)}")
    print(frequent_itemsets)
    print(rules)
    print("\n\n")
