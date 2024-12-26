import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


class ItemDataPreprocessor:
    def __init__(self, ratings_file_path, content_file_path, targets_file_path):
        self.ratings_file_path = ratings_file_path
        self.content_file_path = content_file_path
        self.targets_file_path = targets_file_path
        self.user_ratings = None
        self.item_metadata = None
        self.prediction_targets = None
        self.item_index_mapping = None

        # Arrays para viés e métricas
        self.imdb_votes_array = None
        self.imdb_ratings_array = None
        self.metascore_array = None
        self.award_nominations_array = None
        self.award_wins_array = None
        self.language_array = None

    def load_data(self):
        """
        Carrega os dados de arquivos JSON e CSV.
        """
        self.user_ratings = pd.read_json(self.ratings_file_path, lines=True).drop(columns='Timestamp', errors='ignore')
        
        unused_columns = [
            'DVD', 'BoxOffice', 'Website', 'Production', 'Response', 'Episode',
            'Response', 'totalSeasons', 'seriesID', 'Ratings', 'Poster', 'Season',
            'Rated', 'Released', 'Runtime'
        ]
        self.item_metadata = pd.read_json(self.content_file_path, lines=True).drop(columns=unused_columns, errors='ignore')
        
        self.prediction_targets = pd.read_csv(self.targets_file_path)

    @staticmethod
    def extract_award_counts(metadata, award_type):
        """
        Extrai contagens de nomeações ou vitórias a partir da coluna de prêmios.
        """
        award_pattern = rf'[0-9]+\s{award_type}'
        award_counts = (metadata['Awards']
                        .str.findall(award_pattern)
                        .str.join(',')
                        .str.replace(r'[a-zA-Z]+', '', regex=True)
                        .replace('', '0')
                        .astype(int))
        return award_counts

    @staticmethod
    def categorize_values(metadata, column_name, bins='auto', labels=None):
        """
        Categoriza valores numéricos em intervalos ou rótulos específicos.
        """
        numeric_values = pd.to_numeric(metadata[column_name], errors='coerce')  
        valid_values_mask = numeric_values.notna()  

        if bins == 'auto':
            _, bin_edges = np.histogram(numeric_values[valid_values_mask].dropna(), bins='auto')
        else:
            bin_edges = bins

        # Gera rótulos automáticos se não forem fornecidos
        if labels is None:
            labels = [f"({bin_edges[i]}, {bin_edges[i + 1]}]" for i in range(len(bin_edges) - 1)]

        # Aplica a categorização
        metadata.loc[valid_values_mask, column_name] = pd.cut(numeric_values[valid_values_mask], bins=bin_edges, labels=labels, right=False)
        metadata.loc[~valid_values_mask, column_name] = 'N/A'
        metadata[column_name] = metadata[column_name].astype(str)

    def preprocess_item_metadata(self):
        """
        Pré-processa os metadados de itens, incluindo extração e categorização de atributos.
        """
        self.item_metadata['AwardNominations'] = self.extract_award_counts(self.item_metadata, 'nomination')
        self.item_metadata['AwardWins'] = self.extract_award_counts(self.item_metadata, 'win')

        # Armazena os valores originais para cálculo de viés
        self.award_wins_array = self.item_metadata['AwardWins'].values
        # Processa atributos numéricos para cálculo de viés
        imdb_votes = (self.item_metadata['imdbVotes']
                      .str.replace(',', '', regex=False)
                      .replace('N/A', '0')
                      .astype(int))
        imdb_ratings = pd.to_numeric(self.item_metadata['imdbRating'], errors='coerce').fillna(0)
        metascore_values = pd.to_numeric(self.item_metadata['Metascore'], errors='coerce').fillna(0)
        self.imdb_votes_array = imdb_votes.values

        # Processa linguagens e categoriza idiomas raros
        self.item_metadata['PrimaryLanguage'] = (self.item_metadata['Language']
                                                 .apply(lambda x: x.split(',')[0].strip())
                                                 .replace('N/A', 'None'))
        
        # Categoriza valores numéricos
        self.categorize_values(self.item_metadata, 'imdbVotes', bins='auto')

        # Armazena arrays para viés
        self.imdb_ratings_array = imdb_ratings.values
        self.metascore_array = metascore_values.values
        self.award_nominations_array = self.item_metadata['AwardNominations'].values
        self.language_array = self.item_metadata['PrimaryLanguage'].values

        # Cria mapeamento ItemId -> índice
        self.item_index_mapping = {item_id: i for i, item_id in enumerate(self.item_metadata['ItemId'])}


class FeatureExtractor:
    def __init__(self, item_features):
        self.item_features = item_features

    @staticmethod
    def string_to_int_features(dataframe, column_name):
        """
        Converte valores de uma coluna de string para inteiros em formato de string e gera features.
        """
        def parse_and_clean(value):
            value = value.replace(',', '')
            return str(round(float(value))) if value != 'N/A' else ''
        
        # Aplica a transformação na coluna
        dataframe[column_name] = dataframe[column_name].apply(parse_and_clean)
        
        # Retorna as features únicas no formato "{column_name}:{valor}"
        unique_features = [f"{column_name}:{unique_value}" for unique_value in dataframe[column_name].drop_duplicates().tolist()]
        return unique_features


    @staticmethod
    def append_features_to_items(base_features, additional_features, feature_prefix=None):
        """
        Adiciona novas features a uma base de itens e suas respectivas features existentes.
        """
        updated_features = []
        for index, (item_id, existing_features) in enumerate(base_features):
            new_feature_values = additional_features[index]
            
            # Garante que new_feature_values seja uma lista
            if isinstance(new_feature_values, str):
                new_feature_values = [new_feature_values]
            elif not isinstance(new_feature_values, list):
                new_feature_values = [new_feature_values]
            
            # Adiciona prefixo, se fornecido
            if feature_prefix:
                new_feature_values = [feature_prefix + str(value) for value in new_feature_values]
            
            # Combina as features existentes com as novas
            updated_features.append((item_id, existing_features + new_feature_values))
        
        return updated_features


    def process_genre(self):
        self.item_features['Genre'] = self.item_features['Genre'].str.replace(' ', '')

        # Extrai os valores únicos de gêneros
        unique_genres = set()
        for values in self.item_features['Genre'].str.split(','):
            unique_genres.update(values)

        # Cria colunas de One-Hot Encoding para cada gênero
        for genre in unique_genres:
            self.item_features[genre] = self.item_features['Genre'].apply(lambda x: genre in x)

        # Gera a lista de possíveis features no formato "Genre:True" ou "Genre:False"
        possible_genre_features = [f"{genre}:False" for genre in unique_genres] + [f"{genre}:True" for genre in unique_genres]

        # Gera as features específicas para cada item
        item_genre_features = [
            [f"{genre}:{str(record[genre])}" for genre in unique_genres]
            for record in self.item_features.to_dict(orient='records')
        ]

        return possible_genre_features, item_genre_features


    def process_director(self):
        # Remove espaços e divide os valores em listas
        self.item_features['Director'] = self.item_features['Director'].str.replace(' ', '').str.split(',')

        # Extrai os valores únicos de diretores
        unique_directors = set()
        for directors in self.item_features['Director']:
            unique_directors.update(directors)

        # Lista de possíveis features para diretores no formato "dir:Nome"
        possible_director_features = [f"dir:{director}" for director in unique_directors]

        # Gera as features específicas para cada item
        item_director_features = [
            [f"dir:{director}" for director in record['Director']]
            for record in self.item_features.to_dict(orient='records')
        ]

        return possible_director_features, item_director_features

    def process_ratings(self):
        imdb_rating_features = self.string_to_int_features(self.item_features, 'imdbRating')
        metascore_features = self.string_to_int_features(self.item_features, 'Metascore')
        return imdb_rating_features, metascore_features

    def process_wins(self):
        unique_wins_categories = self.item_features['AwardWins'].unique()
        wins_features_list = [f"w:{category}" for category in unique_wins_categories]
        wins_features_per_item = [f"w:{category}" for category in self.item_features['AwardWins']]
        return wins_features_list, wins_features_per_item

    def process_votes(self):
        unique_vote_intervals = self.item_features['imdbVotes'].unique()
        vote_features_list = [f"imdbVotes:{str(interval)}" for interval in unique_vote_intervals]
        return vote_features_list

    def process_language(self):
        unique_languages = self.item_features['PrimaryLanguage'].unique().tolist()
        return unique_languages

    def build_item_features(self, genre_features_per_item, director_features_per_item, wins_features_per_item):
        item_ids_list = self.item_features['ItemId'].tolist()
        combined_features = [(item_ids_list[i], genre_features_per_item[i]) for i in range(len(genre_features_per_item))]

        # Lista de atributos e prefixos para construir as features
        feature_configs = [
            ('imdbRating', 'imdbRating:'),
            ('Metascore', 'Metascore:'),
            ('imdbVotes', 'imdbVotes:'),
            ('PrimaryLanguage', None),  # Linguagem já está pronta sem prefixo
        ]

        # Processa e adiciona cada conjunto de features
        for column_name, prefix in feature_configs:
            feature_values = self.item_features[column_name].tolist()
            formatted_features = [[f"{prefix}{value}"] if prefix else [value] for value in feature_values]
            combined_features = self.append_features_to_items(combined_features, formatted_features)

        # Adiciona features específicas como diretor e wins
        combined_features = self.append_features_to_items(combined_features, director_features_per_item)
        combined_features = self.append_features_to_items(combined_features, wins_features_per_item)

        return combined_features

    
    def generate_item_features(self):
        """
        Gera todas as possíveis features de itens e o mapeamento necessário para o LightFM.
        """
        # Processa cada categoria de features
        genre_possible_features, item_genre_features = self.process_genre()
        director_possible_features, item_director_features = self.process_director()
        rating_possible_features, metascore_possible_features = self.process_ratings()
        wins_possible_features, item_wins_features = self.process_wins()
        votes_possible_features = self.process_votes()
        language_possible_features = self.process_language()

        # Lista completa de todas as possíveis features
        all_possible_item_features = (
            genre_possible_features
            + director_possible_features
            + rating_possible_features
            + metascore_possible_features
            + votes_possible_features
            + wins_possible_features
            + language_possible_features
        )

        # Mapeamento das features de cada item no formato necessário para LightFM
        item_feature_mapping = self.build_item_features(
            item_genre_features,
            item_director_features,
            item_wins_features,
        )

        return all_possible_item_features, item_feature_mapping



def sort_for_cold_start(user_item_df, preprocessor):
    # Obtemos índices dos itens
    item_indexes = [preprocessor.item_index_mapping[item_id] for item_id in user_item_df['ItemId']]
    votes_array = preprocessor.imdb_votes_array[item_indexes]
    nominations_array = preprocessor.award_nominations_array[item_indexes]
    wins_array = preprocessor.award_wins_array[item_indexes]
    metascore_array = preprocessor.metascore_array[item_indexes]
    ratings_array = preprocessor.imdb_ratings_array[item_indexes]


    # Normalização (z-score) para todas as métricas
    votes_normalized = (votes_array - votes_array.mean()) / votes_array.std()
    nominations_normalized = (nominations_array - nominations_array.mean()) / nominations_array.std()
    wins_normalized = (wins_array - wins_array.mean()) / wins_array.std()
    metascore_normalized = (metascore_array - metascore_array.mean()) / metascore_array.std()
    ratings_normalized = (ratings_array - ratings_array.mean()) / ratings_array.std()

    # Definir pesos para cada métrica
    weight_votes = 0.4
    weight_nominations = 0.3
    weight_wins = 0.15
    weight_metascore = 0.1
    weight_ratings = 0.05

    # Criamos um DataFrame temporário para calcular o score ponderado
    temp_df = user_item_df.copy()
    temp_df['WeightedScore'] = (
        weight_votes * votes_normalized +
        weight_nominations * nominations_normalized +
        weight_wins * wins_normalized +
        weight_metascore * metascore_normalized +
        weight_ratings * ratings_normalized
    )

    # Ordenação com base no score ponderado e ItemId como desempate
    temp_df.sort_values(by=['WeightedScore', 'ItemId'], ascending=[False, True], inplace=True)

    return temp_df[['UserId', 'ItemId']]


def sort_for_known_users(user_item_df, model, preprocessor, user_id_map, item_id_map, item_features):
    # Converte UserId e ItemId para índices internos do LightFM
    user_indices = user_item_df['UserId'].apply(lambda user_id: user_id_map[user_id]).values
    item_indexes = user_item_df['ItemId'].apply(lambda item_id: item_id_map[item_id]).values

    # Previsão do modelo
    user_item_df['Prediction'] = model.predict(user_indices, item_indexes, item_features=item_features)

    # Padroniza a predição (z-score)
    user_item_df['Prediction'] = (user_item_df['Prediction'] - user_item_df['Prediction'].mean()) / user_item_df['Prediction'].std()

    # Mapeia cada ItemId ao seu índice interno
    internal_item_indices = [preprocessor.item_index_mapping[item_id] for item_id in user_item_df['ItemId']]

    # Recupera arrays de votos e indicações
    votes_array = preprocessor.imdb_votes_array[internal_item_indices]
    nominations_array = preprocessor.award_nominations_array[internal_item_indices]

    # Calcula z-score de votos e indicações
    votes_normalized = (votes_array - votes_array.mean()) / votes_array.std()
    nominations_normalized = (nominations_array - nominations_array.mean()) / nominations_array.std()

    # Hiperparâmetros do viés
    bias_alpha = 1  # controle global do viés
    bias_weight_votes = 0.60    # proporção de viés para votes
    bias_weight_nominations = 0.40    # proporção de viés para nominations

    # Ajusta a predição com base no viés combinado
    user_item_df['Prediction'] += bias_alpha * (bias_weight_votes * votes_normalized + bias_weight_nominations * nominations_normalized)

    # Ordena pelo valor da predição (descendente) e usa ItemId como desempate (ascendente)
    user_item_df.sort_values(by=['Prediction', 'ItemId'], ascending=[False, True], inplace=True)
    return user_item_df[['UserId', 'ItemId']]


def predict(model, preprocessor, user_id_map, item_id_map, item_features):
    user_item_groups = {user_id: user_df for user_id, user_df in preprocessor.prediction_targets.groupby('UserId')}
    for user_id in user_item_groups:
        if user_id not in user_id_map:
            user_item_groups[user_id] = sort_for_cold_start(user_item_groups[user_id], preprocessor)
        else:
            user_item_groups[user_id] = sort_for_known_users(user_item_groups[user_id], model, preprocessor, user_id_map, item_id_map, item_features)
    return pd.concat(list(user_item_groups.values()))



if __name__ == "__main__":
    # Inicializa e carrega os dados
    data_preprocessor = ItemDataPreprocessor("ratings.jsonl", "content.jsonl", "targets.csv")
    data_preprocessor.load_data()
    data_preprocessor.preprocess_item_metadata()

    # Extrai todas as features usando o FeatureExtractor
    feature_extractor = FeatureExtractor(data_preprocessor.item_metadata)
    all_possible_item_features, item_feature_mapping = feature_extractor.generate_item_features()

    # Configura o Dataset do LightFM
    dataset = Dataset()
    dataset.fit(
        data_preprocessor.user_ratings['UserId'].tolist(),
        data_preprocessor.user_ratings['ItemId'].tolist(),
    )
    interactions, interaction_weights = dataset.build_interactions(
        (x['UserId'], x['ItemId']) for x in data_preprocessor.user_ratings.to_dict(orient='records')
    )
    dataset.fit_partial(
        items=data_preprocessor.item_metadata['ItemId'].tolist(),
        item_features=all_possible_item_features,
    )
    lightfm_item_features = dataset.build_item_features(item_feature_mapping)

    # Treina o modelo LightFM
    lightfm_model = LightFM(loss='warp', random_state=4)
    lightfm_model.fit(interactions, item_features=lightfm_item_features)

    # Mapeamento interno do LightFM
    user_id_mapping, user_feature_mapping, item_id_mapping, item_feature_mapping = dataset.mapping()

    # Realiza previsões e ordena os itens para os usuários
    prediction_dataframe = predict(
        lightfm_model,
        data_preprocessor,
        user_id_mapping,
        item_id_mapping,
        lightfm_item_features,
    )

    # Imprime as previsões
    print('UserId,ItemId')
    for prediction in prediction_dataframe[['UserId', 'ItemId']].to_dict(orient='records'):
        print(prediction['UserId'] + ',' + prediction['ItemId'])
