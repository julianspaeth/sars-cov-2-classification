import pandas as pd
import numpy as np
import graph_tool as gt
from helper.misc import flatten
from graph_tool import centrality, clustering, topology


def get_neighbors(g: gt.Graph(), nodes: str, proteins: list) -> str:
    nodes = nodes.split(', ')
    neighbors = []
    for node in nodes:
        try:
            node = proteins.index(node.strip())
        except ValueError:
            continue
        neighbors.append(list(g.get_all_neighbors(node)))
    neighbors = flatten(neighbors)
    neighbors = list(map(lambda x: proteins[x], neighbors))
    neighbors = set(neighbors) - set(nodes)
    neighbors = ", ".join(neighbors)
    return neighbors


def extract_graph_features(ppi_G: gt.Graph(), proteins):
    pagerank = pd.DataFrame.from_dict(centrality.pagerank(ppi_G)).rename({0: 'pagerank'}, axis=1)
    vertex_betweenness, _ = gt.centrality.betweenness(ppi_G)
    betweenness = pd.DataFrame.from_dict(vertex_betweenness).rename({0: 'betweenness'}, axis=1)
    closeness = pd.DataFrame.from_dict(centrality.closeness(ppi_G, harmonic=True)).rename({0: 'closeness'}, axis=1)
    _, eigenvector = gt.centrality.eigenvector(ppi_G, max_iter=10000)
    eigenvector = pd.DataFrame.from_dict(eigenvector).rename({0: 'eigenvector'}, axis=1)
    katz = pd.DataFrame.from_dict(centrality.katz(ppi_G)).rename({0: 'katz'}, axis=1)
    local_clustering = pd.DataFrame.from_dict(clustering.local_clustering(ppi_G)).rename({0: 'local_clustering'},
                                                                                         axis=1)
    ppi_df = pd.concat([pagerank, betweenness, closeness, eigenvector, katz, local_clustering], axis=1)
    degrees = [v.out_degree() for v in ppi_G.vertices()]
    ppi_df['degree'] = degrees
    ppi_df = ppi_df.reset_index()
    ppi_df['index'] = ppi_df['index'].apply(lambda x: proteins[x])
    ppi_df = ppi_df.set_index('index')

    return ppi_df


def create_spl_matrix(ppi_G: gt.Graph(), proteins):
    spl_df = pd.DataFrame.from_dict(topology.shortest_distance(ppi_G))
    spl_df = spl_df.replace({2147483647: np.nan})
    spl_df = spl_df.reset_index()
    spl_df['index'] = spl_df['index'].apply(lambda x: proteins[x])
    spl_df = spl_df.set_index('index')
    spl_df.columns = spl_df.index.tolist()

    return spl_df


def create_compound_summary_statistics(drug_df: pd.DataFrame, df: pd.DataFrame, column: str, aim='target',
                                       prefix="") -> pd.DataFrame:
    compound_df = drug_df.copy()
    targets = drug_df[aim].apply(
        lambda x: list(set(x.split(', ')).intersection(set(df.index))) if isinstance(x, str) else x)

    compound_df.loc[:, f'{prefix}{column}_min'] = targets.apply(
        lambda x: df.loc[x, column].min() if isinstance(x, list) else np.nan)
    compound_df.loc[:, f'{prefix}{column}_mean'] = targets.apply(
        lambda x: df.loc[x, column].mean() if isinstance(x, list) else np.nan)
    #compound_df.loc[:, f'{prefix}{column}_median'] = targets.apply(
    #   lambda x: df.loc[x, column].median() if isinstance(x, list) else np.nan)
    compound_df.loc[:, f'{prefix}{column}_max'] = targets.apply(
        lambda x: df.loc[x, column].max() if isinstance(x, list) else np.nan)

    return compound_df


def prepare_compound_features(drugs: pd.DataFrame, ppi_df: pd.DataFrame):
    compounds = drugs.copy()
    for col in ppi_df.columns:
        compounds = create_compound_summary_statistics(compounds, ppi_df, col, aim='Target', prefix='target_')
        compounds = create_compound_summary_statistics(compounds, ppi_df, col, aim='Neighbor', prefix='neighbor_')

    return compounds
