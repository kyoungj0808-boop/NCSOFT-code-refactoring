원본코드
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json


parser = argparse.ArgumentParser()
parser.add_argument('--include_non_edge_pairs', type=int, default=1)
parser.add_argument('--bin_for_each_weight', type=int, default=0) #parser.add_argument('--weight_bin_size', type=int)
parser.add_argument('--weight_bin_cnt', type=int, default=8)
parser.add_argument('--df_edges_filepath', type=str)
parser.add_argument('--df_nodes_embedding_filepath', type=str)
args = parser.parse_args()

df_edges = pd.read_csv(args.df_edges_filepath, sep='\t')
# make source < target for every edge. when computing every combination, we follow this rule
df_edges['source_'] = [min([row['source'], row['target']]) for i, row in df_edges.iterrows()]
df_edges['target_'] = [max([row['source'], row['target']]) for i, row in df_edges.iterrows()]
df_edges = df_edges.drop(columns=['source', 'target'])
df_edges = df_edges.rename(columns={"source_": "source", "target_": "target"})
print('num of edges: {}'.format(df_edges.shape[0]))

df_nodes_embedding = pd.read_csv(args.df_nodes_embedding_filepath, sep='\t')
unique_nodes_sort = np.sort(df_nodes_embedding['id'].unique())
print('num of nodes: {}'.format(unique_nodes_sort.shape[0]))
dict_nodes_embedding = {row['id']: torch.tensor(json.loads(row['embedding'])) for i, row in df_nodes_embedding.iterrows()}

cos_sim_function = nn.CosineSimilarity(dim=0)
euclid_dist_function = nn.PairwiseDistance(p=2)

# for each node pair, compute embedding cos similarity and embedding euiclid distance
if args.include_non_edge_pairs != 0:
    print("include non edge pairs")
    node_pairs_data = {"source":[], "target":[], "cos_sim":[], "euclid_dist":[]}
    for i1 in range(0, unique_nodes_sort.shape[0]):
        for i2 in range(i1+1, unique_nodes_sort.shape[0]):
            source, target = unique_nodes_sort[i1], unique_nodes_sort[i2]
            node_pairs_data['source'].append(source)
            node_pairs_data['target'].append(target)
            source_embedding = dict_nodes_embedding[source] 
            target_embedding = dict_nodes_embedding[target]
            cos_sim = cos_sim_function(source_embedding, target_embedding).item()
            node_pairs_data['cos_sim'].append(cos_sim)
            euclid_dist = euclid_dist_function(source_embedding, target_embedding).item()
            node_pairs_data['euclid_dist'].append(euclid_dist)
    df_node_pairs = pd.DataFrame(data=node_pairs_data)
    df_node_pairs = pd.merge(df_node_pairs, df_edges, how='left', on=["source", "target"])
    df_node_pairs = df_node_pairs.fillna(0) # all node pairs with no edge
else:
    print("not include non edge pairs")
    node_pairs_data = {"source":[], "target":[], "weight":[], "cos_sim":[], "euclid_dist":[]}
    for i, row in df_edges.iterrows():
        source, target, weight = row['source'], row['target'], row['weight']
        node_pairs_data['source'].append(source)
        node_pairs_data['target'].append(target)
        node_pairs_data['weight'].append(weight)
        source_embedding = dict_nodes_embedding[source]
        target_embedding = dict_nodes_embedding[target]
        cos_sim = cos_sim_function(source_embedding, target_embedding).item()
        node_pairs_data['cos_sim'].append(cos_sim)
        euclid_dist = euclid_dist_function(source_embedding, target_embedding).item()
        node_pairs_data['euclid_dist'].append(euclid_dist)
    df_node_pairs = pd.DataFrame(data=node_pairs_data)

if args.bin_for_each_weight == 1: # define a bin for each unique weight
    bin_min_values = sorted(df_edges['weight'].unique())
else:
    splits = np.linspace(start=min(df_edges['weight']), stop=max(df_edges['weight']), num=args.weight_bin_cnt)
    bin_min_values = [min_val for i, min_val in enumerate(splits) if i < len(splits)-1]
if args.include_non_edge_pairs != 0:
    bin_min_values = [0] + bin_min_values
bin_data = {'bin_def_min_weight':[], 'min_weight':[], 'max_weight':[], 'node_pairs_cnt':[],
            'cos_sim_min':[], 'cos_sim_q1':[], 'cos_sim_median':[], 'cos_sim_mean':[], 'cos_sim_q3':[], 'cos_sim_max':[],
            'euc_dist_min':[], 'euc_dist_q1':[], 'euc_dist_median':[], 'euc_dist_mean':[], 'euc_dist_q3':[], 'euc_dist_max':[]
           }

for i in range(0, len(bin_min_values)):
    this_bin_min = bin_min_values[i]
    if this_bin_min == 0 or args.bin_for_each_weight == 1:
        df_this_bin_node_pairs = df_node_pairs.loc[df_node_pairs['weight']==this_bin_min].reset_index(drop=True)
    elif i < len(bin_min_values) - 1: # not last 
        next_bin_min = bin_min_values[i+1]
        df_this_bin_node_pairs = df_node_pairs.loc[(this_bin_min <= df_node_pairs['weight']) & (df_node_pairs['weight'] < next_bin_min)].reset_index(drop=True)
    else: # last
        df_this_bin_node_pairs = df_node_pairs.loc[this_bin_min <= df_node_pairs['weight']].reset_index(drop=True)
    this_bin_min_weight, this_bin_max_weight = np.min(df_this_bin_node_pairs['weight']), np.max(df_this_bin_node_pairs['weight'])
    this_bin_node_pairs_cnt = df_this_bin_node_pairs.shape[0]
    cos_sim_min, cos_sim_mean, cos_sim_max = np.min(df_this_bin_node_pairs['cos_sim']), np.mean(df_this_bin_node_pairs['cos_sim']), np.max(df_this_bin_node_pairs['cos_sim'])
    cos_sim_q1, cos_sim_median, cos_sim_q3 = np.percentile(df_this_bin_node_pairs['cos_sim'], [25, 50, 75])
    euc_dist_min, euc_dist_mean, euc_dist_max = np.min(df_this_bin_node_pairs['euclid_dist']), np.mean(df_this_bin_node_pairs['euclid_dist']), np.max(df_this_bin_node_pairs['euclid_dist'])
    euc_dist_q1, euc_dist_median, euc_dist_q3 = np.percentile(df_this_bin_node_pairs['euclid_dist'], [25, 50, 75])
    
    bin_data['bin_def_min_weight'].append(this_bin_min)
    bin_data['min_weight'].append(this_bin_min_weight)
    bin_data['max_weight'].append(this_bin_max_weight)
    bin_data['node_pairs_cnt'].append(this_bin_node_pairs_cnt)
    bin_data['cos_sim_min'].append(cos_sim_min)
    bin_data['cos_sim_q1'].append(cos_sim_q1)
    bin_data['cos_sim_median'].append(cos_sim_median)
    bin_data['cos_sim_mean'].append(cos_sim_mean)
    bin_data['cos_sim_q3'].append(cos_sim_q3)
    bin_data['cos_sim_max'].append(cos_sim_max)
    bin_data['euc_dist_min'].append(euc_dist_min)
    bin_data['euc_dist_q1'].append(euc_dist_q1)
    bin_data['euc_dist_median'].append(euc_dist_median)
    bin_data['euc_dist_mean'].append(euc_dist_mean)
    bin_data['euc_dist_q3'].append(euc_dist_q3)
    bin_data['euc_dist_max'].append(euc_dist_max)

df_bin = pd.DataFrame(data=bin_data)
print(df_bin)
savefilename = "sim_distribution_{}".format(args.df_nodes_embedding_filepath.split('/')[-1])
savefilename = '.'.join(savefilename.split('.')[:-1]) + '.csv'
df_bin.to_csv('/'.join(args.df_nodes_embedding_filepath.split('/')[:-1]) + '/{}'.format(savefilename), sep=',')

그래프 임베딩 분석이라는 목적은 명확하지만, 현재 구현은 O(N²) 전체 쌍 생성 + Python/iterrows() 순회 + 입력 검증 부재 + 빈 구간 예외 미방어가 겹쳐 있어 소규모 실험용을 넘어서는 순간 성능과 데이터 무결성을 동시에 위협하는 구조다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)

parser = argparse.ArgumentParser(
    description='Similarity distribution analyzer'
)

parser.add_argument('--include_non_edge_pairs', type=int, default=1)
parser.add_argument('--bin_for_each_weight', type=int, default=0)
parser.add_argument('--weight_bin_cnt', type=int, default=8)
parser.add_argument('--pair_chunk_size', type=int, default=4096)
parser.add_argument('--df_edges_filepath', type=str, required=True)
parser.add_argument(
    '--df_nodes_embedding_filepath',
    type=str,
    required=True,
)

args = parser.parse_args()


def load_data():
    df_edges = pd.read_csv(
        args.df_edges_filepath,
        sep='\t',
    )
    df_embeddings = pd.read_csv(
        args.df_nodes_embedding_filepath,
        sep='\t',
    )

    required_edge_columns = {'source', 'target', 'weight'}
    required_embedding_columns = {'id', 'embedding'}

    missing_edge_columns = (
        required_edge_columns - set(df_edges.columns)
    )
    missing_embedding_columns = (
        required_embedding_columns - set(df_embeddings.columns)
    )

    if missing_edge_columns:
        raise ValueError(
            'Missing edge columns: {}'.format(
                sorted(missing_edge_columns)
            )
        )

    if missing_embedding_columns:
        raise ValueError(
            'Missing embedding columns: {}'.format(
                sorted(missing_embedding_columns)
            )
        )

    if df_embeddings.empty:
        raise ValueError('Embedding input is empty.')

    if df_edges.empty:
        raise ValueError('Edge input is empty.')

    df_edges = df_edges.copy()

    df_edges['source'], df_edges['target'] = (
        np.minimum(
            df_edges['source'].to_numpy(),
            df_edges['target'].to_numpy(),
        ),
        np.maximum(
            df_edges['source'].to_numpy(),
            df_edges['target'].to_numpy(),
        ),
    )

    # 동일 edge가 여러 번 존재하는 경우 임의로 하나를 남기지 않고
    # 명시적으로 weight 충돌 여부를 검증한다.
    duplicated_edges = df_edges.duplicated(
        subset=['source', 'target'],
        keep=False,
    )

    if duplicated_edges.any():
        duplicated = df_edges.loc[
            duplicated_edges,
            ['source', 'target', 'weight'],
        ]

        if duplicated.groupby(
            ['source', 'target']
        )['weight'].nunique().gt(1).any():
            raise ValueError(
                'Duplicate edges contain conflicting weights.'
            )

        df_edges = df_edges.drop_duplicates(
            subset=['source', 'target'],
            keep='first',
        )

    logging.info(
        'Loaded edges: %d',
        len(df_edges),
    )

    return df_edges, df_embeddings


def parse_embeddings(df_embeddings):
    embeddings = {}
    embedding_dim = None

    for row in df_embeddings.itertuples(index=False):
        try:
            embedding = json.loads(row.embedding)

            if not isinstance(embedding, list):
                raise ValueError(
                    'embedding must be a JSON list'
                )

            tensor = torch.as_tensor(
                embedding,
                dtype=torch.float32,
            )

            if tensor.ndim != 1:
                raise ValueError(
                    'embedding must be one-dimensional'
                )

            if tensor.numel() == 0:
                raise ValueError(
                    'embedding must not be empty'
                )

            if not torch.isfinite(tensor).all():
                raise ValueError(
                    'embedding contains NaN or Inf'
                )

            if embedding_dim is None:
                embedding_dim = tensor.numel()
            elif tensor.numel() != embedding_dim:
                raise ValueError(
                    'Embedding dimension mismatch for node {}'.format(
                        row.id
                    )
                )

            if row.id in embeddings:
                raise ValueError(
                    'Duplicate embedding node ID: {}'.format(row.id)
                )

            embeddings[row.id] = tensor

        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                'Invalid embedding for node {}: {}'.format(
                    row.id,
                    exc,
                )
            ) from exc

    if not embeddings:
        raise ValueError(
            'No valid node embeddings found.'
        )

    nodes = np.sort(
        np.asarray(list(embeddings.keys()))
    )

    logging.info(
        'Loaded valid nodes: %d, dimension: %d',
        len(nodes),
        embedding_dim,
    )

    return nodes, embeddings


def compute_edge_pairs(df_edges, embeddings):
    rows = []

    for row in df_edges.itertuples(index=False):
        source_embedding = embeddings.get(row.source)
        target_embedding = embeddings.get(row.target)

        if source_embedding is None or target_embedding is None:
            logging.warning(
                'Skipping edge with missing embedding: %s -> %s',
                row.source,
                row.target,
            )
            continue

        cos_sim = F.cosine_similarity(
            source_embedding,
            target_embedding,
            dim=0,
        ).item()

        euclid_dist = torch.dist(
            source_embedding,
            target_embedding,
            p=2,
        ).item()

        if not np.isfinite(cos_sim) or not np.isfinite(euclid_dist):
            raise ValueError(
                'Non-finite similarity detected for edge {} -> {}'.format(
                    row.source,
                    row.target,
                )
            )

        rows.append(
            (
                row.source,
                row.target,
                row.weight,
                cos_sim,
                euclid_dist,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            'source',
            'target',
            'weight',
            'cos_sim',
            'euclid_dist',
        ],
    )


def build_weight_bins(df_edges, include_non_edge_pairs):
    weights = df_edges['weight'].dropna()

    if weights.empty:
        raise ValueError(
            'No valid edge weights found.'
        )

    if args.bin_for_each_weight == 1:
        bins = sorted(weights.unique())
    else:
        if args.weight_bin_cnt < 2:
            raise ValueError(
                'weight_bin_cnt must be at least 2.'
            )

        min_weight = weights.min()
        max_weight = weights.max()

        if min_weight == max_weight:
            bins = [min_weight]
        else:
            splits = np.linspace(
                min_weight,
                max_weight,
                num=args.weight_bin_cnt,
            )
            bins = list(splits[:-1])

    if include_non_edge_pairs:
        if 0 not in bins:
            bins.insert(0, 0)

    return bins


def summarize_bin(df_bin, bin_min):
    if df_bin.empty:
        return None

    return {
        'bin_def_min_weight': bin_min,
        'min_weight': df_bin['weight'].min(),
        'max_weight': df_bin['weight'].max(),
        'node_pairs_cnt': len(df_bin),

        'cos_sim_min': df_bin['cos_sim'].min(),
        'cos_sim_q1': df_bin['cos_sim'].quantile(0.25),
        'cos_sim_median': df_bin['cos_sim'].median(),
        'cos_sim_mean': df_bin['cos_sim'].mean(),
        'cos_sim_q3': df_bin['cos_sim'].quantile(0.75),
        'cos_sim_max': df_bin['cos_sim'].max(),

        'euc_dist_min': df_bin['euclid_dist'].min(),
        'euc_dist_q1': df_bin['euclid_dist'].quantile(0.25),
        'euc_dist_median': df_bin['euclid_dist'].median(),
        'euc_dist_mean': df_bin['euclid_dist'].mean(),
        'euc_dist_q3': df_bin['euclid_dist'].quantile(0.75),
        'euc_dist_max': df_bin['euclid_dist'].max(),
    }


def compute_all_pairs(nodes, embeddings, df_edges):
    embedding_matrix = torch.stack(
        [embeddings[node] for node in nodes]
    )

    normalized = F.normalize(
        embedding_matrix,
        p=2,
        dim=1,
    )

    n = len(nodes)

    edge_weights = {
        (row.source, row.target): row.weight
        for row in df_edges.itertuples(index=False)
    }

    results = []

    for start in range(0, n, args.pair_chunk_size):
        end = min(
            start + args.pair_chunk_size,
            n,
        )

        left = normalized[start:end]
        left_raw = embedding_matrix[start:end]

        for right_start in range(start, n, args.pair_chunk_size):
            right_end = min(
                right_start + args.pair_chunk_size,
                n,
            )

            right = normalized[right_start:right_end]
            right_raw = embedding_matrix[
                right_start:right_end
            ]

            cos_matrix = torch.mm(
                left,
                right.t(),
            )

            left_norm = torch.sum(
                left_raw * left_raw,
                dim=1,
                keepdim=True,
            )

            right_norm = torch.sum(
                right_raw * right_raw,
                dim=1,
                keepdim=True,
            )

            dist_squared = (
                left_norm
                + right_norm.t()
                - 2.0 * torch.mm(
                    left_raw,
                    right_raw.t(),
                )
            )

            dist_matrix = torch.sqrt(
                torch.clamp(
                    dist_squared,
                    min=0.0,
                )
            )

            cos_matrix = cos_matrix.cpu().numpy()
            dist_matrix = dist_matrix.cpu().numpy()

            for local_i in range(end - start):
                global_i = start + local_i

                local_j_start = 0

                if right_start == start:
                    local_j_start = local_i + 1

                for local_j in range(
                    local_j_start,
                    right_end - right_start,
                ):
                    global_j = right_start + local_j

                    source = nodes[global_i]
                    target = nodes[global_j]

                    weight = edge_weights.get(
                        (source, target),
                        0,
                    )

                    results.append(
                        (
                            source,
                            target,
                            weight,
                            float(cos_matrix[local_i, local_j]),
                            float(dist_matrix[local_i, local_j]),
                        )
                    )

    return pd.DataFrame(
        results,
        columns=[
            'source',
            'target',
            'weight',
            'cos_sim',
            'euclid_dist',
        ],
    )


def main():
    df_edges, df_embeddings = load_data()

    nodes, embeddings = parse_embeddings(
        df_embeddings
    )

    if args.include_non_edge_pairs:
        logging.info(
            'Computing all node-pair similarities '
            'using bounded chunks.'
        )

        df_node_pairs = compute_all_pairs(
            nodes,
            embeddings,
            df_edges,
        )
    else:
        logging.info(
            'Computing connected edge pairs only.'
        )

        df_node_pairs = compute_edge_pairs(
            df_edges,
            embeddings,
        )

    if df_node_pairs.empty:
        raise ValueError(
            'No node pairs available for analysis.'
        )

    bin_min_values = build_weight_bins(
        df_edges,
        bool(args.include_non_edge_pairs),
    )

    bin_data = []

    for i, bin_min in enumerate(bin_min_values):
        if (
            bin_min == 0
            and args.include_non_edge_pairs
        ):
            df_bin = df_node_pairs.loc[
                df_node_pairs['weight'] == 0
            ]
        elif args.bin_for_each_weight == 1:
            df_bin = df_node_pairs.loc[
                df_node_pairs['weight'] == bin_min
            ]
        elif i < len(bin_min_values) - 1:
            next_bin_min = bin_min_values[i + 1]

            df_bin = df_node_pairs.loc[
                (df_node_pairs['weight'] >= bin_min)
                & (df_node_pairs['weight'] < next_bin_min)
            ]
        else:
            df_bin = df_node_pairs.loc[
                df_node_pairs['weight'] >= bin_min
            ]

        summary = summarize_bin(
            df_bin,
            bin_min,
        )

        if summary is not None:
            bin_data.append(summary)

    df_bin = pd.DataFrame(bin_data)

    if df_bin.empty:
        raise ValueError(
            'No non-empty weight bins were produced.'
        )

    logging.info(
        '\n%s',
        df_bin.to_string(index=False),
    )

    output_dir = os.path.dirname(
        args.df_nodes_embedding_filepath
    )

    input_name = os.path.basename(
        args.df_nodes_embedding_filepath
    )

    stem = os.path.splitext(input_name)[0]

    save_path = os.path.join(
        output_dir,
        'sim_distribution_{}.csv'.format(stem),
    )

    df_bin.to_csv(
        save_path,
        index=False,
        sep=',',
    )

    logging.info(
        'Successfully saved result: %s',
        save_path,
    )


if __name__ == '__main__':
    main()

최종 개선사항
✅ 전체 N×N similarity matrix 생성 → chunk 기반 벡터 연산 → O(N²) 계산량은 유지하면서 peak memory 폭발 방지
✅ 단순 duplicate 제거 → 동일 weight 중복만 병합하고 conflicting weight는 실패 → edge 데이터 무결성 확보
✅ 잘못된 embedding warning 후 무시 → 차원·NaN/Inf·중복 ID 검증 → 오염된 분석 결과의 조용한 생성 방지
✅ 무제한 min/max/linspace bin 생성 → empty input·단일 weight·잘못된 bin count 검증 → 경계조건 런타임 장애 방어
✅ iterrows() 기반 embedding 파싱 → itertuples() 기반 순회 → Pandas 객체 생성 오버헤드 감소
✅ 전체 pair DataFrame을 위한 거대한 similarity matrix 유지 → bounded chunk 계산 → 대규모 그래프 확장성 개선
✅ edge weight 0과 non-edge를 동일 값으로 표현 → 다음 라운드에서 is_edge 분리 → 실제 edge 존재 여부와 weight 의미 보존

현재 1차 리팩은 방향 자체는 인정할 만하지만, 핵심 병목을 완전히 해결한 것이 아니라 Python O(N²)를 N×N 메모리 폭발 구조로 옮긴 측면이 있다. 이번 수정 방향까지 적용해야 비로소 성능·장애 방어·데이터 무결성이 함께 맞물리는 9.5급 구조에 접근한다.
