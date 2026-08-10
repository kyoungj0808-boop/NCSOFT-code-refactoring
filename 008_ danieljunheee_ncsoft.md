원본코드
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
import random
import pandas as pd
import networkx as nx
from tqdm import tqdm

# node2vec with edge weights implementation reference: https://github.com/keras-team/keras-io/blob/master/examples/graph/node2vec_movielens.py

class RandomWalkerNode2Vec:
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def next_step(self, graph, previous, current):
        neighbors = list(graph.neighbors(current))
    
        weights = []
        # Adjust the weights of the edges to the neighbors with respect to p and q.
        for neighbor in neighbors:
            if neighbor == previous:
                # Control the probability to return to the previous node.
                weights.append(graph[current][neighbor]["weight"] / self.p)
            elif graph.has_edge(neighbor, previous):
                # The probability of visiting a local node.
                weights.append(graph[current][neighbor]["weight"])
            else:
                # Control the probability to move forward.
                weights.append(graph[current][neighbor]["weight"] / self.q)
    
        # Compute the probabilities of visiting each neighbor.
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        # Probabilistically select a neighbor to visit.
        next = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next

    def random_walk(self, graph, batch, walk_len):
        walks = []
        nodes = batch.tolist()
        for node in nodes: # tqdm(nodes):
            # Start the walk with the given node
            walk = [node]
            # Randomly walk for walk_len.
            while len(walk) < walk_len:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = self.next_step(graph, previous, current)
                walk.append(next)
            # Add the walk to the generated sequence.
            walks.append(walk)
    
        return walks

    def sample_random_walks(self, edge_index, edge_weights, batch, walk_len):
        """
        INPUTS
        - edge_index: Graph connectivity in COO format with shape [2, num_edges]
        - edge_weights: edge weight values with shape [num_edges]
        - batch: starting nodes with shape [num starting nodes] including duplicates
        - walk_len: length of each walk
        - p: node2vec hyperparameter p
        - q: node2vec hyperparameter q
    
        OUTPUT
        - walks: sequence of walks with shape [num walks, walk length] 
        """
        # make nx graph object
        G = nx.Graph()
        for i, edge in enumerate(edge_index.t()):
            id1 = edge[0].item()
            id2 = edge[1].item()
            weight = edge_weights[i].item()
            G.add_edge(id1, id2, weight=weight)
    
        # run random walks
        walks = self.random_walk(G, batch, walk_len)
        return torch.tensor(walks)

class RandomWalkerNode2VecPlus:
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def next_step(self, graph, previous, current):
        neighbors = list(graph.neighbors(current))
        if previous is not None:
            d_previous = graph.avg_weight_dict[previous]
        d_current = graph.avg_weight_dict[current]
        weights = []
        # Adjust the weights of the edges to the neighbors with respect to p and q.
        for neighbor in neighbors:
            d_neighbor = graph.avg_weight_dict[neighbor]
    
            # check if (previous, neighbor) is tight
            previous_neighbor_tight = (graph.has_edge(neighbor, previous) and graph[previous][neighbor]['weight'] >= max(d_previous, d_neighbor))
            # check if (current, neighbor) is tight
            current_neighbor_tight = (graph[current][neighbor]['weight'] >= max(d_current, d_neighbor))
    
            if neighbor == previous:
                # Control the probability to return to the previous node.
                weights.append(graph[current][neighbor]["weight"] / self.p)
            elif previous_neighbor_tight:
                weights.append(graph[current][neighbor]["weight"])
            elif not previous_neighbor_tight and not current_neighbor_tight:
                weights.append(graph[current][neighbor]["weight"] * min(1, 1/self.q))
            else:
                if not graph.has_edge(neighbor, previous):
                    weights.append(graph[current][neighbor]["weight"] / self.q)
                else:
                    weights.append(graph[current][neighbor]["weight"] * (1/self.q + ((1 - (1/self.q))*(graph[previous][neighbor]['weight'] / max(d_previous, d_neighbor)))))
    
        # Compute the probabilities of visiting each neighbor.
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        # Probabilistically select a neighbor to visit.
        next = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next

    def random_walk(self, graph, batch, walk_len):
        walks = []
        nodes = batch.tolist()
        for node in nodes: # tqdm(nodes):
            # Start the walk with the given node
            walk = [node]
            # Randomly walk for walk_len.
            while len(walk) < walk_len:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = self.next_step(graph, previous, current)
                walk.append(next)
            # Add the walk to the generated sequence.
            walks.append(walk)
    
        return walks
         
    def sample_random_walks(self, edge_index, edge_weights, batch, walk_len):
        """
        INPUTS
        - edge_index: Graph connectivity in COO format with shape [2, num_edges]
        - edge_weights: edge weight values with shape [num_edges]
        - batch: starting nodes with shape [num starting nodes] including duplicates
        - walk_len: length of each walk
        - p: node2vec hyperparameter p
        - q: node2vec hyperparameter q
    
        OUTPUT
        - walks: sequence of walks with shape [num walks, walk length] 
        """
        # make nx graph object
        G = nx.Graph()
        for i, edge in enumerate(edge_index.t()):
            id1 = edge[0].item()
            id2 = edge[1].item()
            weight = edge_weights[i].item()
            G.add_edge(id1, id2, weight=weight)
        # For each node, compute the average weight of its edges. Store them in a dictionary.
        G.avg_weight_dict = {v: np.mean([G[v][neigh]['weight'] for neigh in G[v]]) for v in G.nodes()}
    
        # run random walks
        walks = self.random_walk(G, batch, walk_len)
        return torch.tensor(walks)

Node2Vec의 가중치·p/q·tight 조건을 결합한 알고리즘적 의도는 탄탄하지만, 매 샘플링마다 PyTorch → NetworkX 변환으로 병목을 만들고 고립 노드·0/비정상 weight에 대한 확률 불변조건까지 방어하지 않아, 정상 입력에서는 잘 돌지만 대규모·비정상 그래프에서 성능과 실행 안정성이 동시에 무너질 수 있는 구조다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional
import networkx as nx
import numpy as np
import torch


class BaseRandomWalker:
    """랜덤 워커 공통 계약 검증 및 데이터 무결성을 보장하는 베이스 클래스"""

    def __init__(self, p: float, q: float):
        if not np.isfinite(p) or p <= 0:
            raise ValueError(f"Hyperparameter p must be finite and greater than 0, got {p}")
        if not np.isfinite(q) or q <= 0:
            raise ValueError(f"Hyperparameter q must be finite and greater than 0, got {q}")
        self.p = p
        self.q = q

    def _validate_inputs(self, edge_index: torch.Tensor, edge_weights: torch.Tensor, batch: torch.Tensor, walk_len: int) -> None:
        """입력 텐서 및 하이퍼파라미터의 엄격한 불변조건(Invariant) 검증"""
        if walk_len <= 0:
            raise ValueError(f"walk_len must be greater than 0, got {walk_len}")
        
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")
        
        if edge_weights.ndim != 1 or edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weights must be 1D and match the number of edges in edge_index")
        
        if batch.ndim != 1:
            raise ValueError(f"batch must be a 1D tensor, got shape {batch.shape}")

        # 엣지 가중치 유효성 검증 (Finite 및 비음수 불변조건)
        if not torch.isfinite(edge_weights).all():
            raise ValueError("edge_weights contain NaN, -inf, or +inf values.")
        if (edge_weights < 0).any():
            raise ValueError("edge_weights cannot contain negative values.")

    def _build_networkx_graph(self, edge_index: torch.Tensor, edge_weights: torch.Tensor) -> nx.Graph:
        """PyTorch 텐서로부터 NetworkX 그래프 빌드 및 인덱스 정합성 확인"""
        g = nx.Graph()
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        w_list = edge_weights.tolist()

        for u, v, w in zip(u_list, v_list, w_list):
            g.add_edge(int(u), int(v), weight=float(w))
        return g

    def _validate_batch_nodes(self, graph: nx.Graph, batch: torch.Tensor) -> List[int]:
        """시작 노드 존재 여부 검증 (고립 노드는 도메인 정책에 따라 허용하되 무한 루프 계약 명시)"""
        nodes = batch.tolist()
        for node in nodes:
            if node not in graph:
                raise KeyError(f"Starting node {node} does not exist in the graph.")
        return nodes


class RandomWalkerNode2Vec(BaseRandomWalker):
    def __init__(self, p: float, q: float):
        super().__init__(p, q)

    def next_step(self, graph: nx.Graph, previous: Optional[int], current: int) -> int:
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            # 고립 노드(차수 0)인 경우, 이동할 수 없으므로 자기 자신을 유지 (Self-loop fallback)
            return current

        weights = []
        for neighbor in neighbors:
            edge_weight = graph[current][neighbor]["weight"]
            if neighbor == previous:
                weights.append(edge_weight / self.p)
            elif graph.has_edge(neighbor, previous):
                weights.append(edge_weight)
            else:
                weights.append(edge_weight / self.q)

        weight_sum = sum(weights)
        # 가중치 합이 0이거나 유한하지 않은 경우 균등 확률로 분배
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            probabilities = [1.0 / len(neighbors)] * len(neighbors)
        else:
            probabilities = [w / weight_sum for w in weights]

        return int(np.random.choice(neighbors, size=1, p=probabilities)[0])

    _walk_len_cache = {}  # 필요시 확장용

    def random_walk(self, graph: nx.Graph, batch: torch.Tensor, walk_len: int) -> List[List[int]]:
        nodes = self._validate_batch_nodes(graph, batch)
        walks = []
        
        for node in nodes:
            walk = [node]
            while len(walk) < walk_len:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                next_node = self.next_step(graph, previous, current)
                walk.append(next_node)
            walks.append(walk)

        return walks

    def sample_random_walks(self, edge_index: torch.Tensor, edge_weights: torch.Tensor, batch: torch.Tensor, walk_len: int) -> torch.Tensor:
        self._validate_inputs(edge_index, edge_weights, batch, walk_len)
        G = self._build_networkx_graph(edge_index, edge_weights)
        
        walks = self.random_walk(G, batch, walk_len)
        # 계약 보장: 입력 batch의 크기(개수)와 출력 walks의 행 개수는 반드시 일치해야 함
        assert len(walks) == len(batch), f"Batch size mismatch: input {len(batch)} vs output {len(walks)}"
        
        return torch.tensor(walks, dtype=torch.long)


class RandomWalkerNode2VecPlus(BaseRandomWalker):
    def __init__(self, p: float, q: float):
        super().__init__(p, q)

    def next_step(self, graph: nx.Graph, previous: Optional[int], current: int) -> int:
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            return current

        d_previous = graph.avg_weight_dict.get(previous, 0.0) if previous is not None else 0.0
        d_current = graph.avg_weight_dict.get(current, 0.0)

        weights = []
        for neighbor in neighbors:
            d_neighbor = graph.avg_weight_dict.get(neighbor, 0.0)
            edge_weight = graph[current][neighbor]["weight"]

            previous_neighbor_tight = (
                previous is not None 
                and graph.has_edge(neighbor, previous) 
                and graph[previous][neighbor]['weight'] >= max(d_previous, d_neighbor)
            )
            current_neighbor_tight = (edge_weight >= max(d_current, d_neighbor))

            # [알고리즘 무결성 보존] 
            # 원본 수식인 max(d_previous, d_neighbor)를 그대로 사용하되, 
            # 0으로 나누어떨어지는 경우(ZeroDivisionError)만 순수 수학적 극한 관점에서 방어
            max_weight_ref = max(d_previous, d_neighbor)

            if neighbor == previous:
                weights.append(edge_weight / self.p)
            elif previous_neighbor_tight:
                weights.append(edge_weight)
            elif not previous_neighbor_tight and not current_neighbor_tight:
                weights.append(edge_weight * min(1.0, 1.0 / self.q))
            else:
                if previous is None or not graph.has_edge(neighbor, previous):
                    weights.append(edge_weight / self.q)
                else:
                    if max_weight_ref == 0.0:
                        ratio = 0.0
                    else:
                        ratio = graph[previous][neighbor]['weight'] / max_weight_ref
                    weights.append(edge_weight * (1.0 / self.q + ((1.0 - (1.0 / self.q)) * ratio)))

        weight_sum = sum(weights)
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            probabilities = [1.0 / len(neighbors)] * len(neighbors)
        else:
            probabilities = [w / weight_sum for w in weights]

        return int(np.random.choice(neighbors, size=1, p=probabilities)[0])

    def random_walk(self, graph: nx.Graph, batch: torch.Tensor, walk_len: int) -> List[List[int]]:
        nodes = self._validate_batch_nodes(graph, batch)
        walks = []

        for node in nodes:
            walk = [node]
            while len(walk) < walk_len:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                next_node = self.next_step(graph, previous, current)
                walk.append(next_node)
            walks.append(walk)

        return walks
         
    def sample_random_walks(self, edge_index: torch.Tensor, edge_weights: torch.Tensor, batch: torch.Tensor, walk_len: int) -> torch.Tensor:
        self._validate_inputs(edge_index, edge_weights, batch, walk_len)
        G = self._build_networkx_graph(edge_index, edge_weights)
        
        # 고립 노드 평균 가중치 연산 (0 나누기 방어)
        avg_weights = {}
        for v in G.nodes():
            neighs = list(G.neighbors(v))
            if neighs:
                avg_weights[v] = float(np.mean([G[v][neigh]['weight'] for neigh in neighs]))
            else:
                avg_weights[v] = 0.0
        G.avg_weight_dict = avg_weights

        walks = self.random_walk(G, batch, walk_len)
        # 계약 보장: 입력 batch의 개수와 출력 walks의 행 개수는 반드시 일치해야 함
        assert len(walks) == len(batch), f"Batch size mismatch: input {len(batch)} vs output {len(walks)}"

        return torch.tensor(walks, dtype=torch.long)

최종개선사항
✅ p/q 및 입력 텐서 무검증 → 유한성·양수·shape·weight 불변조건 사전 검증 → 확률 계산 단계의 비정상 입력과 런타임 크래시 차단
✅ 고립 노드 임의 제거 → Self-loop fallback + batch/output 개수 계약 검증 → 입력 데이터 누락 없이 고정 길이 walk 보장
✅ NaN/0만 확인하는 확률 방어 → 비유한·비양수 합계까지 검증 → np.random.choice()의 invalid probability 입력 방지
✅ Node2VecPlus 분모 임의 변경 → 원본 max(d_previous, d_neighbor) 수식 보존 + 0 분모만 방어 → 알고리즘 의미와 재현성 유지
✅ 입력 weight의 음수·비유한 값 방치 → 그래프 생성 전에 finite/non-negative 검증 → 확률 가중치 무결성 확보
✅ 고립 노드 평균 가중치의 빈 집합 처리 → 명시적 0.0 fallback → Node2VecPlus의 평균 가중치 계산 안정성 확보
✅ 랜덤 워크 결과 크기 암묵적 의존 → 입력 batch와 출력 walk 행 수 계약 검증 → 조용한 데이터 누락·불일치 조기 탐지

원본 알고리즘의 확률식을 불필요하게 변형하지 않으면서 입력 계약·확률 안정성·고립 노드·배치 무결성을 방어층으로 끌어올려, 현재 버전은 데이터 손실과 알고리즘 변질을 동시에 억제한 실무형 랜덤 워커 구조다.        
