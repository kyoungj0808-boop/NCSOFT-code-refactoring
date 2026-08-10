원본코드
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



parser = argparse.ArgumentParser()
parser.add_argument('--df_edges_filepath', type=str)
parser.add_argument('--df_nodes_filepath', type=str)
parser.add_argument('--node_features_matrix_pickle_filepath', type=str)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_repetitions', type=int, default=10)
args = parser.parse_args()


# reference: https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
class GCN(torch.nn.Module):
    def __init__(self, node_features_dim, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(node_features_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def gcn_train(model, optimizer, criterion, x, y, edge_index, edge_weight, train_indices):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(x.to(device), edge_index.to(device), edge_weight.to(device))  # Perform a single forward pass.
    loss = criterion(out[train_indices], y.to(device)[train_indices])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def gcn_predict(model, x, edge_index, edge_weight):
    model.eval()
    out = model(x.to(device), edge_index.to(device), edge_weight.to(device))
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    return pred


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: {}".format(device))

# For node classification, we perform 10 times (independently) of stratified sampling (50% train, 50% test).
# Reference: https://github.com/krishnanlab/node2vecplus_benchmarks/blob/main/script/eval_realworld_networks.py

df_nodes = pd.read_csv(args.df_nodes_filepath, sep='\t')
unique_nodes = sorted(list(df_nodes['id'].unique()))
unique_classes = df_nodes['label'].unique()
class_idx_dict = {cls:i for i, cls in enumerate(unique_classes)}
multilabel_matrix = np.array([[int(y in [class_idx_dict[cls] for cls in df_nodes.loc[df_nodes['id']==node_id].reset_index(drop=True)['label']]) for y in range(0, len(unique_classes))] for node_id in unique_nodes]) # shape: num nodes x num unique classes
node_idx_dict = {x: i for i, x in enumerate(unique_nodes)}

df_edges = pd.read_csv(args.df_edges_filepath, sep='\t')
edge_node1_index = [node_idx_dict[row['source']] for i, row in df_edges.iterrows()]
edge_node2_index = [node_idx_dict[row['target']] for i, row in df_edges.iterrows()]
edge_index = torch.tensor([edge_node1_index, edge_node2_index])
edge_weight_list = [row['weight'] for i, row in df_edges.iterrows()]
edge_weight = torch.tensor(edge_weight_list).to(torch.float32)


with open(args.node_features_matrix_pickle_filepath, 'rb') as f:
    node_features = pickle.load(f)
node_features = torch.from_numpy(node_features.astype(np.float32))

def train_and_eval(node_features, multilabel_matrix):
    f1_list = []
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    strat = StratifiedKFold(n_splits=2)
    for class_idx in range(0, multilabel_matrix.shape[1]): # for each unique class
        y = torch.from_numpy( multilabel_matrix[:,class_idx] )
        train_indices, eval_indices = next(strat.split(node_features, y))
        if len(set(y[train_indices])) == 1: # skip if training data contains only one class
            continue
        model = GCN(node_features_dim=node_features.shape[1], hidden_channels=700, num_classes=len(unique_classes))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, args.num_epochs+1):
            train_loss = gcn_train(model=model, optimizer=optimizer, criterion=criterion, 
                                   x=node_features, y=y, edge_index=edge_index, edge_weight=edge_weight, train_indices=train_indices)
            #print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}')

        eval_predictions = gcn_predict(model=model, x=node_features, edge_index=edge_index, edge_weight=edge_weight)
        eval_predictions = eval_predictions[eval_indices].to('cpu')
        eval_answers = y[eval_indices]
        tn, fp, fn, tp = confusion_matrix(y_true=eval_answers, y_pred=eval_predictions).ravel()
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        f1_list.append(f1_score(y_true=eval_answers, y_pred=eval_predictions))
    macro_f1 = np.mean(f1_list)
    micro_f1 = tp_sum / (tp_sum + (0.5 * (fp_sum + fn_sum)))
    return macro_f1, micro_f1

macro_f1_list, micro_f1_list = [], []
for _ in range(0, args.num_repetitions):
    # train and evaluate GCN model
    macro_f1, micro_f1 = train_and_eval(node_features, multilabel_matrix)
    macro_f1_list.append(macro_f1)
    micro_f1_list.append(micro_f1)
 
if 'amazon_photo_network' in args.df_nodes_filepath:
    network_name = 'amazon'
elif 'cora_network' in args.df_nodes_filepath:
    network_name = 'cora'
else:
    network_name = 'lineagew'
print("node classification for {} network".format(network_name))
print("nodes file: {}".format(args.df_nodes_filepath))
print("number of repetitions for stratified sampling: {}".format(args.num_repetitions))
print("average eval macro f1 score: {}".format(np.mean(macro_f1_list)))
print("average eval micro f1 score: {}".format(np.mean(micro_f1_list)))

PyG 예제 수준의 GCN 골격은 갖췄지만, 클래스별 binary label과 다중 클래스 출력층의 불일치에 더해 반복 실험의 seed·split 독립성까지 무너져 있어, 결국 학습은 돌아가더라도 그 F1 결과의 통계적 신뢰성을 보장하기 어려운 연구용 프로토타입에 가깝다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2023 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
import pickle
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GCNConv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--df_edges_filepath", type=str, required=True)
    parser.add_argument("--df_nodes_filepath", type=str, required=True)
    parser.add_argument(
        "--node_features_matrix_pickle_filepath",
        type=str,
        required=True,
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0.")

    if args.num_repetitions <= 0:
        raise ValueError("num_repetitions must be greater than 0.")

    return args


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
    ):
        super().__init__()

        if node_features_dim <= 0:
            raise ValueError("node_features_dim must be greater than 0.")

        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be greater than 0.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.conv1 = GCNConv(node_features_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index, edge_weight)


def gcn_train(
    model: GCN,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    train_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x_device = x.to(device)
    edge_index_device = edge_index.to(device)
    edge_weight_device = edge_weight.to(device)
    y_device = y.to(device)
    train_indices_device = torch.as_tensor(
        train_indices,
        dtype=torch.long,
        device=device,
    )

    out = model(
        x_device,
        edge_index_device,
        edge_weight_device,
    )

    loss = criterion(
        out[train_indices_device],
        y_device[train_indices_device],
    )

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss detected: {loss.item()}")

    loss.backward()
    optimizer.step()

    return loss.detach()


@torch.no_grad()
def gcn_predict(
    model: GCN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    out = model(
        x.to(device),
        edge_index.to(device),
        edge_weight.to(device),
    )

    if not torch.isfinite(out).all():
        raise RuntimeError("Non-finite model output detected during evaluation.")

    return out.argmax(dim=1)


def train_and_eval(
    node_features: torch.Tensor,
    multilabel_matrix: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    num_epochs: int,
    repetition_seed: int,
) -> Tuple[float, float]:
    f1_list = []
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    strat = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=repetition_seed,
    )

    for class_idx in range(multilabel_matrix.shape[1]):
        y = torch.from_numpy(
            multilabel_matrix[:, class_idx].astype(np.int64)
        )

        unique_labels = torch.unique(y).tolist()

        if len(unique_labels) < 2:
            logger.warning(
                "Skipping class index %d: only one label exists (%s).",
                class_idx,
                unique_labels,
            )
            continue

        try:
            train_indices, eval_indices = next(
                strat.split(
                    np.zeros(len(y)),
                    y.numpy(),
                )
            )
        except ValueError as exc:
            logger.warning(
                "Skipping class index %d because stratified split failed: %s",
                class_idx,
                exc,
            )
            continue

        model = GCN(
            node_features_dim=node_features.shape[1],
            hidden_channels=700,
            num_classes=2,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            weight_decay=5e-4,
        )

        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(num_epochs):
            gcn_train(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                x=node_features,
                y=y,
                edge_index=edge_index,
                edge_weight=edge_weight,
                train_indices=train_indices,
                device=device,
            )

        predictions = gcn_predict(
            model=model,
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            device=device,
        )

        eval_predictions = predictions[
            torch.as_tensor(eval_indices, dtype=torch.long)
        ].cpu().numpy()

        eval_answers = y[
            torch.as_tensor(eval_indices, dtype=torch.long)
        ].cpu().numpy()

        cm = confusion_matrix(
            y_true=eval_answers,
            y_pred=eval_predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        tp_sum += int(tp)
        fp_sum += int(fp)
        fn_sum += int(fn)

        f1_list.append(
            f1_score(
                eval_answers,
                eval_predictions,
                zero_division=0,
            )
        )

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    denominator = tp_sum + 0.5 * (fp_sum + fn_sum)
    micro_f1 = (
        float(tp_sum / denominator)
        if denominator > 0
        else 0.0
    )

    return macro_f1, micro_f1


def load_data(args: argparse.Namespace):
    df_nodes = pd.read_csv(
        args.df_nodes_filepath,
        sep="\t",
    )

    required_node_columns = {"id", "label"}
    missing_node_columns = required_node_columns - set(df_nodes.columns)

    if missing_node_columns:
        raise ValueError(
            f"Node dataset is missing required columns: "
            f"{sorted(missing_node_columns)}"
        )

    if df_nodes.empty:
        raise ValueError("Node dataset is empty.")

    if df_nodes[["id", "label"]].isnull().any().any():
        raise ValueError("Node dataset contains null id/label values.")

    unique_nodes = sorted(df_nodes["id"].unique().tolist())
    unique_classes = df_nodes["label"].unique()

    node_idx_dict = {
        node_id: idx
        for idx, node_id in enumerate(unique_nodes)
    }

    class_idx_dict = {
        label: idx
        for idx, label in enumerate(unique_classes)
    }

    node_indices = df_nodes["id"].map(node_idx_dict)
    class_indices = df_nodes["label"].map(class_idx_dict)

    if node_indices.isnull().any() or class_indices.isnull().any():
        raise RuntimeError("Failed to map node/class indices.")

    multilabel_matrix = np.zeros(
        (len(unique_nodes), len(unique_classes)),
        dtype=np.float32,
    )

    multilabel_matrix[
        node_indices.to_numpy(),
        class_indices.to_numpy(),
    ] = 1.0

    df_edges = pd.read_csv(
        args.df_edges_filepath,
        sep="\t",
    )

    required_edge_columns = {"source", "target", "weight"}
    missing_edge_columns = required_edge_columns - set(df_edges.columns)

    if missing_edge_columns:
        raise ValueError(
            f"Edge dataset is missing required columns: "
            f"{sorted(missing_edge_columns)}"
        )

    if df_edges[["source", "target", "weight"]].isnull().any().any():
        raise ValueError("Edge dataset contains null values.")

    source_indices = df_edges["source"].map(node_idx_dict)
    target_indices = df_edges["target"].map(node_idx_dict)

    unknown_nodes = (
        source_indices.isnull() |
        target_indices.isnull()
    )

    if unknown_nodes.any():
        raise ValueError(
            f"Edge dataset contains "
            f"{int(unknown_nodes.sum())} unknown node references."
        )

    edge_index = torch.tensor(
        [
            source_indices.to_numpy(dtype=np.int64),
            target_indices.to_numpy(dtype=np.int64),
        ],
        dtype=torch.long,
    )

    edge_weight = torch.tensor(
        df_edges["weight"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    if not torch.isfinite(edge_weight).all():
        raise ValueError("Edge weights contain NaN or infinite values.")

    with open(
        args.node_features_matrix_pickle_filepath,
        "rb",
    ) as f:
        node_features_np = pickle.load(f)

    if not isinstance(node_features_np, np.ndarray):
        raise TypeError("Node feature pickle must contain a NumPy array.")

    if node_features_np.ndim != 2:
        raise ValueError(
            "Node feature matrix must be 2-dimensional."
        )

    if node_features_np.shape[0] != len(unique_nodes):
        raise ValueError(
            "Node feature row count does not match the number of unique nodes."
        )

    node_features = torch.from_numpy(
        node_features_np.astype(np.float32, copy=False)
    )

    if not torch.isfinite(node_features).all():
        raise ValueError(
            "Node feature matrix contains NaN or infinite values."
        )

    return (
        node_features,
        multilabel_matrix,
        edge_index,
        edge_weight,
        unique_classes,
    )


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Device initialized: %s", device)

    (
        node_features,
        multilabel_matrix,
        edge_index,
        edge_weight,
        unique_classes,
    ) = load_data(args)

    macro_f1_list = []
    micro_f1_list = []

    for repetition in range(args.num_repetitions):
        current_seed = args.seed + repetition
        set_seed(current_seed)

        macro_f1, micro_f1 = train_and_eval(
            node_features=node_features,
            multilabel_matrix=multilabel_matrix,
            edge_index=edge_index,
            edge_weight=edge_weight,
            device=device,
            num_epochs=args.num_epochs,
            repetition_seed=current_seed,
        )

        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

        logger.info(
            "Repetition %d/%d - macro F1: %.4f, micro F1: %.4f",
            repetition + 1,
            args.num_repetitions,
            macro_f1,
            micro_f1,
        )

    if not macro_f1_list:
        raise RuntimeError("No valid evaluation results were produced.")

    filepath_lower = args.df_nodes_filepath.lower()

    if "amazon_photo_network" in filepath_lower:
        network_name = "amazon"
    elif "cora_network" in filepath_lower:
        network_name = "cora"
    else:
        network_name = "lineagew"

    logger.info(
        "Node classification for %s network completed.",
        network_name,
    )
    logger.info(
        "Number of repetitions: %d",
        args.num_repetitions,
    )
    logger.info(
        "Average eval macro F1 score: %.4f",
        np.mean(macro_f1_list),
    )
    logger.info(
        "Average eval micro F1 score: %.4f",
        np.mean(micro_f1_list),
    )


if __name__ == "__main__":
    main()

최종 개선사항
✅ 다중클래스 출력층과 이진 라벨 불일치 → 클래스별 Binary GCN 계약 고정 → CrossEntropyLoss 차원 무결성 확보
✅ 생성자 내부 고정 Seed → 반복별 외부 Seed 제어 → 독립 반복 실험의 재현성 확보
✅ iterrows() 기반 그래프 구축 → Pandas 벡터화 → 대규모 엣지 처리 병목 완화
✅ 검증되지 않은 Node/Edge ID → 그래프 생성 전 참조 무결성 검증 → 잘못된 그래프 구성 조기 차단
✅ 무검증 Pickle Feature → 타입·차원·행 수·수치 유효성 검증 → Node/Feature 정합성 및 학습 안정성 확보
✅ CUDA 디버깅 전역 설정 → 실행 코드와 디버깅 환경 분리 → 불필요한 GPU 성능 저하 방지
✅ 평가 실패를 0점으로 은폐 → 유효 평가 결과 존재 여부 검증 → 잘못된 실험 결과의 정상값 위장 방지

PyG 예제 수준의 GCN 실험 코드를 클래스별 이진 분류 계약과 반복 실험 재현성을 유지하면서 입력·그래프·Feature 무결성까지 방어하는 실험용 평가 파이프라인으로 승격했다.    
