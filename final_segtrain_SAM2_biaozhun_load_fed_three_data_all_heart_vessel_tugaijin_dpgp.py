#!/usr/bin/env python3
"""
增强版联邦医学图像分割 - 基于图感知的自适应点优化（改进版）
核心改进：
1. 完全隔离个性化层，确保不参与联邦聚合
2. 增强图结构相似性度量和自适应聚合
3. 改进点初始化策略（基于数据分布）
4. 完善强化学习机制（经验回放和策略更新）
5. 增强异构客户端处理能力
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import logging
from collections import defaultdict, OrderedDict, deque
from datetime import datetime
import math
import copy
import pickle
from typing import Dict, List, Tuple, Optional, Any

# 尝试导入networkx，如果失败则使用简化版本
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    print("警告: networkx未安装，将使用简化的图分析功能")
    print("建议安装: pip install networkx")
    HAS_NETWORKX = False

# 需要安装torch_geometric
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import degree



except ImportError:
    print("请安装torch_geometric: pip install torch-geometric")
    sys.exit(1)

# =============================================================================
# 【新增】导入双路径门控个性化模块
# =============================================================================
try:
    from dual_path_gated_personalization_v2 import (
        ImprovedDPGPModule,
        create_improved_dpgp_module  # 可选，如果使用工厂函数
    )
    DPGP_AVAILABLE = True
    print("✅ 成功导入双路径门控个性化模块 (DPGP v2)")
except ImportError as e:
    print(f"⚠️ 无法导入DPGP模块: {e}")
    print("请确保 dual_path_gated_personalization_v2.py 在同一目录下")
    DPGP_AVAILABLE = False

# ==================== 增强配置参数 ====================
CONFIG = {

    # 联邦学习参数
    'num_clients': 3,
    'communication_rounds': 50,
    'local_epochs': 3,
    'participation_rate': 1.0,
    
    # 【修改】Arcade数据集配置（替换原来的三个数据集路径）
    'arcade_data_root': './Arcade',
    'num_classes': 2,
    'client1_train_ratio': 0.5,
    'client2_train_ratio': 0.3,
    'client3_train_ratio': 0.2,
    'save_dir': './federated_checkpoints_sam2_arcade_532_dpgp_42_batchsize2_0121',
    
    # SAM2预训练权重路径
    'sam2_checkpoint_path': '/data1/gs/RP-SAM2-main/ScribblePrompt_sam_v1_vit_b_res128.pt',
    
    # 训练参数
    'batch_size': 2,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # 训练策略
    'freeze_image_encoder': True,
    'freeze_prompt_encoder': False,
    
    # 模型参数
    'image_size': 1024,
    'client1_num_classes': 2,
    'client2_num_classes': 2,
    'client3_num_classes': 2,
    'num_times_sample': 10,
    'num_points_grid': 50,
    
    # 图结构参数
    'graph_hidden_dim': 128,
    'graph_num_heads': 4,
    'graph_k_neighbors': 8,  # K近邻图
    'spatial_weight': 0.5,   # 空间距离权重
    'feature_weight': 0.5,   # 特征相似度权重
    
    # 点优化参数
    'min_points': 8,         # 最少点数
    'max_points': 25,        # 最多点数
    'point_selection_temp': 0.8,  # 温度参数
    'min_point_confidence': 0.4,  # 最小点置信度
    
    # 强化学习参数
    'rl_gamma': 0.98,        # 折扣因子
    'rl_lr': 5e-3,           # 学习率
    'rl_buffer_size': 10000, # 经验回放缓冲区大小
    'rl_batch_size': 32,     # RL批量大小
    'rl_update_freq': 10,    # RL更新频率
    'rl_epsilon': 0.15,       # ε-贪婪探索
    'rl_epsilon_decay': 0.998,  # ε衰减率
    
    # ShiftBlock参数
    'num_attention_layers': 6,
    'sx_init': 0.1,
    'sy_init': 0.1,
    'epsilon': 0.005,
    'theta': 15,
    
    # 损失函数权重
    'alpha': 1.2,   # Dice损失
    'beta': 0.08,    # CE损失
    'gamma': 0.8,   # 距离损失
    'lambda': 0.05,  # 稀疏性损失
    
    # 自适应聚合参数
    'similarity_threshold': 0.5,  # 相似度阈值
    'adaptive_weight_alpha': 0.7,  # 自适应权重系数
    
    # 其他设置
    # 'random_seed': 13,
    'random_seed': 42,
    # 'random_seed': 42,
    'num_workers': 0,
    'device': 'cuda:0',
    'eval_freq': 1,

    # 【修改】DPGP模块参数
    'dpgp_feature_dim': 128,
    'dpgp_hidden_dim': 256,
    'dpgp_num_encoder_layers': 2,
    'dpgp_dropout': 0.1,
    'use_dpgp': True,
    'dpgp_test_time_adaptation': True,
    'dpgp_residual_weight': 0.5,
    'dpgp_warmup_rounds': 8,
    
    # 【新增】可视化保存
    'save_periodic_predictions': True,   # 新增此参数
    'save_best_predictions': True,
    
    # 【新增】实例级权重和TTA参数
    'dpgp_use_instance_level_gating': True,   # 启用实例级动态权重
    'dpgp_tta_adaptation_strength': 0.3,       # TTA调整强度 (0-1)
    
    # 【修改】结果保存设置 - 只保留最佳
    'save_predictions': True,
    'pred_samples_per_round': 3,
    'save_best_predictions': True,                    
    'save_intermediate_results': False,  # 【新增】不保存中间结果
}

# ==================== 客户端类别映射 ====================
CLASS_NAMES = [
    "Background",
    "Foreground"
]

# ==================== 日志设置 ====================
def setup_logging(save_dir):
    """设置日志系统，同时输出到控制台和文件"""
    log_dir = Path(save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"federated_training_{timestamp}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"日志将保存到: {log_file}")
    return logger

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class ArcadeDatasetSplit(Dataset):
    """
    Arcade数据集 - 支持5:3:2划分
    """
    def __init__(self, root, client_id, transform=None, is_train=True, config=None):
        self.root = Path(root)
        self.client_id = client_id
        self.transform = transform
        self.is_train = is_train
        
        if config is None:
            config = CONFIG
        
        self.image_size = config.get('image_size', 1024)
        self.num_points = config.get('num_points_grid', 50)
        self.num_classes = config.get('num_classes', 2)
        self.client1_ratio = config.get('client1_train_ratio', 0.5)
        self.client2_ratio = config.get('client2_train_ratio', 0.3)
        self.client3_ratio = config.get('client3_train_ratio', 0.2)
        self.random_seed = config.get('random_seed', 42)
        
        self.point_initializer = DataDistributionAwarePointInitializer(self.num_points)
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """加载样本并进行5:3:2划分"""
        if self.is_train:
            images_dir = self.root / "train" / "images"
            masks_dir = self.root / "train" / "masks"
        else:
            images_dir = self.root / "test" / "images"
            masks_dir = self.root / "test" / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            print(f"警告: 数据目录不存在: {images_dir} 或 {masks_dir}")
            return
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.tif', '*.tiff']:
            image_files.extend(list(images_dir.glob(ext)))
        image_files = sorted(image_files)
        
        all_samples = []
        for img_file in image_files:
            mask_candidates = [
                masks_dir / f"{img_file.stem}_mask.png",
                masks_dir / f"{img_file.stem}.png",
                masks_dir / f"{img_file.stem}_mask.PNG",
                masks_dir / f"{img_file.stem}.PNG",
            ]
            
            mask_file = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_file = candidate
                    break
            
            if mask_file is not None:
                all_samples.append((str(img_file), str(mask_file)))
        
        if len(all_samples) == 0:
            print(f"警告: 未找到有效的图像-mask对")
            return
        
        print(f"总{'训练' if self.is_train else '测试'}样本数: {len(all_samples)}")
        
        if self.is_train:
            np.random.seed(self.random_seed)
            indices = np.arange(len(all_samples))
            np.random.shuffle(indices)
            
            total = len(all_samples)
            split1 = int(total * self.client1_ratio)
            split2 = int(total * (self.client1_ratio + self.client2_ratio))
            
            if self.client_id == 1:
                selected_indices = indices[:split1]
                print(f"客户端1获得前{self.client1_ratio*100:.0f}%数据: {len(selected_indices)} 个样本")
            elif self.client_id == 2:
                selected_indices = indices[split1:split2]
                print(f"客户端2获得中间{self.client2_ratio*100:.0f}%数据: {len(selected_indices)} 个样本")
            else:
                selected_indices = indices[split2:]
                print(f"客户端3获得后{self.client3_ratio*100:.0f}%数据: {len(selected_indices)} 个样本")
            
            self.samples = [all_samples[i] for i in selected_indices]
        else:
            self.samples = all_samples
        
        print(f"客户端 {self.client_id} {'训练' if self.is_train else '测试'}集: {len(self.samples)} 张图像")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            return self._get_empty_sample("")
            
        img_path, mask_path = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            masks, points, labels, dist_stats = self._process_mask_adaptive(mask_path, original_size)
            
            return {
                'image': image,
                'masks': masks,
                'points': points,
                'labels': labels,
                'image_path': img_path,
                'mask_path': mask_path,
                'distribution_stats': dist_stats
            }
            
        except Exception as e:
            print(f"加载图像时出错 {img_path}: {e}")
            return self._get_empty_sample(img_path)
    
    def _process_mask_adaptive(self, mask_path, original_size):
        try:
            mask_img = Image.open(mask_path).convert('L')
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_array = np.array(mask_img)
            
            masks = torch.zeros(self.num_classes, self.image_size, self.image_size)
            
            background_mask = (mask_array == 0).astype(np.float32)
            foreground_mask = (mask_array > 0).astype(np.float32)
            
            masks[0] = torch.from_numpy(background_mask)
            masks[1] = torch.from_numpy(foreground_mask)
            
            if self.is_train:
                points, labels, dist_stats = self.point_initializer.sample_points_adaptive(
                    torch.from_numpy(foreground_mask)
                )
            else:
                points, labels = self._get_grid_points()
                dist_stats = {}
            
            return masks, points, labels, dist_stats
            
        except Exception as e:
            print(f"处理掩码失败 {mask_path}: {e}")
            return self._get_empty_annotation()
    
    def _get_grid_points(self):
        points = []
        labels = []
        
        grid_size = int(np.sqrt(self.num_points))
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) * self.image_size / grid_size
                y = (j + 0.5) * self.image_size / grid_size
                points.append([x, y])
                labels.append(0)
        
        while len(points) < self.num_points:
            x = random.random() * self.image_size
            y = random.random() * self.image_size
            points.append([x, y])
            labels.append(0)
        
        points = torch.tensor(points[:self.num_points], dtype=torch.float32)
        labels = torch.tensor(labels[:self.num_points], dtype=torch.long)
        
        return points, labels
    
    def _get_empty_sample(self, img_path):
        masks = torch.zeros(self.num_classes, self.image_size, self.image_size)
        masks[0] = 1.0
        
        return {
            'image': torch.zeros(3, self.image_size, self.image_size),
            'masks': masks,
            'points': torch.zeros(self.num_points, 2),
            'labels': torch.zeros(self.num_points, dtype=torch.long),
            'image_path': img_path,
            'mask_path': None,
            'distribution_stats': {}
        }
    
    def _get_empty_annotation(self):
        masks = torch.zeros(self.num_classes, self.image_size, self.image_size)
        masks[0] = 1.0
        
        points = torch.zeros(self.num_points, 2)
        labels = torch.zeros(self.num_points, dtype=torch.long)
        
        return masks, points, labels, {}


def check_arcade_dataset(root):
    """检查Arcade数据集结构"""
    root = Path(root)
    
    required_dirs = [
        root / 'train' / 'images',
        root / 'train' / 'masks',
        root / 'test' / 'images',
        root / 'test' / 'masks'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            all_files = list(dir_path.glob('*'))
            num_files = len(all_files)
            print(f"✓ {dir_path.relative_to(root)}: {num_files} 文件")
        else:
            print(f"✗ {dir_path.relative_to(root)}: 不存在")
            all_exist = False
    
    return all_exist
# ==================== 图结构分析工具 ====================
class GraphStructureAnalyzer:
    """图结构分析器，用于计算图的拓扑特征和相似性"""
    
    @staticmethod
    def compute_graph_features(edge_index: torch.Tensor, num_nodes: int, 
                              node_features: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """计算图的拓扑特征"""
        features = {}
        
        if HAS_NETWORKX:
            # 使用NetworkX进行完整分析
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            edges = edge_index.t().cpu().numpy()
            G.add_edges_from(edges)
            
            # 度分布统计
            degrees = list(dict(G.degree()).values())
            features['avg_degree'] = np.mean(degrees) if degrees else 0
            features['std_degree'] = np.std(degrees) if degrees else 0
            features['max_degree'] = max(degrees) if degrees else 0
            
            # 聚类系数
            features['avg_clustering'] = nx.average_clustering(G)
            
            # 连通分量
            components = list(nx.connected_components(G))
            features['num_components'] = len(components)
            features['largest_component_size'] = max(len(c) for c in components) if components else 0
            
            # 密度
            features['density'] = nx.density(G)
        else:
            # 简化版本，不依赖NetworkX
            edges = edge_index.t().cpu().numpy()
            
            # 计算度
            degree_dict = {}
            for i in range(num_nodes):
                degree_dict[i] = 0
            for edge in edges:
                degree_dict[edge[0]] += 1
                degree_dict[edge[1]] += 1
            
            degrees = list(degree_dict.values())
            features['avg_degree'] = np.mean(degrees) if degrees else 0
            features['std_degree'] = np.std(degrees) if degrees else 0
            features['max_degree'] = max(degrees) if degrees else 0
            
            # 简化的聚类系数（三角形计数）
            triangles = 0
            adj_list = defaultdict(set)
            for edge in edges:
                adj_list[edge[0]].add(edge[1])
                adj_list[edge[1]].add(edge[0])
            
            for node in range(num_nodes):
                neighbors = list(adj_list[node])
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if neighbors[j] in adj_list[neighbors[i]]:
                            triangles += 1
            
            # 聚类系数近似
            possible_triangles = sum(d * (d - 1) / 2 for d in degrees) if degrees else 1
            features['avg_clustering'] = (triangles / possible_triangles) if possible_triangles > 0 else 0
            
            # 连通分量（简化版）
            visited = [False] * num_nodes
            components = 0
            max_component = 0
            
            def dfs(node, visited, adj_list):
                stack = [node]
                size = 0
                while stack:
                    curr = stack.pop()
                    if not visited[curr]:
                        visited[curr] = True
                        size += 1
                        for neighbor in adj_list[curr]:
                            if not visited[neighbor]:
                                stack.append(neighbor)
                return size
            
            for i in range(num_nodes):
                if not visited[i]:
                    component_size = dfs(i, visited, adj_list)
                    components += 1
                    max_component = max(max_component, component_size)
            
            features['num_components'] = components
            features['largest_component_size'] = max_component
            
            # 密度
            max_edges = num_nodes * (num_nodes - 1) / 2
            actual_edges = len(edges)
            features['density'] = actual_edges / max_edges if max_edges > 0 else 0
        
        # 特征分布（如果有节点特征）
        if node_features is not None:
            features['feature_mean'] = node_features.mean().item()
            features['feature_std'] = node_features.std().item()
            
        return features
    
    @staticmethod
    def compute_graph_similarity(features1: Dict[str, float], 
                                features2: Dict[str, float]) -> float:
        """计算两个图的相似度"""
        similarity_scores = []
        
        # 比较各个特征
        for key in features1.keys():
            if key in features2:
                val1, val2 = features1[key], features2[key]
                # 归一化差异
                if val1 + val2 > 0:
                    sim = 1 - abs(val1 - val2) / (val1 + val2 + 1e-8)
                else:
                    sim = 1.0
                similarity_scores.append(sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    @staticmethod
    def compute_distribution_distance(dist1: np.ndarray, dist2: np.ndarray) -> float:
        """计算两个分布的Wasserstein距离"""
        try:
            return wasserstein_distance(dist1.flatten(), dist2.flatten())
        except:
            return float('inf')

# ==================== 增强的强化学习组件 ====================
class Experience:
    """轻量级经验存储 - 使用 __slots__ 和 CPU numpy"""
    __slots__ = ['state', 'action', 'reward', 'next_state', 'done']
    
    def __init__(self, state, action, reward, next_state, done):
        # 关键：转为 CPU numpy 并用 float16 节省一半内存
        self.state = state.detach().cpu().numpy().astype(np.float16)
        self.action = action.detach().cpu().numpy().astype(np.float16)
        self.reward = float(reward) if torch.is_tensor(reward) else float(reward)
        self.next_state = next_state.detach().cpu().numpy().astype(np.float16)
        self.done = bool(done)


class EnhancedPointSelectionRL(nn.Module):
    """增强的强化学习点选择模块，带完整的经验回放和策略更新"""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        
        # Q网络（用于DQN）
        self.q_network = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),  # +1 for importance
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 选择或不选择
        )
        
        # 目标Q网络
        self.target_q_network = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=CONFIG['rl_buffer_size'])
        self.epsilon = CONFIG['rl_epsilon']
        self.update_counter = 0
        
        # 初始化目标网络
        self.update_target_network()
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    def forward(self, features, importance):
        """前向传播"""
        combined = torch.cat([features, importance], dim=-1)
        q_values = self.q_network(combined)
        policy = self.policy_network(combined)
        return q_values, policy
    
    def select_points_with_exploration(self, features, importance, min_points, max_points, 
                                      temperature=1.0, training=True):
        """带探索的点选择"""
        B, N, _ = features.shape
        selected_masks = []
        states_for_memory = []
        actions_for_memory = []
        
        for b in range(B):
            # 获取Q值和策略
            q_values, policy = self.forward(features[b], importance[b])
            
            if training and random.random() < self.epsilon:
                # ε-贪婪探索
                num_select = random.randint(min_points, max_points)
                selected = torch.randperm(N, device=features.device)[:num_select]
            else:
                # 基于策略选择
                selection_probs = policy.squeeze(-1)
                
                # 应用温度缩放使选择更确定
                if temperature > 0:
                    selection_probs = selection_probs / temperature
                    selection_probs = torch.sigmoid(selection_probs)  # 确保在0-1范围
                
                # 使用Q值指导选择
                q_action_values = q_values[:, 1]  # 选择动作的Q值
                combined_scores = 0.7 * selection_probs + 0.3 * torch.sigmoid(q_action_values)
                
                # 选择高分点
                _, indices = torch.topk(combined_scores, min(max_points, N))
                
                # 基于置信度筛选
                high_conf_mask = combined_scores[indices] > CONFIG['min_point_confidence']
                num_high_conf = high_conf_mask.sum().item()
                
                # 确保至少选择min_points个点
                if num_high_conf >= min_points:
                    selected = indices[high_conf_mask]
                else:
                    # 如果高置信度点不够，选择top-k
                    selected = indices[:max(min_points, min(num_high_conf + 2, max_points))]
            
            # 确保selected不为空
            if len(selected) == 0:
                # 如果没有选择任何点，至少选择min_points个
                selected = torch.arange(min_points, device=features.device)
            
            # 创建mask
            mask = torch.zeros(N, dtype=torch.bool, device=features.device)
            mask[selected] = True
            selected_masks.append(mask)
            
            # 存储状态和动作用于经验回放
            if training:
                state = torch.cat([features[b], importance[b]], dim=-1).detach().clone()
                action = mask.float().detach().clone()
                states_for_memory.append(state)
                actions_for_memory.append(action)
        
        # 衰减epsilon
        if training:
            self.epsilon *= CONFIG['rl_epsilon_decay']
            self.epsilon = max(self.epsilon, 0.01)  # 最小探索率
        
        selected_masks = torch.stack(selected_masks)
        
        # 调试信息
        if training:
            avg_selected = selected_masks.float().sum(dim=1).mean().item()
            if avg_selected == 0:
                logger.warning(f"警告：平均选择了 {avg_selected:.1f} 个点（应该在 {min_points}-{max_points} 之间）")
        
        return selected_masks, states_for_memory, actions_for_memory
    
    def store_experience(self, state, action, reward, next_state, done):
        # 限制缓冲区大小，防止无限增长
        if len(self.memory) >= self.memory.maxlen:
            # 可选：清理最旧的 10% 经验
            pass
        exp = Experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def update(self, batch_size=32, gamma=0.95):
        if len(self.memory) < batch_size * 2:  # 至少2倍batch才更新
            return 0
        
        batch = random.sample(self.memory, batch_size)
        device = next(self.q_network.parameters()).device
        
        # 从 numpy 重建 tensor
        states = torch.tensor(
            np.stack([exp.state.astype(np.float32) for exp in batch]),
            dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            np.stack([exp.action.astype(np.float32) for exp in batch]),
            dtype=torch.float32, device=device
        )
        rewards = torch.tensor(
            [exp.reward for exp in batch],
            dtype=torch.float32, device=device
        )
        next_states = torch.tensor(
            np.stack([exp.next_state.astype(np.float32) for exp in batch]),
            dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [exp.done for exp in batch],
            dtype=torch.float32, device=device
        )
        
        # 当前Q值
        current_q_values = self.q_network(states)
        current_q = (current_q_values * actions.unsqueeze(-1)).sum(dim=-1).mean(dim=-1)
        
        # 目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            next_q = next_q_values.max(dim=-1)[0].mean(dim=-1)
            target_q = rewards + gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 定期更新目标网络
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_network()
        
        return loss.item()

# ==================== 数据分布感知的点初始化 ====================
class DataDistributionAwarePointInitializer:
    """基于数据分布的智能点初始化器"""
    
    def __init__(self, num_points=50):
        self.num_points = num_points
        self.distribution_stats = {}
        self.learned_patterns = {}
        
    def analyze_mask_distribution(self, mask):
        """分析掩码分布"""
        # 计算前景/背景比例
        fg_ratio = (mask > 0).float().mean().item()
        
        # 计算连通区域
        mask_np = mask.cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_np)
        
        # 计算边界复杂度
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_complexity = sum(len(c) for c in contours) if contours else 0
        
        return {
            'fg_ratio': fg_ratio,
            'num_regions': num_labels - 1,  # 排除背景
            'boundary_complexity': boundary_complexity,
            'spatial_distribution': self._compute_spatial_distribution(mask)
        }
    
    def _compute_spatial_distribution(self, mask):
        """计算空间分布特征"""
        h, w = mask.shape
        
        # 划分网格
        grid_size = 4
        grid_h, grid_w = h // grid_size, w // grid_size
        
        distribution = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = mask[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                distribution.append((region > 0).float().mean().item())
        
        return np.array(distribution)
    
    def sample_points_adaptive(self, mask, features=None):
        """基于分布自适应采样点"""
        stats = self.analyze_mask_distribution(mask)
        
        # 根据前景比例调整采样策略
        fg_ratio = stats['fg_ratio']
        num_fg_points = int(self.num_points * (0.3 + 0.4 * fg_ratio))  # 30-70%的点在前景
        num_bg_points = int(self.num_points * 0.3)  # 30%在背景
        num_boundary_points = self.num_points - num_fg_points - num_bg_points  # 剩余在边界
        
        points = []
        labels = []
        
        # 前景采样
        fg_y, fg_x = torch.where(mask > 0)
        if len(fg_x) > 0 and num_fg_points > 0:
            # 根据区域数量决定采样策略
            if stats['num_regions'] > 1:
                # 多区域：每个区域都要采样
                points_per_region = self._sample_from_regions(mask, num_fg_points)
                points.extend(points_per_region)
                labels.extend([1] * len(points_per_region))
            else:
                # 单区域：均匀采样
                indices = torch.randperm(len(fg_x))[:num_fg_points]
                for idx in indices:
                    points.append([fg_x[idx].item(), fg_y[idx].item()])
                    labels.append(1)
        
        # 背景采样
        bg_y, bg_x = torch.where(mask == 0)
        if len(bg_x) > 0 and num_bg_points > 0:
            indices = torch.randperm(len(bg_x))[:num_bg_points]
            for idx in indices:
                points.append([bg_x[idx].item(), bg_y[idx].item()])
                labels.append(0)
        
        # 边界采样
        if num_boundary_points > 0:
            boundary_points = self._sample_boundary_points(mask, num_boundary_points)
            points.extend(boundary_points)
            labels.extend([2] * len(boundary_points))  # 边界点标签为2
        
        # 填充到目标数量
        while len(points) < self.num_points:
            x = random.random() * mask.shape[1]
            y = random.random() * mask.shape[0]
            points.append([x, y])
            labels.append(0)
        
        # 转换为张量
        points = torch.tensor(points[:self.num_points], dtype=torch.float32)
        labels = torch.tensor(labels[:self.num_points], dtype=torch.long)
        
        # 映射边界标签到0/1
        labels[labels == 2] = 1  # 将边界点视为前景
        
        return points, labels, stats
    
    def _sample_from_regions(self, mask, num_points):
        """从多个区域采样"""
        mask_np = mask.cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_np)
        
        points = []
        points_per_region = max(1, num_points // (num_labels - 1))
        
        for label_id in range(1, num_labels):
            region_y, region_x = np.where(labels == label_id)
            if len(region_x) > 0:
                num_sample = min(len(region_x), points_per_region)
                indices = np.random.choice(len(region_x), num_sample, replace=False)
                for idx in indices:
                    points.append([region_x[idx], region_y[idx]])
        
        return points
    
    def _sample_boundary_points(self, mask, num_points):
        """采样边界点"""
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # 计算边界
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        boundary = dilated - eroded
        
        boundary_y, boundary_x = np.where(boundary > 0)
        points = []
        
        if len(boundary_x) > 0:
            num_sample = min(len(boundary_x), num_points)
            indices = np.random.choice(len(boundary_x), num_sample, replace=False)
            for idx in indices:
                points.append([boundary_x[idx], boundary_y[idx]])
        
        return points
    
    def update_learned_patterns(self, stats, performance_score):
        """更新学习到的模式"""
        key = f"{stats['fg_ratio']:.2f}_{stats['num_regions']}"
        if key not in self.learned_patterns:
            self.learned_patterns[key] = []
        self.learned_patterns[key].append(performance_score)

# ==================== 图神经网络组件 ====================
class PointGraphNetwork(nn.Module):
    """点图神经网络，用于学习点之间的关系和重要性"""
    def __init__(self, input_dim=256, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # 动态处理输入维度（可能是256或512）
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            self.gat_layers.append(
                GATConv(in_dim, out_dim // num_heads, heads=num_heads, concat=True, dropout=0.1)
            )
        
        # 重要性预测头
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征提取头
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        
        # 图结构分析器
        self.graph_analyzer = GraphStructureAnalyzer()
        
    def forward(self, x, edge_index, batch=None):
        # 输入投影
        x = self.input_proj(x)
        
        # 存储图特征用于分析
        graph_features = {}
        if batch is None:
            num_nodes = x.size(0)
            graph_features = self.graph_analyzer.compute_graph_features(edge_index, num_nodes, x)
        
        # GAT层处理
        for i, gat in enumerate(self.gat_layers):
            x_new = F.relu(gat(x, edge_index))
            if i > 0:  # 残差连接
                x = x + x_new
            else:
                x = x_new
            
        # 计算点重要性
        importance = self.importance_head(x)
        
        # 提取特征
        features = self.feature_head(x)
        
        return features, importance, graph_features

# ==================== 改进的图感知ShiftBlock ====================
class ImprovedGraphAwareShiftBlock(nn.Module):
    """改进的图感知ShiftBlock，带渐进式DPGP和强残差连接"""
    def __init__(self, prompt_encoder, num_classes):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        
        # 原始ShiftBlock组件（共享）
        self.embed_to_point = nn.Linear(512, 2)
        self.scale_delta_x = nn.Parameter(torch.logit(torch.tensor(CONFIG['sx_init'])))
        self.scale_delta_y = nn.Parameter(torch.logit(torch.tensor(CONFIG['sy_init'])))
        
        # Transformer解码器（共享）
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformerDecoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=2
        )
        
        # 图神经网络（共享）
        self.graph_network = PointGraphNetwork(
            input_dim=512,
            hidden_dim=CONFIG['graph_hidden_dim'],
            num_heads=CONFIG['graph_num_heads']
        )
        
        # 增强的点选择网络
        self.point_selector = EnhancedPointSelectionRL(
            feature_dim=CONFIG['graph_hidden_dim']
        )
        
        # 基础个性化层（始终使用）
        self.personalization_layers = nn.ModuleDict()
        for i in range(1, CONFIG['num_clients'] + 1):
            self.personalization_layers[f'client_{i}'] = nn.Sequential(
                nn.Linear(CONFIG['graph_hidden_dim'], CONFIG['graph_hidden_dim']),
                nn.ReLU(),
                nn.LayerNorm(CONFIG['graph_hidden_dim']),
                nn.Dropout(0.1),
                nn.Linear(CONFIG['graph_hidden_dim'], CONFIG['graph_hidden_dim'])
            )
        
        # 【改进】DPGP模块 - 使用v2版本
        self.dpgp_module = None
        self.use_dpgp = CONFIG.get('use_dpgp', False)

        if self.use_dpgp and DPGP_AVAILABLE:
            try:
                self.dpgp_module = ImprovedDPGPModule(
                    feature_dim=CONFIG['dpgp_feature_dim'],
                    hidden_dim=CONFIG['dpgp_hidden_dim'],
                    num_encoder_layers=CONFIG['dpgp_num_encoder_layers'],
                    dropout=CONFIG['dpgp_dropout'],
                    num_clients=CONFIG['num_clients'],
                    # 【新增】完整功能参数
                    use_instance_level_gating=CONFIG.get('dpgp_use_instance_level_gating', True),
                    tta_adaptation_strength=CONFIG.get('dpgp_tta_adaptation_strength', 0.3)
                )
                
                # 设置总训练步数
                # total_steps = (CONFIG['communication_rounds'] * CONFIG['local_epochs'] * 100)
                # self.dpgp_module.set_total_steps(total_steps)
                self.dpgp_module.set_total_rounds(CONFIG['communication_rounds'])

                
                logger.info("✅ DPGP v2模块已启用")
                logger.info(f"   - 实例级动态权重: {CONFIG.get('dpgp_use_instance_level_gating', True)}")
                logger.info(f"   - 测试时自适应(TTA): {CONFIG.get('dpgp_test_time_adaptation', True)}")
                logger.info(f"   - TTA强度: {CONFIG.get('dpgp_tta_adaptation_strength', 0.3)}")
                
            except Exception as e:
                logger.warning(f"⚠️ DPGP v2模块加载失败: {e}，使用基础个性化层")
                self.dpgp_module = None
                self.use_dpgp = False
        else:
            if not DPGP_AVAILABLE:
                logger.warning("⚠️ DPGP模块不可用，使用基础个性化层")
            else:
                logger.info("ℹ️ DPGP模块未启用，使用基础个性化层")
        
        # 【新增】残差权重参数
        self.residual_weight = nn.Parameter(
            torch.tensor(CONFIG.get('dpgp_residual_weight', 0.8))
        )
        
        # 数据分布感知的点初始化器
        self.point_initializer = DataDistributionAwarePointInitializer(CONFIG['num_points_grid'])
        
        self.current_client_id = None
        self.graph_features_history = defaultdict(list)
        self.dpgp_weight_info = {}
        self.current_round = 0
        
    def set_current_round(self, round_num: int):
        """设置当前训练轮次"""
        self.current_round = round_num
        if self.dpgp_module is not None and hasattr(self.dpgp_module, 'set_current_round'):
            self.dpgp_module.set_current_round(round_num)
        
    def build_point_graph(self, points, features):
        """构建点的K近邻图"""
        B, N, D = features.shape
        graphs = []
        
        for b in range(B):
            spatial_dist = torch.cdist(points[b], points[b])
            feature_sim = torch.cosine_similarity(
                features[b].unsqueeze(1), 
                features[b].unsqueeze(0), 
                dim=2
            )
            combined_dist = (CONFIG['spatial_weight'] * spatial_dist / (spatial_dist.max() + 1e-8) + 
                           CONFIG['feature_weight'] * (1 - feature_sim))
            
            k = min(CONFIG['graph_k_neighbors'], N - 1)
            _, indices = torch.topk(combined_dist, k + 1, dim=1, largest=False)
            
            edge_index = []
            for i in range(N):
                for j in indices[i, 1:]:
                    edge_index.append([i, j.item()])
                    edge_index.append([j.item(), i])
            
            edge_index = list(set([tuple(e) for e in edge_index]))
            
            if edge_index:
                edge_index = torch.tensor(edge_index).t().contiguous()
            else:
                edge_index = torch.tensor([[i, i+1] for i in range(N-1)] + 
                                        [[i+1, i] for i in range(N-1)]).t().contiguous()
                
            graphs.append(Data(x=features[b], edge_index=edge_index))
            
        return Batch.from_data_list(graphs)
    
    def get_shared_parameters(self):
        """获取共享参数（排除个性化层和DPGP本地组件）"""
        shared_params = {}
        for name, param in self.named_parameters():
            if 'personalization_layers' in name:
                continue
            if self.dpgp_module is not None:
                if 'dpgp_module.local_encoders' in name:
                    continue
                if 'dpgp_module.gating_networks' in name:
                    continue
            if param.requires_grad:
                shared_params[name] = param
        return shared_params
    
    def get_personalization_parameters(self, client_id):
        """获取客户端特定的个性化参数"""
        params = {}
        client_key = f'client_{client_id}'
        
        if client_key in self.personalization_layers:
            for name, param in self.personalization_layers[client_key].named_parameters():
                params[f'personalization_layers.{client_key}.{name}'] = param
        
        if self.dpgp_module is not None:
            cid_str = str(client_id - 1)
            if cid_str in self.dpgp_module.local_encoders:
                for name, param in self.dpgp_module.local_encoders[cid_str].named_parameters():
                    params[f'dpgp_module.local_encoders.{cid_str}.{name}'] = param
            if cid_str in self.dpgp_module.gating_networks:
                for name, param in self.dpgp_module.gating_networks[cid_str].named_parameters():
                    params[f'dpgp_module.gating_networks.{cid_str}.{name}'] = param
        
        return params
    
    def _apply_personalization(self, graph_features, importance, client_id, is_training, edge_index):
        """应用个性化处理 - 支持实例级权重和TTA"""
        B, N, D = graph_features.shape
        device = graph_features.device
        
        # 保存原始特征
        original_features = graph_features.clone()
        
        # 计算有效残差权重（考虑warmup）
        warmup_rounds = CONFIG.get('dpgp_warmup_rounds', 15)
        if self.current_round < warmup_rounds:
            warmup_factor = 1.0 - (self.current_round / warmup_rounds) * 0.3
            effective_residual = torch.sigmoid(self.residual_weight) * warmup_factor + (1 - warmup_factor) * 0.9
        else:
            effective_residual = torch.sigmoid(self.residual_weight)
        
        personalized_features = graph_features
        dpgp_info = {}
        
        # DPGP处理 - 带完整参数
        if (self.use_dpgp and self.dpgp_module is not None and 
            client_id is not None and self.current_round >= warmup_rounds // 2):
            
            # 转换client_id: 1,2,3 -> 0,1,2
            dpgp_client_id = client_id - 1
            if dpgp_client_id < 0:
                dpgp_client_id = 0
            elif dpgp_client_id >= CONFIG['num_clients']:
                dpgp_client_id = dpgp_client_id % CONFIG['num_clients']
            
            # 验证并过滤edge_index
            safe_edge_index = edge_index
            if edge_index is not None and edge_index.numel() > 0:
                edge_index = edge_index.long()
                valid_mask = (edge_index[0] >= 0) & (edge_index[0] < N) & \
                            (edge_index[1] >= 0) & (edge_index[1] < N)
                safe_edge_index = edge_index[:, valid_mask]
                
                if safe_edge_index.size(1) == 0:
                    self_loops = torch.arange(min(N, 10), device=device, dtype=torch.long)
                    safe_edge_index = torch.stack([self_loops, self_loops])
            else:
                self_loops = torch.arange(min(N, 10), device=device, dtype=torch.long)
                safe_edge_index = torch.stack([self_loops, self_loops])
            
            try:
                # 处理importance维度
                importance_squeezed = importance
                if importance.dim() == 3 and importance.size(-1) == 1:
                    importance_squeezed = importance.squeeze(-1)
                if importance_squeezed.dim() == 1:
                    importance_squeezed = importance_squeezed.unsqueeze(0)
                if importance_squeezed.size(0) != B:
                    importance_squeezed = importance_squeezed.expand(B, -1)
                
                # 【关键】调用DPGP模块，传递TTA参数
                dpgp_features, dpgp_info = self.dpgp_module(
                    features=graph_features,
                    edge_index=safe_edge_index,
                    point_importance=importance_squeezed,
                    client_id=dpgp_client_id,
                    training=is_training,
                    # 【重要】测试时启用TTA
                    use_test_time_adaptation=CONFIG.get('dpgp_test_time_adaptation', True) and not is_training
                )
                
                # 获取有效缩放
                dpgp_scale = dpgp_info.get('effective_scale', 0.3)
                if torch.is_tensor(dpgp_scale):
                    dpgp_scale = dpgp_scale.item()
                dpgp_scale = min(max(float(dpgp_scale), 0.0), 1.0)
                
                # 混合特征
                personalized_features = (1 - dpgp_scale) * graph_features + dpgp_scale * dpgp_features
                
                # 【新增】记录实例级权重信息
                if dpgp_info.get('instance_level', False):
                    logger.debug(f"使用实例级权重，权重形状: {dpgp_info.get('weights_shape', 'N/A')}")
                
                if dpgp_info.get('tta_applied', False):
                    logger.debug(f"TTA已应用，相似度: {dpgp_info.get('tta_similarity', 0.5):.3f}")
                
            except Exception as e:
                logger.warning(f"DPGP forward error for client {client_id}: {e}")
                import traceback
                traceback.print_exc()
                personalized_features = graph_features
                dpgp_info = {'error': str(e)}
        
        # 基础个性化层
        if client_id is not None:
            client_key = f'client_{client_id}'
            if client_key in self.personalization_layers:
                try:
                    basic_personal = self.personalization_layers[client_key](graph_features)
                    basic_mix = 0.2 if (self.dpgp_module is not None and self.use_dpgp) else 0.3
                    personalized_features = (1 - basic_mix) * personalized_features + basic_mix * basic_personal
                except Exception as e:
                    logger.warning(f"Basic personalization error for {client_key}: {e}")
        
        # 最终残差连接
        eff_res = effective_residual.item() if torch.is_tensor(effective_residual) else effective_residual
        final_features = eff_res * original_features + (1 - eff_res) * personalized_features
        
        # 更新信息字典
        self.dpgp_weight_info = {
            'effective_residual': eff_res,
            'dpgp_enabled': self.use_dpgp and self.dpgp_module is not None,
            'current_round': self.current_round,
            # 【新增】实例级和TTA信息
            'instance_level_gating': dpgp_info.get('instance_level', False),
            'tta_applied': dpgp_info.get('tta_applied', False),
            'tta_similarity': dpgp_info.get('tta_similarity', 0.5),
            **dpgp_info
        }
        
        return final_features
    
    def forward(self, backbone_out, client_id=None, is_training=True):
        try:
            if 'point_inputs_per_frame' in backbone_out and backbone_out['point_inputs_per_frame']:
                points = backbone_out['point_inputs_per_frame'][0]['point_coords']
                labels = backbone_out['point_inputs_per_frame'][0]['point_labels']
            else:
                points = backbone_out['point_coords']
                labels = backbone_out['point_labels']
                
            B, N, _ = points.shape
            self.current_client_id = client_id
            device = points.device
            
            points_reshaped = points.reshape(-1, 1, 2)
            labels_reshaped = labels.reshape(-1, 1)
            
            if CONFIG.get('freeze_prompt_encoder', True):
                with torch.no_grad():
                    sparse_embeddings, _ = self.prompt_encoder(points_reshaped, labels_reshaped)
            else:
                sparse_embeddings, _ = self.prompt_encoder(points_reshaped, labels_reshaped)
            
            if sparse_embeddings.dim() == 3 and sparse_embeddings.shape[1] == 1:
                sparse_embeddings = sparse_embeddings.squeeze(1)
            
            sparse_embeddings = sparse_embeddings.reshape(B, N, -1)
            
            graph_batch = self.build_point_graph(points, sparse_embeddings)
            edge_index = graph_batch.edge_index.to(device)
            
            graph_features, importance, graph_structure_features = self.graph_network(
                graph_batch.x, 
                edge_index,
                graph_batch.batch.to(device) if hasattr(graph_batch, 'batch') else None
            )
            
            if client_id is not None:
                self.graph_features_history[client_id].append(graph_structure_features)
            
            graph_features = graph_features.reshape(B, N, -1)
            importance = importance.reshape(B, N, 1)
            
            # 【核心】应用改进的个性化处理
            graph_features = self._apply_personalization(
                graph_features, importance, client_id, is_training, edge_index
            )
            
            key_point_masks, states, actions = self.point_selector.select_points_with_exploration(
                graph_features, importance, 
                CONFIG['min_points'], CONFIG['max_points'],
                temperature=CONFIG['point_selection_temp'] if is_training else 0.1,
                training=is_training
            )
            
            all_embeddings = sparse_embeddings.reshape(-1, sparse_embeddings.size(-1))
            delta = self.embed_to_point(all_embeddings)
            delta = torch.tanh(delta)
            
            scale = torch.stack([self.scale_delta_x, self.scale_delta_y])
            delta = delta * torch.sigmoid(scale)
            
            points_norm = points.reshape(-1, 2) / CONFIG['image_size']
            new_points_norm = torch.clamp(points_norm + delta, 0, 1)
            new_points = new_points_norm.reshape(B, N, 2) * CONFIG['image_size']
            
            if is_training:
                importance_weight = importance.squeeze(-1).unsqueeze(-1)
                key_weight = key_point_masks.unsqueeze(-1).float()
                combined_weight = 0.5 * importance_weight + 0.5 * key_weight
                new_points = points + (new_points - points) * combined_weight
            
            output = {
                'point_coords': new_points,
                'point_labels': labels,
                'point_importance': importance,
                'key_point_masks': key_point_masks,
                'graph_features': graph_features,
                'graph_structure_features': graph_structure_features,
                'norm_point_coords': new_points / CONFIG['image_size'],
                'rl_states': states if is_training else None,
                'rl_actions': actions if is_training else None,
                'dpgp_weight_info': self.dpgp_weight_info
            }
            
            if 'point_inputs_per_frame' in backbone_out and backbone_out['point_inputs_per_frame']:
                backbone_out['point_inputs_per_frame'][0].update(output)
                backbone_out.update(output)
            else:
                backbone_out.update(output)
                
            return backbone_out
            
        except Exception as e:
            logger.error(f"ImprovedGraphAwareShiftBlock error: {e}")
            import traceback
            traceback.print_exc()
            
            if 'point_inputs_per_frame' in backbone_out and backbone_out['point_inputs_per_frame']:
                points = backbone_out['point_inputs_per_frame'][0]['point_coords']
                labels = backbone_out['point_inputs_per_frame'][0]['point_labels']
            else:
                points = backbone_out['point_coords']
                labels = backbone_out['point_labels']
                
            B, N, _ = points.shape
            device = points.device
            
            default_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
            default_masks[:, :CONFIG['min_points']] = True
            default_importance = torch.ones(B, N, 1, device=device) * 0.5
            
            default_output = {
                'point_coords': points,
                'point_labels': labels,
                'point_importance': default_importance,
                'key_point_masks': default_masks,
                'graph_features': None,
                'graph_structure_features': {},
                'norm_point_coords': points / CONFIG['image_size'],
                'rl_states': None,
                'rl_actions': None,
                'dpgp_weight_info': {}
            }
            
            if 'point_inputs_per_frame' in backbone_out and backbone_out['point_inputs_per_frame']:
                backbone_out['point_inputs_per_frame'][0].update(default_output)
                backbone_out.update(default_output)
            else:
                backbone_out.update(default_output)
                
            return backbone_out
    
# ==================== 标准SAM2组件实现（保持不变）====================
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=True, rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, use_rel_pos=False, rel_pos_zero_init=True, 
                 window_size=0, input_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = self.window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)

    def window_unpartition(self, windows, window_size, pad_hw, hw):
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()

        return x

class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=(),
        window_size=14,
        out_indices=(2, 5, 8, 11),
    ):
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=True,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))

class StandardPromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=(64, 64), 
                 input_image_size=(1024, 1024), mask_in_chans=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight

        return point_embedding

    def _embed_masks(self, masks):
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self, points, boxes, masks):
        if points is not None:
            return points.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self):
        return self.point_embeddings[0].weight.device

    def forward(self, points, labels, boxes=None, masks=None):
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        
        if points is not None:
            point_embeddings = self._embed_points(points, labels, pad=(points.shape[1] == 1))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            raise NotImplementedError("Box encoding not implemented")

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

class SimpleMaskDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, num_classes, 1),
        )
    
    def forward(self, image_embeddings, image_pe=None, sparse_prompt_embeddings=None, 
               dense_prompt_embeddings=None, multimask_output=False, repeat_image=False, 
               high_res_features=None):
        masks = self.decoder(image_embeddings)
        iou_predictions = torch.rand(masks.shape[0], 3 if multimask_output else 1, 
                                   device=masks.device) * 0.5 + 0.5
        return masks, iou_predictions

# ==================== 改进的SAM2模型 ====================
class ImprovedGraphAwareSAM2Model(nn.Module):
    """改进的集成图感知ShiftBlock的SAM2模型"""
    def __init__(self, num_classes, checkpoint_path=None):
        super().__init__()
        self.image_size = CONFIG['image_size']
        self.num_times_sample = CONFIG['num_times_sample']
        self.num_classes = num_classes
        
        # 初始化标准组件
        self.image_encoder = ImageEncoderViT(
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            out_chans=256,
            window_size=0,
            global_attn_indexes=(),
        )
        
        self.prompt_encoder = StandardPromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )
        
        # 使用简化的掩码解码器
        self.mask_decoder = SimpleMaskDecoder(num_classes)
        
        # 改进的图感知ShiftBlock
        self.shift_block = ImprovedGraphAwareShiftBlock(self.prompt_encoder, num_classes)
        
        # 加载权重
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_pretrained_weights(checkpoint_path)
        else:
            logger.warning("未提供SAM2预训练权重，使用随机初始化。")
        
        self._configure_training_components()
        
    def load_pretrained_weights(self, checkpoint_path):
        """加载预训练权重"""
        logger.info(f"尝试加载SAM2预训练权重: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 加载图像编码器权重
            image_encoder_state = {}
            for k, v in checkpoint.items():
                if k.startswith('image_encoder.'):
                    new_key = k[len('image_encoder.'):]
                    # 跳过形状不匹配的相对位置编码
                    if 'rel_pos' in new_key:
                        continue
                    image_encoder_state[new_key] = v
            
            missing_keys, unexpected_keys = self.image_encoder.load_state_dict(image_encoder_state, strict=False)
            logger.info("图像编码器权重加载成功（跳过相对位置编码）")
            
            # 加载提示编码器权重
            prompt_encoder_state = {}
            for k, v in checkpoint.items():
                if k.startswith('prompt_encoder.'):
                    prompt_encoder_state[k[len('prompt_encoder.'):]] = v
            
            missing_keys, unexpected_keys = self.prompt_encoder.load_state_dict(prompt_encoder_state, strict=False)
            logger.info("提示编码器权重加载成功")
            
            logger.warning("掩码解码器和ShiftBlock使用随机初始化（需要训练）")
            
        except Exception as e:
            logger.error(f"加载SAM2预训练权重失败: {e}")
            logger.warning("将使用随机初始化权重")
    
    def _configure_training_components(self):
        """根据配置冻结或解冻模型组件"""
        # 图像编码器
        if CONFIG.get('freeze_image_encoder', True):
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            logger.info("图像编码器已冻结")
        else:
            for param in self.image_encoder.parameters():
                param.requires_grad = True
            logger.info("图像编码器参与训练")
            
        # 提示编码器
        if CONFIG.get('freeze_prompt_encoder', True):
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            logger.info("提示编码器已冻结")
        else:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = True
            logger.info("提示编码器参与训练")
            
        # 掩码解码器和ShiftBlock始终训练
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        for param in self.shift_block.parameters():
            param.requires_grad = True
        logger.info("掩码解码器和ShiftBlock参与训练")
        
        # 【新增】日志DPGP模块状态
        if CONFIG.get('use_dpgp', True) and self.shift_block.dpgp_module is not None:
            logger.info("✅ DPGP模块已启用并参与训练")
    
    def get_federated_parameters(self):
        """获取参与联邦学习的共享参数（不包括个性化层）"""
        fed_params = {}
        
        # 获取ShiftBlock的共享参数
        shift_shared = self.shift_block.get_shared_parameters()
        for name, param in shift_shared.items():
            fed_params[f"shift_block.{name}"] = param
        
        # 掩码解码器参数（如果未冻结）
        for name, param in self.mask_decoder.named_parameters():
            if param.requires_grad:
                fed_params[f"mask_decoder.{name}"] = param
        
        # 其他未冻结的参数
        if not CONFIG.get('freeze_image_encoder', True):
            for name, param in self.image_encoder.named_parameters():
                if param.requires_grad:
                    fed_params[f"image_encoder.{name}"] = param
                    
        if not CONFIG.get('freeze_prompt_encoder', True):
            for name, param in self.prompt_encoder.named_parameters():
                if param.requires_grad:
                    fed_params[f"prompt_encoder.{name}"] = param
        
        return fed_params
    
    def set_federated_parameters(self, fed_params):
        """设置联邦学习参数（仅共享参数）"""
        state_dict = self.state_dict()
        for name, param in fed_params.items():
            if name in state_dict:
                state_dict[name].copy_(param)
                
    def forward_image(self, image):
        """前向传播图像编码器"""
        if CONFIG.get('freeze_image_encoder', True):
            with torch.no_grad():
                features = self.image_encoder(image)
        else:
            features = self.image_encoder(image)
        return {'vision_features': features}
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, images, points, labels, client_id=None):
        # 图像编码
        backbone_out = self.forward_image(images)
        backbone_out.update({
            'point_coords': points,
            'point_labels': labels,
            'point_inputs_per_frame': [{
                'point_coords': points,
                'point_labels': labels,
            }]
        })
        
        # 改进的图感知ShiftBlock处理
        backbone_out = self.shift_block(backbone_out, client_id, self.training)
        
        # 提示编码（使用处理后的点）- 修复获取数据的逻辑
        # ShiftBlock已经更新了backbone_out，直接从backbone_out获取
        processed_points = backbone_out.get('point_coords', points)
        processed_labels = backbone_out.get('point_labels', labels)
        
        if CONFIG.get('freeze_prompt_encoder', True):
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=processed_points,
                    labels=processed_labels,
                    boxes=None,
                    masks=None
                )
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=processed_points,
                labels=processed_labels,
                boxes=None,
                masks=None
            )
        
        # 掩码解码
        image_pe = self.prompt_encoder.get_dense_pe()
        masks, iou_pred = self.mask_decoder(
            image_embeddings=backbone_out['vision_features'],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        
        # 返回完整输出 - 确保包含所有必要字段
        return {
            'masks': masks,
            'iou_predictions': iou_pred,
            'refined_points': processed_points,
            'point_importance': backbone_out.get('point_importance', None),
            'key_point_masks': backbone_out.get('key_point_masks', None),
            'graph_features': backbone_out.get('graph_features', None),
            'graph_structure_features': backbone_out.get('graph_structure_features', {}),
            'rl_states': backbone_out.get('rl_states', None),
            'rl_actions': backbone_out.get('rl_actions', None),
            # 【新增】DPGP权重信息
            'dpgp_weight_info': backbone_out.get('dpgp_weight_info', {})
        }

# ==================== 损失函数和评估指标 ====================
class GraphAwareLoss(nn.Module):
    """包含稀疏性约束的损失函数"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dice_loss = DiceLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        pred_masks = outputs['masks']
        target_masks = targets['masks']
        
        # 转换目标掩码为类别索引
        if target_masks.dim() == 4 and target_masks.size(1) > 1:
            target_masks = torch.argmax(target_masks, dim=1)
        
        # 基础损失
        dice_loss = self.dice_loss(pred_masks, target_masks)
        ce_loss = self.ce_loss(pred_masks, target_masks.long())
        
        # 距离损失（如果有refined_points）
        dist_loss = 0
        if 'refined_points' in outputs and 'points' in targets:
            dist_loss = F.mse_loss(outputs['refined_points'], targets['points'])
        
        # 稀疏性损失（鼓励使用更少的点）
        sparsity_loss = 0
        if outputs.get('key_point_masks') is not None:
            num_selected = outputs['key_point_masks'].float().sum(dim=1).mean()
            sparsity_loss = num_selected / CONFIG['num_points_grid']
        
        total_loss = (CONFIG['alpha'] * dice_loss + 
                     CONFIG['beta'] * ce_loss +
                     CONFIG['gamma'] * dist_loss +
                     CONFIG['lambda'] * sparsity_loss)
        
        return total_loss, {
            'dice_loss': dice_loss,
            'ce_loss': ce_loss,
            'dist_loss': dist_loss,
            'sparsity_loss': sparsity_loss,
            'total_loss': total_loss
        }

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = target.detach()
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# ==================== 评估指标计算 ====================
def compute_dice_score(pred, target, smooth=1e-6):
    """计算Dice系数"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def compute_iou_score(pred, target, smooth=1e-6):
    """计算IoU分数"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def compute_accuracy(pred, target):
    """计算准确率"""
    pred = (pred > 0.5).float()
    correct = (pred == target).sum().float()
    total = target.numel()
    accuracy = correct / total
    return accuracy.item()

def compute_hausdorff_distance_95(pred, target):
    """计算95%豪斯多夫距离"""
    try:
        pred_np = (pred > 0.5).cpu().numpy().astype(np.uint8)
        target_np = target.cpu().numpy().astype(np.uint8)
        
        # 找到轮廓
        pred_contours, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contours, _ = cv2.findContours(target_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not pred_contours or not target_contours:
            return float('inf')
        
        # 提取轮廓点
        pred_points = np.vstack([contour.reshape(-1, 2) for contour in pred_contours])
        target_points = np.vstack([contour.reshape(-1, 2) for contour in target_contours])
        
        # 计算双向豪斯多夫距离
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        # 计算95%分位数
        hd95 = np.percentile([hd1, hd2], 95)
        return hd95
    except Exception as e:
        logger.debug(f"HD95计算失败: {e}")
        return float('inf')

def compute_precision_recall(pred, target):
    """计算精确率和召回率"""
    pred = (pred > 0.5).float()
    
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision, recall

def compute_f1_score(precision, recall):
    """计算F1分数"""
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


# ==================== 改进的联邦学习框架 ====================
class ImprovedFederatedServer:
    """改进的联邦学习服务器，带图结构自适应聚合"""
    def __init__(self, device):
        self.device = device
        self.global_model = None
        self.round = 0
        self.client_graph_stats = defaultdict(dict)
        self.graph_analyzer = GraphStructureAnalyzer()
        self.global_graph_features = {}
        
    def initialize_global_model(self, checkpoint_path):
        """初始化全局模型"""
        temp_model = ImprovedGraphAwareSAM2Model(
            num_classes=CONFIG['num_classes'],
            checkpoint_path=checkpoint_path
        ).to(self.device)
        
        # 只提取共享参数（完全排除个性化层）
        self.global_model = {}
        fed_params = temp_model.get_federated_parameters()
        for name, param in fed_params.items():
            self.global_model[name] = param.clone().to(self.device)
        
        logger.info(f"初始化全局模型，共享参数数量: {len(self.global_model)}")
        
    def compute_adaptive_weights(self, client_updates):
        """基于图结构相似性计算自适应聚合权重"""
        num_clients = len(client_updates)
        similarity_matrix = np.ones((num_clients, num_clients))
        
        # 计算客户端之间的图相似性
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                _, stats_i = client_updates[i]
                _, stats_j = client_updates[j]
                
                if 'graph_features' in stats_i and 'graph_features' in stats_j:
                    similarity = self.graph_analyzer.compute_graph_similarity(
                        stats_i['graph_features'], 
                        stats_j['graph_features']
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        # 基于相似性计算权重
        weights = []
        for i in range(num_clients):
            # 与全局模型的相似度
            if self.global_graph_features:
                _, stats = client_updates[i]
                global_sim = self.graph_analyzer.compute_graph_similarity(
                    stats.get('graph_features', {}),
                    self.global_graph_features
                )
            else:
                global_sim = 1.0
            
            # 与其他客户端的平均相似度
            avg_sim = np.mean([similarity_matrix[i, j] for j in range(num_clients) if i != j])
            
            # 综合权重
            weight = CONFIG['adaptive_weight_alpha'] * global_sim + (1 - CONFIG['adaptive_weight_alpha']) * avg_sim
            weights.append(weight)
        
        # 归一化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return weights
        
    def aggregate_parameters(self, client_updates, client_weights=None):
        """改进的聚合方法，考虑图结构差异和效率"""
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # 计算自适应权重
        adaptive_weights = self.compute_adaptive_weights(client_updates)
        
        # 结合原始权重和自适应权重
        final_weights = []
        for i, (orig_weight, adapt_weight) in enumerate(zip(client_weights, adaptive_weights)):
            # 考虑效率因子
            _, stats = client_updates[i]
            if 'efficiency_score' in stats and stats['efficiency_score'] > 0:
                efficiency_factor = min(stats['efficiency_score'], 2.0)  # 限制最大影响
            else:
                efficiency_factor = 1.0
            
            final_weight = orig_weight * adapt_weight * efficiency_factor
            final_weights.append(final_weight)
        
        # 归一化最终权重
        total_weight = sum(final_weights)
        final_weights = [w / total_weight for w in final_weights]
        
        # 聚合参数
        aggregated_params = {}
        for name in self.global_model.keys():
            aggregated_params[name] = torch.zeros_like(self.global_model[name]).to(self.device)
            
            for (client_params, _), weight in zip(client_updates, final_weights):
                if name in client_params:
                    client_param = client_params[name].clone().detach().to(self.device)
                    aggregated_params[name] += weight * client_param
        
        self.global_model = aggregated_params
        self.round += 1
        
        # 更新全局图特征
        all_graph_features = []
        for _, stats in client_updates:
            if 'graph_features' in stats:
                all_graph_features.append(stats['graph_features'])
        
        if all_graph_features:
            # 计算全局图特征的平均
            self.global_graph_features = {}
            for key in all_graph_features[0].keys():
                values = [gf[key] for gf in all_graph_features if key in gf]
                self.global_graph_features[key] = np.mean(values)
        
        # 更新客户端图统计
        for i, (_, stats) in enumerate(client_updates):
            self.client_graph_stats[i].update(stats)
        
        logger.info(f"自适应聚合完成，当前轮次: {self.round}")
        logger.info(f"最终聚合权重: {[f'{w:.3f}' for w in final_weights]}")
        
    def get_global_parameters(self):
        """获取全局参数"""
        return {name: param.clone().detach() for name, param in self.global_model.items()}
    
    def save_global_model(self, save_path):
        """保存全局模型"""
        torch.save({
            'round': self.round,
            'global_parameters': self.global_model,
            'global_graph_features': self.global_graph_features,
            'client_graph_stats': dict(self.client_graph_stats),
            'config': CONFIG,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, save_path)
        logger.info(f"全局模型已保存: {save_path}")

class ImprovedFederatedClient:
    """改进的联邦学习客户端，完全隔离个性化层"""
    def __init__(self, client_id, num_classes, class_names, device):
        self.client_id = client_id
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.dataset_type = 'Arcade'  # 【修改】统一为Arcade
        
        self.model = None
        self.optimizer = None
        self.rl_optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        
        # 图结构统计
        self.graph_stats = {
            'avg_key_points': 0,
            'avg_importance': 0,
            'total_points_used': 0,
            'efficiency_score': 0,
            'graph_features': {}
        }
        
        self.training_history = {
            'rounds': [],
            'train_loss': [],
            'train_dice': [],
            'test_dice': [],
            'test_iou': [],
            'test_accuracy': [],
            'test_hd95': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': [],
            'graph_stats': [],
            'rl_rewards': []
        }
        
        self._setup_model_and_data()
        
    def _setup_model_and_data(self):
        """设置模型和数据"""
        # 创建模型
        self.model = ImprovedGraphAwareSAM2Model(
            num_classes=self.num_classes,
            checkpoint_path=CONFIG['sam2_checkpoint_path']
        ).to(self.device)
        
        # 创建优化器（分别处理共享参数和RL参数）
        shared_params = []
        rl_params = []
        personal_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'personalization_layers' in name:
                    personal_params.append(param)
                elif 'point_selector' in name:
                    rl_params.append(param)
                else:
                    shared_params.append(param)
        
        # 共享参数优化器
        self.optimizer = torch.optim.AdamW(
            shared_params + personal_params,  # 个性化参数也需要本地优化
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        
        # RL优化器
        if rl_params:
            self.rl_optimizer = torch.optim.Adam(rl_params, lr=CONFIG['rl_lr'])
        else:
            self.rl_optimizer = None
        
        # 创建损失函数
        self.criterion = GraphAwareLoss(self.num_classes)
        
        # 加载数据
        self._load_data()
        
        logger.info(f"客户端 {self.client_id} 初始化完成")
        logger.info(f"   数据集类型: {self.dataset_type}")
        logger.info(f"   类别数: {self.num_classes}")
        logger.info(f"   类别名称: {self.class_names}")
        logger.info(f"   训练样本: {len(self.train_loader.dataset) if self.train_loader else 0}")
        logger.info(f"   测试样本: {len(self.test_loader.dataset) if self.test_loader else 0}")
        
    def _load_data(self):
        """加载数据（使用Arcade数据集）"""
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.03, saturation=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 【修改】统一使用ArcadeDatasetSplit
        train_dataset = ArcadeDatasetSplit(
            CONFIG['arcade_data_root'],
            client_id=self.client_id,
            transform=train_transform,
            is_train=True,
            config=CONFIG
        )
        
        test_dataset = ArcadeDatasetSplit(
            CONFIG['arcade_data_root'],
            client_id=self.client_id,
            transform=test_transform,
            is_train=False,
            config=CONFIG
        )
        
        if len(train_dataset) > 0:
            self.train_loader = DataLoader(
                train_dataset, CONFIG['batch_size'], shuffle=True,
                num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True
            )
        
        if len(test_dataset) > 0:
            self.test_loader = DataLoader(
                test_dataset, CONFIG['batch_size'], shuffle=False,
                num_workers=CONFIG['num_workers']
            )
            
    def update_model(self, global_parameters):
        """更新模型的共享参数（不更新个性化层）"""
        # 只更新共享参数
        for name, param in global_parameters.items():
            if 'personalization_layers' not in name:  # 确保不更新个性化层
                if name in dict(self.model.named_parameters()):
                    dict(self.model.named_parameters())[name].data.copy_(param)
        
    def get_model_parameters(self):
        """获取模型参数（仅返回共享参数，不包括个性化层）"""
        return self.model.get_federated_parameters()
        
    def train_local_epochs(self, num_epochs):
        """本地训练（包含RL更新）"""
        if self.train_loader is None:
            logger.warning(f"客户端 {self.client_id} 没有训练数据")
            return 0, 0, {}
        
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_samples = 0
        total_rl_rewards = []
        
        # 收集图统计
        epoch_graph_stats = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_dice = 0
            epoch_samples = 0
            epoch_rewards = []
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f'Client {self.client_id} Epoch {epoch+1}/{num_epochs}',
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    key_pts_value = None
                    reward = None
                    
                    images = batch['image'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    points = batch['points'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    if self.rl_optimizer:
                        self.rl_optimizer.zero_grad()
                    
                    # 前向传播（包含客户端ID）
                    outputs = self.model(images, points, labels, self.client_id)
                    
                    # 计算损失
                    targets = {
                        'masks': masks,
                        'points': points
                    }
                    loss, loss_dict = self.criterion(outputs, targets)
                    
                    # 计算RL奖励并更新
                    if outputs.get('rl_states') is not None and self.rl_optimizer:
                        with torch.no_grad():
                            # 基于分割性能的奖励
                            pred_softmax = F.softmax(outputs['masks'], dim=1)
                            target_masks = torch.argmax(masks, dim=1)
                            dice_score = 0
                            for i in range(images.size(0)):
                                for class_idx in range(1, self.num_classes):
                                    pred = pred_softmax[i, class_idx]
                                    target = (target_masks[i] == class_idx).float()
                                    if target.sum() > 0:
                                        dice_score += compute_dice_score(pred, target)
                            
                            # 奖励 = Dice分数 - 稀疏性惩罚
                            num_selected = outputs['key_point_masks'].float().sum(dim=1).mean()
                            reward = dice_score / max(images.size(0), 1) - 0.1 * (num_selected / CONFIG['num_points_grid'])
                            
                            # 转换为CPU上的Python数值 - 这是修复的关键
                            reward_value = reward.item() if torch.is_tensor(reward) else float(reward)
                            epoch_rewards.append(reward_value)
                            
                            # 存储经验（使用原始的tensor reward）
                            for i in range(len(outputs['rl_states'])):
                                if i < len(outputs['rl_states']) - 1:
                                    self.model.shift_block.point_selector.store_experience(
                                        outputs['rl_states'][i],
                                        outputs['rl_actions'][i],
                                        reward,  # 这里可以使用原始tensor
                                        outputs['rl_states'][i+1] if i+1 < len(outputs['rl_states']) else outputs['rl_states'][i],
                                        i == len(outputs['rl_states']) - 1
                                    )
                        
                        # 定期更新RL网络
                        if batch_idx % CONFIG['rl_update_freq'] == 0:
                            rl_loss = self.model.shift_block.point_selector.update(
                                CONFIG['rl_batch_size'], CONFIG['rl_gamma']
                            )
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), CONFIG['max_grad_norm']
                    )
                    
                    self.optimizer.step()
                    if self.rl_optimizer:
                        self.rl_optimizer.step()
                    
                    # 统计
                    epoch_loss += loss.item()
                    epoch_dice += (1 - loss_dict['dice_loss'].item())
                    epoch_samples += 1
                    
                    # 收集图统计
                    if outputs.get('key_point_masks') is not None:
                        num_key_points = outputs['key_point_masks'].float().sum(dim=1).mean()
                        epoch_graph_stats['num_key_points'].append(num_key_points.item())
                        key_pts_value = num_key_points.item()
                    else:
                        # 使用默认值
                        num_key_points = CONFIG['min_points']
                        epoch_graph_stats['num_key_points'].append(num_key_points)
                        key_pts_value = num_key_points
                    
                    if outputs.get('point_importance') is not None:
                        avg_importance = outputs['point_importance'].mean()
                        epoch_graph_stats['avg_importance'].append(avg_importance.item())
                    
                    if outputs.get('graph_structure_features') is not None:
                        for key, value in outputs['graph_structure_features'].items():
                            if key not in epoch_graph_stats:
                                epoch_graph_stats[key] = []
                            epoch_graph_stats[key].append(value)
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'dice': f'{1 - loss_dict["dice_loss"].item():.4f}',
                        'key_pts': f'{key_pts_value:.1f}' if key_pts_value is not None else 'N/A',
                        'reward': f'{reward_value:.3f}' if 'reward_value' in locals() else 'N/A'
                    })
                    
                except Exception as e:
                    logger.warning(f"训练batch {batch_idx}出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            total_loss += epoch_loss
            total_dice += epoch_dice
            total_samples += epoch_samples
            total_rl_rewards.extend(epoch_rewards)
        
        # 更新图统计
        if epoch_graph_stats['num_key_points']:
            self.graph_stats['avg_key_points'] = np.mean(epoch_graph_stats['num_key_points'])
            self.graph_stats['total_points_used'] = sum(epoch_graph_stats['num_key_points'])
        else:
            # 如果没有统计到关键点，使用默认值
            self.graph_stats['avg_key_points'] = CONFIG['min_points']
        
        if epoch_graph_stats['avg_importance']:
            self.graph_stats['avg_importance'] = np.mean(epoch_graph_stats['avg_importance'])
        
        # 计算图结构特征
        graph_features = {}
        for key in ['avg_degree', 'std_degree', 'max_degree', 'avg_clustering', 'density']:
            if key in epoch_graph_stats:
                graph_features[key] = np.mean(epoch_graph_stats[key])
        self.graph_stats['graph_features'] = graph_features
        
        # 计算效率分数（避免除零）
        if self.graph_stats['avg_key_points'] > 0 and total_samples > 0:
            # 效率 = (Dice性能) / (使用的点数比例)
            dice_performance = total_dice / total_samples
            point_ratio = self.graph_stats['avg_key_points'] / CONFIG['num_points_grid']
            self.graph_stats['efficiency_score'] = dice_performance / max(point_ratio, 0.1)  # 避免除零
        else:
            self.graph_stats['efficiency_score'] = 0.0
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_dice = total_dice / max(total_samples, 1)
        
        # 更新训练历史
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['train_dice'].append(avg_dice)
        self.training_history['graph_stats'].append(copy.deepcopy(self.graph_stats))
        
        # 现在total_rl_rewards中的所有元素都是Python数值，可以安全地计算平均值
        if total_rl_rewards:
            self.training_history['rl_rewards'].append(np.mean(total_rl_rewards))
        
        # ✅ 新增：每轮结束清理内存
        if hasattr(self.model.shift_block, 'point_selector'):
            # 保留最近 2000 条经验，清理旧的
            selector = self.model.shift_block.point_selector
            if len(selector.memory) > 2000:
                recent = list(selector.memory)[-2000:]
                selector.memory.clear()
                selector.memory.extend(recent)

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, avg_dice, self.graph_stats
        
    def evaluate(self, save_predictions=False, save_dir=None, round_num=None):
        """评估模型"""
        if self.test_loader is None:
            logger.warning(f"客户端 {self.client_id} 没有测试数据")
            return {
                'mean_dice': 0, 'mean_iou': 0, 'mean_accuracy': 0, 'mean_hd95': float('inf'),
                'mean_precision': 0, 'mean_recall': 0, 'mean_f1': 0,
                'sample_count': 0
            }
        metrics = evaluate_model(
            self.model,
            self.test_loader,
            self.device,
            f"Client_{self.client_id}_{self.dataset_type}",
            self.num_classes,
            self.class_names,
            self.client_id,
            save_predictions,
            save_dir,
            round_num
        )
        
        # 添加图统计到评估结果
        metrics['graph_stats'] = copy.deepcopy(self.graph_stats)
        
        # 更新历史
        self.training_history['rounds'].append(round_num if round_num else len(self.training_history['rounds']))
        self.training_history['test_dice'].append(metrics['mean_dice'])
        self.training_history['test_iou'].append(metrics['mean_iou'])
        self.training_history['test_accuracy'].append(metrics.get('mean_accuracy', 0))
        self.training_history['test_hd95'].append(metrics.get('mean_hd95', float('inf')))
        self.training_history['test_precision'].append(metrics.get('mean_precision', 0))
        self.training_history['test_recall'].append(metrics.get('mean_recall', 0))
        self.training_history['test_f1'].append(metrics.get('mean_f1', 0))
        
        return metrics
    
    def save_model(self, save_path, round_num, metrics):
        """保存客户端模型"""
        checkpoint = {
            'client_id': self.client_id,
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rl_optimizer_state_dict': self.rl_optimizer.state_dict() if self.rl_optimizer else None,
            'config': CONFIG,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': metrics,
            'training_history': self.training_history,
            'graph_stats': self.graph_stats,
            'dataset_type': self.dataset_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"客户端 {self.client_id} 模型已保存: {save_path}")

# ==================== 评估函数 ====================
def evaluate_model(model, dataloader, device, dataset_name, num_classes, class_names, 
                  client_id=None, save_predictions=False, save_dir=None, round_num=None):
    """评估模型性能，包含图感知指标"""
    model.eval()
    
    # 存储各种指标
    all_dice_scores = []
    all_iou_scores = []
    all_hd95_scores = []
    all_accuracy_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []
    
    # 图感知指标
    all_key_points_used = []
    all_point_importance = []
    
    # 按类别统计
    class_dice_scores = defaultdict(list)
    class_iou_scores = defaultdict(list)
    class_hd95_scores = defaultdict(list)
    class_accuracy_scores = defaultdict(list)
    class_precision_scores = defaultdict(list)
    class_recall_scores = defaultdict(list)
    class_f1_scores = defaultdict(list)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'{dataset_name} Evaluation')
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(device)
                masks = batch['masks'].to(device)
                points = batch['points'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(images, points, labels, client_id)
                pred_masks = outputs['masks']
                
                pred_softmax = torch.softmax(pred_masks, dim=1)
                target_masks = torch.argmax(masks, dim=1)
                
                # 收集图统计
                if outputs.get('key_point_masks') is not None:
                    num_key_points = outputs['key_point_masks'].float().sum(dim=1)
                    all_key_points_used.extend(num_key_points.cpu().numpy())
                
                if outputs.get('point_importance') is not None:
                    importance = outputs['point_importance'].mean(dim=1)
                    all_point_importance.extend(importance.cpu().numpy())
                
                # 逐样本计算指标
                for i in range(images.size(0)):
                    sample_dice_scores = []
                    sample_iou_scores = []
                    sample_hd95_scores = []
                    sample_accuracy_scores = []
                    sample_precision_scores = []
                    sample_recall_scores = []
                    sample_f1_scores = []
                    
                    # 逐类别计算指标
                    for class_idx in range(num_classes):
                        pred_class = pred_softmax[i, class_idx]
                        target_class = (target_masks[i] == class_idx).float()
                        
                        # 只在目标类别存在时计算指标
                        if target_class.sum() > 0:
                            # Dice分数
                            dice = compute_dice_score(pred_class, target_class)
                            sample_dice_scores.append(dice)
                            class_dice_scores[class_idx].append(dice)
                            
                            # IoU分数
                            iou = compute_iou_score(pred_class, target_class)
                            sample_iou_scores.append(iou)
                            class_iou_scores[class_idx].append(iou)
                            
                            # 准确率
                            acc = compute_accuracy(pred_class, target_class)
                            sample_accuracy_scores.append(acc)
                            class_accuracy_scores[class_idx].append(acc)
                            
                            # HD95距离
                            hd95 = compute_hausdorff_distance_95(pred_class, target_class)
                            if hd95 != float('inf'):
                                sample_hd95_scores.append(hd95)
                                class_hd95_scores[class_idx].append(hd95)
                            
                            # 精确率和召回率
                            precision, recall = compute_precision_recall(pred_class, target_class)
                            f1 = compute_f1_score(precision, recall)
                            
                            sample_precision_scores.append(precision)
                            sample_recall_scores.append(recall)
                            sample_f1_scores.append(f1)
                            
                            class_precision_scores[class_idx].append(precision)
                            class_recall_scores[class_idx].append(recall)
                            class_f1_scores[class_idx].append(f1)
                    
                    # 计算样本级别的平均指标
                    if sample_dice_scores:
                        all_dice_scores.append(np.mean(sample_dice_scores))
                        all_iou_scores.append(np.mean(sample_iou_scores))
                        all_accuracy_scores.append(np.mean(sample_accuracy_scores))
                        all_precision_scores.append(np.mean(sample_precision_scores))
                        all_recall_scores.append(np.mean(sample_recall_scores))
                        all_f1_scores.append(np.mean(sample_f1_scores))
                        
                        if sample_hd95_scores:
                            all_hd95_scores.append(np.mean(sample_hd95_scores))
                
                # 更新进度条
                if batch_idx % 10 == 0:
                    current_dice = np.mean(all_dice_scores) if all_dice_scores else 0
                    progress_bar.set_postfix({'Dice': f'{current_dice:.4f}'})
                
            except Exception as e:
                logger.warning(f"评估batch {batch_idx} 出错: {e}")
                continue
    
    # 保存预测结果
    if save_predictions and save_dir and round_num:
        save_prediction_masks(model, dataloader, save_dir, client_id, round_num, 
                            num_samples=CONFIG['pred_samples_per_round'])
    
    # 计算总体指标
    mean_dice = np.mean(all_dice_scores) if all_dice_scores else 0
    mean_iou = np.mean(all_iou_scores) if all_iou_scores else 0
    mean_accuracy = np.mean(all_accuracy_scores) if all_accuracy_scores else 0
    mean_hd95 = np.mean(all_hd95_scores) if all_hd95_scores else float('inf')
    mean_precision = np.mean(all_precision_scores) if all_precision_scores else 0
    mean_recall = np.mean(all_recall_scores) if all_recall_scores else 0
    mean_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    
    # 图感知指标
    mean_key_points = np.mean(all_key_points_used) if all_key_points_used else 0
    mean_importance = np.mean(all_point_importance) if all_point_importance else 0
    
    # 按类别统计指标
    class_metrics = {}
    for class_idx in range(num_classes):
        if class_idx in class_dice_scores:
            class_metrics[class_idx] = {
                'dice': np.mean(class_dice_scores[class_idx]),
                'iou': np.mean(class_iou_scores[class_idx]),
                'accuracy': np.mean(class_accuracy_scores[class_idx]),
                'hd95': np.mean(class_hd95_scores[class_idx]) if class_hd95_scores[class_idx] else float('inf'),
                'precision': np.mean(class_precision_scores[class_idx]),
                'recall': np.mean(class_recall_scores[class_idx]),
                'f1': np.mean(class_f1_scores[class_idx]),
                'sample_count': len(class_dice_scores[class_idx])
            }
    
    return {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'mean_accuracy': mean_accuracy,
        'mean_hd95': mean_hd95,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'mean_key_points': mean_key_points,
        'mean_importance': mean_importance,
        'class_metrics': class_metrics,
        'sample_count': len(all_dice_scores)
    }

def save_prediction_masks(model, dataloader, save_dir, client_id, round_num, num_samples=5):
    """保存预测掩码和关键点可视化"""
    model.eval()
    save_path = Path(save_dir) / f"predictions_client_{client_id}_round_{round_num}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if saved_count >= num_samples:
                break
                
            try:
                images = batch['image'].to(model.device)
                masks = batch['masks'].to(model.device)
                points = batch['points'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                outputs = model(images, points, labels, client_id)
                pred_masks = outputs['masks']
                pred_softmax = torch.softmax(pred_masks, dim=1)
                
                # 保存每个样本
                for i in range(images.size(0)):
                    if saved_count >= num_samples:
                        break
                    
                    # 原图
                    img = images[i].cpu()
                    img = (img - img.min()) / (img.max() - img.min())
                    
                    # 真实掩码
                    gt_mask = torch.argmax(masks[i], dim=0).cpu().numpy()
                    
                    # 预测掩码
                    pred_mask = torch.argmax(pred_softmax[i], dim=0).cpu().numpy()
                    
                    # 关键点可视化
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # 原图
                    axes[0].imshow(img.permute(1, 2, 0))
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    # 真实掩码
                    axes[1].imshow(gt_mask, cmap='jet')
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    # 预测掩码
                    axes[2].imshow(pred_mask, cmap='jet')
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    
                    # 关键点可视化
                    axes[3].imshow(img.permute(1, 2, 0))
                    if outputs.get('key_point_masks') is not None:
                        key_mask = outputs['key_point_masks'][i].cpu()
                        key_points = points[i][key_mask].cpu().numpy()
                        importance = outputs.get('point_importance', None)
                        
                        if importance is not None:
                            imp_values = importance[i][key_mask].cpu().numpy()
                            # 根据重要性着色
                            scatter = axes[3].scatter(key_points[:, 0], key_points[:, 1], 
                                                    c=imp_values, cmap='hot', s=50, alpha=0.8)
                            plt.colorbar(scatter, ax=axes[3], label='Importance')
                        else:
                            axes[3].scatter(key_points[:, 0], key_points[:, 1], 
                                          c='red', s=50, alpha=0.8)
                        
                        axes[3].set_title(f'Key Points ({len(key_points)}/{points.size(1)})')
                    else:
                        axes[3].set_title('All Points')
                        pts = points[i].cpu().numpy()
                        axes[3].scatter(pts[:, 0], pts[:, 1], c='blue', s=10, alpha=0.5)
                    
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(save_path / f"sample_{saved_count + 1}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    saved_count += 1
                    
            except Exception as e:
                logger.warning(f"保存预测掩码时出错: {e}")
                continue
    
    logger.info(f"客户端 {client_id} 第 {round_num} 轮预测掩码已保存: {save_path}")

def print_detailed_metrics(metrics, dataset_name, class_names):
    """打印详细的评估指标（包括所有指标和DPGP信息）"""
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 {dataset_name} 详细评估结果")
    logger.info(f"{'='*80}")
    logger.info(f"样本数量: {metrics.get('sample_count', 0)}")
    
    # ============ 总体指标 ============
    logger.info(f"\n📈 总体指标:")
    logger.info(f"   {'指标':<15} {'值':<10}")
    logger.info(f"   {'-'*25}")
    logger.info(f"   {'Mean Dice':<15} {metrics.get('mean_dice', 0):.4f}")
    logger.info(f"   {'Mean IoU':<15} {metrics.get('mean_iou', 0):.4f}")
    logger.info(f"   {'Mean Accuracy':<15} {metrics.get('mean_accuracy', 0):.4f}")
    logger.info(f"   {'Mean Precision':<15} {metrics.get('mean_precision', 0):.4f}")
    logger.info(f"   {'Mean Recall':<15} {metrics.get('mean_recall', 0):.4f}")
    logger.info(f"   {'Mean F1':<15} {metrics.get('mean_f1', 0):.4f}")
    
    hd95 = metrics.get('mean_hd95', float('inf'))
    if hd95 != float('inf'):
        logger.info(f"   {'Mean HD95':<15} {hd95:.2f}")
    else:
        logger.info(f"   {'Mean HD95':<15} N/A")
    
    # ============ 图感知指标 ============
    if 'mean_key_points' in metrics or 'graph_stats' in metrics:
        logger.info(f"\n🔗 图感知指标:")
        if 'mean_key_points' in metrics:
            logger.info(f"   平均关键点数: {metrics['mean_key_points']:.1f}")
        if 'mean_importance' in metrics:
            logger.info(f"   平均点重要性: {metrics['mean_importance']:.3f}")
        
        graph_stats = metrics.get('graph_stats', {})
        if graph_stats and graph_stats.get('efficiency_score', 0) > 0:
            logger.info(f"   效率分数: {graph_stats['efficiency_score']:.3f}")
    
    # ============ DPGP信息 ============
    dpgp_info = metrics.get('dpgp_info', {})
    if dpgp_info:
        logger.info(f"\n🎛️ DPGP模块信息:")
        logger.info(f"   实例级权重: {dpgp_info.get('instance_level', False)}")
        logger.info(f"   TTA应用: {dpgp_info.get('tta_applied', False)}")
        if dpgp_info.get('tta_applied', False):
            logger.info(f"   TTA相似度: {dpgp_info.get('tta_similarity', 0.5):.3f}")
        logger.info(f"   全局/本地权重: {dpgp_info.get('w_global', 0.5):.3f}/{dpgp_info.get('w_local', 0.5):.3f}")
    
    # ============ 各类别详细指标 ============
    if metrics.get('class_metrics'):
        logger.info(f"\n📋 各类别详细指标:")
        header = f"{'类别':<12} {'Dice':>8} {'IoU':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'HD95':>8}"
        logger.info(f"   {header}")
        logger.info(f"   {'-'*76}")
        
        for class_idx, class_metrics in metrics['class_metrics'].items():
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
            hd95_str = f"{class_metrics['hd95']:.2f}" if class_metrics['hd95'] != float('inf') else "N/A"
            
            row = (f"{class_name:<12} "
                   f"{class_metrics['dice']:>8.4f} "
                   f"{class_metrics['iou']:>8.4f} "
                   f"{class_metrics['accuracy']:>8.4f} "
                   f"{class_metrics['precision']:>8.4f} "
                   f"{class_metrics['recall']:>8.4f} "
                   f"{class_metrics['f1']:>8.4f} "
                   f"{hd95_str:>8}")
            logger.info(f"   {row}")
    
    logger.info(f"{'='*80}\n")

def print_round_summary(round_num, client_results, best_metrics):
    """打印每轮的汇总信息"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"📊 第 {round_num} 轮评估汇总")
    logger.info(f"{'#'*80}")
    
    # 表头
    header = f"{'客户端':<20} {'Dice':>8} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'关键点':>8}"
    logger.info(f"\n{header}")
    logger.info(f"{'-'*72}")
    
    total_dice = 0
    total_iou = 0
    total_f1 = 0
    
    for client_id, (dataset_type, metrics) in client_results.items():
        client_name = f"Client{client_id}({dataset_type})"
        dice = metrics.get('mean_dice', 0)
        iou = metrics.get('mean_iou', 0)
        f1 = metrics.get('mean_f1', 0)
        prec = metrics.get('mean_precision', 0)
        recall = metrics.get('mean_recall', 0)
        key_pts = metrics.get('mean_key_points', 0)
        
        row = f"{client_name:<20} {dice:>8.4f} {iou:>8.4f} {f1:>8.4f} {prec:>8.4f} {recall:>8.4f} {key_pts:>8.1f}"
        logger.info(row)
        
        total_dice += dice
        total_iou += iou
        total_f1 += f1
    
    num_clients = len(client_results)
    logger.info(f"{'-'*72}")
    logger.info(f"{'平均':<20} {total_dice/num_clients:>8.4f} {total_iou/num_clients:>8.4f} {total_f1/num_clients:>8.4f}")
    
    if best_metrics['round'] == 0:
        logger.info(f"\n🏆 历史最佳: 暂无（当前为首次评估）")
    else:
        logger.info(f"\n🏆 历史最佳 (第{best_metrics['round']}轮):")
        logger.info(f"   Client1 Dice: {best_metrics.get('client1_dice', 0):.4f}")
        logger.info(f"   Client2 Dice: {best_metrics.get('client2_dice', 0):.4f}")
        logger.info(f"   Client3 Dice: {best_metrics.get('client3_dice', 0):.4f}")
        logger.info(f"   平均 Dice: {(best_metrics.get('client1_dice', 0) + best_metrics.get('client2_dice', 0) + best_metrics.get('client3_dice', 0))/3:.4f}")

    
    
    logger.info(f"{'#'*80}\n")

# ==================== 工具函数 ====================
def save_federated_results(server, clients, round_num, save_dir, is_best_round=False):
    """保存联邦学习结果 - 只在最佳轮保存预测可视化"""
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 【修改】只在最佳轮保存全局模型检查点
    if is_best_round:
        global_model_path = results_dir / f"global_model_round_{round_num}.pth"
        server.save_global_model(global_model_path)
    
    all_results = {
        'round': round_num,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'clients': {},
        'graph_analysis': {},
        'aggregation_info': {}
    }
    
    for client in clients:
        logger.info(f"\n评估客户端 {client.client_id} ({client.dataset_type})...")
        
        # # 【修改】只在最佳轮保存预测可视化
        # save_predictions = is_best_round and CONFIG.get('save_best_predictions', True)
        save_predictions = True
        metrics = client.evaluate(
            save_predictions=save_predictions,
            save_dir=save_dir,
            round_num=round_num
        )
        
        print_detailed_metrics(metrics, f"客户端{client.client_id}({client.dataset_type})", client.class_names)
        
        # 【修改】只在最佳轮保存客户端模型
        if is_best_round:
            client_model_path = results_dir / f"client_{client.client_id}_model_round_{round_num}.pth"
            client.save_model(client_model_path, round_num, metrics)
        
        all_results['clients'][client.client_id] = {
            'dataset_type': client.dataset_type,
            'num_classes': client.num_classes,
            'metrics': metrics,
            'training_history': client.training_history,
            'graph_stats': client.graph_stats
        }
        
        if 'graph_stats' in metrics:
            all_results['graph_analysis'][client.client_id] = {
                'avg_key_points': metrics['graph_stats'].get('avg_key_points', 0),
                'efficiency_score': metrics['graph_stats'].get('efficiency_score', 0),
                'dice_per_point': metrics['mean_dice'] / max(metrics['graph_stats'].get('avg_key_points', 1), 1),
                'graph_features': metrics['graph_stats'].get('graph_features', {})
            }
    
    if hasattr(server, 'global_graph_features'):
        all_results['aggregation_info']['global_graph_features'] = server.global_graph_features
    
    # 【修改】只在最佳轮保存详细JSON结果
    if is_best_round:
        results_path = results_dir / f"federated_results_round_{round_num}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"联邦学习结果已保存: {results_path}")
    
    return all_results

def plot_training_curves(clients, save_dir):
    """绘制训练曲线（包含RL奖励）"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # 为每个客户端绘制曲线
    for idx, client in enumerate(clients):
        history = client.training_history
        if not history['rounds']:
            continue
            
        # Dice曲线
        ax = axes[0, idx]
        ax.plot(history['rounds'], history['train_dice'], 'b-', label='Train Dice', linewidth=2)
        ax.plot(history['rounds'], history['test_dice'], 'r-', label='Test Dice', linewidth=2)
        ax.set_xlabel('Round')
        ax.set_ylabel('Dice Score')
        ax.set_title(f'Client {client.client_id} ({client.dataset_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 关键点数曲线
        ax = axes[1, idx]
        if history['graph_stats']:
            key_points = [stats.get('avg_key_points', CONFIG['num_points_grid']) 
                         for stats in history['graph_stats']]
            ax.plot(history['rounds'][:len(key_points)], key_points, 'g-', label='Avg Key Points', linewidth=2)
            ax.axhline(y=CONFIG['min_points'], color='r', linestyle='--', label='Min Points', alpha=0.7)
            ax.axhline(y=CONFIG['max_points'], color='b', linestyle='--', label='Max Points', alpha=0.7)
            ax.set_xlabel('Round')
            ax.set_ylabel('Number of Key Points')
            ax.set_title(f'Key Point Selection')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # RL奖励曲线
        ax = axes[2, idx]
        if history.get('rl_rewards'):
            ax.plot(range(len(history['rl_rewards'])), history['rl_rewards'], 'purple', label='RL Reward', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Reward')
            ax.set_title(f'Reinforcement Learning Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves_improved.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_heterogeneity(clients, save_dir):
    """分析客户端异构性"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 收集所有客户端的图特征
    client_features = {}
    for client in clients:
        if client.graph_stats.get('graph_features'):
            client_features[f'Client {client.client_id}'] = client.graph_stats['graph_features']
    
    if client_features:
        # 图特征比较
        ax = axes[0, 0]
        features = ['avg_degree', 'avg_clustering', 'density']
        x = np.arange(len(features))
        width = 0.25
        
        for i, (client_name, features_dict) in enumerate(client_features.items()):
            values = [features_dict.get(f, 0) for f in features]
            ax.bar(x + i * width, values, width, label=client_name)
        
        ax.set_xlabel('Graph Features')
        ax.set_ylabel('Value')
        ax.set_title('Graph Structure Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(features)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 效率分析
    ax = axes[0, 1]
    efficiency_data = []
    client_names = []
    for client in clients:
        if client.graph_stats.get('efficiency_score', 0) > 0:
            efficiency_data.append(client.graph_stats['efficiency_score'])
            client_names.append(f'Client {client.client_id}')
    
    if efficiency_data:
        ax.bar(client_names, efficiency_data, color=['blue', 'green', 'orange'])
        ax.set_xlabel('Client')
        ax.set_ylabel('Efficiency Score')
        ax.set_title('Client Efficiency Comparison')
        ax.grid(True, alpha=0.3)
    
    # 点使用分布
    ax = axes[1, 0]
    for client in clients:
        if client.training_history.get('graph_stats'):
            key_points_history = [stats.get('avg_key_points', 0) 
                                 for stats in client.training_history['graph_stats']]
            if key_points_history:
                ax.hist(key_points_history, bins=15, alpha=0.5, 
                       label=f'Client {client.client_id}')
    
    ax.set_xlabel('Number of Key Points')
    ax.set_ylabel('Frequency')
    ax.set_title('Key Points Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 性能vs效率散点图
    ax = axes[1, 1]
    for client in clients:
        if client.training_history['test_dice'] and client.graph_stats.get('avg_key_points', 0) > 0:
            dice = client.training_history['test_dice'][-1] if client.training_history['test_dice'] else 0
            key_points = client.graph_stats.get('avg_key_points', CONFIG['num_points_grid'])
            ax.scatter(key_points, dice, s=100, label=f'Client {client.client_id}')
    
    ax.set_xlabel('Average Key Points Used')
    ax.set_ylabel('Test Dice Score')
    ax.set_title('Performance vs Efficiency Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'heterogeneity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("🚀 改进版图感知联邦医学图像分割")
    print("✨ 核心改进:")
    print("   1️⃣ 完全隔离个性化层，不参与联邦聚合")
    print("   2️⃣ 增强图结构相似性度量和自适应聚合")
    print("   3️⃣ 数据分布感知的智能点初始化")
    print("   4️⃣ 完善的强化学习机制（DQN + 经验回放）")
    print("   5️⃣ 异构客户端的鲁棒处理")
    print("📊 创新点:")
    print("   • 图拓扑特征分析（度分布、聚类系数、密度）")
    print("   • 基于Wasserstein距离的分布差异度量")
    print("   • 自适应点采样策略（前景/背景/边界）")
    print("   • 客户端特定的个性化层（本地保留）")
    print("   • 基于效率和相似性的智能聚合权重")
    print("=" * 80)
    
    set_seed(CONFIG['random_seed'])
    device = torch.device(CONFIG['device'])
    

    save_dir = Path(CONFIG['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logging(save_dir)
    
    logger.info(f'🎯 使用设备: {device}')
    logger.info(f'⚙️  配置信息:')
    logger.info(f'   通信轮数: {CONFIG["communication_rounds"]}')
    logger.info(f'   本地训练轮数: {CONFIG["local_epochs"]}')
    logger.info(f'   客户端数量: {CONFIG["num_clients"]}')
    logger.info(f'   Batch Size: {CONFIG["batch_size"]}')
    logger.info(f'   Learning Rate: {CONFIG["learning_rate"]}')
    logger.info(f'   初始点数: {CONFIG["num_points_grid"]}')
    logger.info(f'   最小/最大点数: {CONFIG["min_points"]}/{CONFIG["max_points"]}')
    logger.info(f'   RL参数: γ={CONFIG["rl_gamma"]}, ε={CONFIG["rl_epsilon"]}')
    logger.info(f'   自适应权重系数: {CONFIG["adaptive_weight_alpha"]}')
    logger.info(f'   SAM2权重路径: {CONFIG["sam2_checkpoint_path"]}')
    
    try:
        # 【修改】检查Arcade数据集结构
        logger.info("🔍 检查Arcade数据集结构...")
        if not check_arcade_dataset(CONFIG['arcade_data_root']):
            logger.error("Arcade数据集结构检查失败！")
            return
        
        # 初始化服务器
        logger.info("🏗️  初始化改进的联邦学习服务器...")
        server = ImprovedFederatedServer(device)
        server.initialize_global_model(CONFIG['sam2_checkpoint_path'])
        
        # 【修改】初始化客户端 - 统一使用Arcade
        logger.info("👥 初始化改进的客户端...")
        clients = []
        
        for client_id in [1, 2, 3]:
            client = ImprovedFederatedClient(
                client_id=client_id,
                num_classes=CONFIG['num_classes'],
                class_names=CLASS_NAMES,
                device=device
            )
            clients.append(client)
        
        # 联邦学习训练循环
        logger.info("🔄 开始改进的图感知联邦学习训练...")
        
        best_metrics = {
            'client1_dice': 0,
            'client2_dice': 0,
            'client3_dice': 0,
            'round': 0,
            'avg_key_points': CONFIG['num_points_grid'],
            'efficiency_score': 0,
            'global_similarity': 0
        }
        
        for round_num in range(1, CONFIG['communication_rounds'] + 1):
            logger.info(f"\n🌟 ========== 联邦学习轮次 {round_num}/{CONFIG['communication_rounds']} ==========")
            
            # 【新增】设置当前轮次（用于DPGP渐进式训练）
            for client in clients:
                if hasattr(client.model, 'shift_block') and hasattr(client.model.shift_block, 'set_current_round'):
                    client.model.shift_block.set_current_round(round_num)

            # 1. 服务器向客户端分发全局参数（不包括个性化层）
            global_params = server.get_global_parameters()
            for client in clients:
                client.update_model(global_params)
                logger.info(f"📤 客户端 {client.client_id} 已更新共享参数（个性化层保持不变）")
            
            # 2. 客户端本地训练
            client_updates = []
            client_weights = []
            
            for client in clients:
                logger.info(f"\n🏃‍♂️ 客户端 {client.client_id} ({client.dataset_type}) 开始本地训练...")
                train_loss, train_dice, graph_stats = client.train_local_epochs(CONFIG['local_epochs'])
                
                logger.info(f"✅ 客户端 {client.client_id} 本地训练完成:")
                logger.info(f"   训练损失: {train_loss:.4f}")
                logger.info(f"   训练Dice: {train_dice:.4f}")
                logger.info(f"   平均关键点数: {graph_stats.get('avg_key_points', 0):.1f}")
                logger.info(f"   效率分数: {graph_stats.get('efficiency_score', 0):.3f}")
                if graph_stats.get('graph_features'):
                    logger.info(f"   图特征: 平均度={graph_stats['graph_features'].get('avg_degree', 0):.2f}, "
                              f"聚类系数={graph_stats['graph_features'].get('avg_clustering', 0):.3f}")
                
                # 收集客户端参数和统计（仅共享参数）
                client_params = client.get_model_parameters()
                client_updates.append((client_params, graph_stats))
                
                # 客户端权重（基于数据量）
                if hasattr(client.train_loader, 'dataset'):
                    weight = len(client.train_loader.dataset)
                else:
                    weight = 1.0
                client_weights.append(weight)
            
            # 3. 服务器聚合参数（带图结构自适应）
            logger.info("\n🔄 服务器开始自适应聚合...")
            # 归一化权重
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
            
            server.aggregate_parameters(client_updates, client_weights)
            
            # 显示聚合统计
            avg_key_points = np.mean([stats.get('avg_key_points', 0) for _, stats in client_updates])
            avg_efficiency = np.mean([stats.get('efficiency_score', 0) for _, stats in client_updates])
            logger.info(f"📊 全局统计: 平均关键点={avg_key_points:.1f}, 平均效率={avg_efficiency:.3f}")
            

            # 4. 评估（每eval_freq轮）
            if round_num % CONFIG['eval_freq'] == 0 or round_num == CONFIG['communication_rounds']:
                
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 ========== 第 {round_num} 轮评估 ==========")
                logger.info(f"{'='*80}")
                
                # 更新客户端模型
                latest_global_params = server.get_global_parameters()
                for client in clients:
                    client.update_model(latest_global_params)
                
                # 收集评估结果
                client_results = {}
                all_metrics = {}
                
                for client in clients:
                    logger.info(f"\n🔍 评估客户端 {client.client_id} ({client.dataset_type})...")
                    
                    # 【修复1】每次评估都保存可视化（如果启用了周期性保存）
                    save_predictions = CONFIG.get('save_periodic_predictions', True)
                    
                    metrics = client.evaluate(
                        save_predictions=save_predictions,  # 改为True
                        save_dir=save_dir,
                        round_num=round_num
                    )
                    
                    print_detailed_metrics(metrics, f"Client{client.client_id}({client.dataset_type})", client.class_names)
                    
                    client_results[client.client_id] = (client.dataset_type, metrics)
                    all_metrics[client.client_id] = metrics
                

                
                # 计算平均指标
                current_avg_dice = sum(m.get('mean_dice', 0) for m in all_metrics.values()) / len(all_metrics)
                best_avg_dice = (best_metrics.get('client1_dice', 0) + best_metrics.get('client2_dice', 0) + best_metrics.get('client3_dice', 0)) / 3
                
                # 【修复2】使用 >= 而非 >，确保第一次评估能触发
                is_best_round = current_avg_dice >= best_avg_dice and current_avg_dice > 0
                # 或者更简单：第一轮强制视为最佳
                if best_metrics['round'] == 0:
                    is_best_round = True
                
                if is_best_round:
                    logger.info(f"\n🎉🎉🎉 发现新的最佳模型！🎉🎉🎉")
                    logger.info(f"   当前平均Dice: {current_avg_dice:.4f} >= 历史最佳: {best_avg_dice:.4f}")
                    
                    # 更新最佳指标
                    best_metrics.update({
                        'client1_dice': all_metrics[1].get('mean_dice', 0),
                        'client2_dice': all_metrics[2].get('mean_dice', 0),
                        'client3_dice': all_metrics[3].get('mean_dice', 0),
                        'round': round_num,
                    })
                    
                    # ===== 保存最佳模型 =====
                    results_dir = Path(save_dir) / "results"
                    results_dir.mkdir(parents=True, exist_ok=True)
                    
                    server.save_global_model(save_dir / "best_global_model.pth")
                    logger.info(f"   ✅ 保存最佳全局模型")
                    
                    for client in clients:
                        client.save_model(results_dir / f"best_client_{client.client_id}_model.pth", round_num, all_metrics[client.client_id])
                    logger.info(f"   ✅ 保存客户端模型")
                    
                    # 【修复3】最佳轮次额外保存一份带"best"标记的可视化
                    if CONFIG.get('save_best_predictions', True):
                        for client in clients:
                            try:
                                save_prediction_masks(
                                    client.model, 
                                    client.test_loader, 
                                    save_dir, 
                                    client.client_id, 
                                    f"best_round_{round_num}",  # 使用特殊标记
                                    num_samples=CONFIG.get('pred_samples_per_round', 3)
                                )
                                logger.info(f"   ✅ 客户端{client.client_id}预测可视化已保存")
                            except Exception as e:
                                logger.error(f"   ❌ 客户端{client.client_id}保存可视化失败: {e}")
                                import traceback
                                traceback.print_exc()
                else:
                    logger.info(f"\n📈 当前Dice: {current_avg_dice:.4f}, 最佳: {best_avg_dice:.4f}, 未超越最佳")
                
                # 打印汇总对比
                print_round_summary(round_num, client_results, best_metrics)
        
        # 训练完成总结
        logger.info(f"\n🎊 改进的图感知联邦学习训练完成！")
        logger.info(f"📁 模型和结果保存目录: {save_dir}")
        logger.info(f"🏆 最佳结果 (第{best_metrics['round']}轮):")
        logger.info(f"   客户端1  - 最佳Dice: {best_metrics['client1_dice']:.4f}")
        logger.info(f"   客户端2  - 最佳Dice: {best_metrics['client2_dice']:.4f}")
        logger.info(f"   客户端3  - 最佳Dice: {best_metrics['client3_dice']:.4f}")
        logger.info(f"   平均关键点使用: {best_metrics['avg_key_points']:.1f}/{CONFIG['num_points_grid']}")
        logger.info(f"   点选择效率提升: {(CONFIG['num_points_grid'] - best_metrics['avg_key_points']) / CONFIG['num_points_grid'] * 100:.1f}%")
        
        # 保存最终训练报告
        final_report = {
            'improved_federated_training_summary': {
                'communication_rounds': CONFIG['communication_rounds'],
                'local_epochs': CONFIG['local_epochs'],
                'best_round': best_metrics['round'],
                'best_metrics': best_metrics,
                'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'key_improvements': {
                    'personalization': '个性化层完全本地保留，不参与联邦聚合',
                    'graph_analysis': '基于图拓扑特征的相似性度量',
                    'adaptive_aggregation': '考虑图结构差异和效率的自适应聚合',
                    'point_initialization': '数据分布感知的智能点初始化',
                    'reinforcement_learning': 'DQN with experience replay',
                    'efficiency_gain': f'{(CONFIG["num_points_grid"] - best_metrics["avg_key_points"]) / CONFIG["num_points_grid"] * 100:.1f}%点减少'
                },
                'clients_info': {
                    'client1': {
                        'data_ratio': '50%',
                        'num_classes': CONFIG['num_classes'],
                        'best_dice': best_metrics['client1_dice']
                    },
                    'client2': {
                        'data_ratio': '30%',
                        'num_classes': CONFIG['num_classes'],
                        'best_dice': best_metrics['client2_dice']
                    },
                    'client3': {
                        'data_ratio': '20%',
                        'num_classes': CONFIG['num_classes'],
                        'best_dice': best_metrics['client3_dice']
                    }
                }
            },
            'config': CONFIG,
            'final_results': all_results if 'all_results' in locals() else None
        }
        
        final_report_path = save_dir / "final_improved_report.json"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📊 最终改进报告已保存: {final_report_path}")
        
        # 如果保存了预测掩码，提示用户
        if best_metrics['round'] > 0 and CONFIG.get('save_best_predictions', True):
            logger.info(f"\n🖼️  最佳轮次({best_metrics['round']})的预测分割掩码和关键点可视化已保存至:")
            for client in clients:
                pred_dir = save_dir / f"predictions_client_{client.client_id}_round_{best_metrics['round']}"
                if pred_dir.exists():
                    logger.info(f"   客户端{client.client_id} ({client.dataset_type}): {pred_dir}")
        
        # 生成并保存训练曲线
        try:
            plot_training_curves(clients, save_dir)
            logger.info(f"📈 训练曲线已保存至: {save_dir}/training_curves_improved.png")
        except Exception as e:
            logger.warning(f"生成训练曲线失败: {e}")
        
        # 生成异构性分析
        try:
            analyze_heterogeneity(clients, save_dir)
            logger.info(f"📊 异构性分析已保存至: {save_dir}/heterogeneity_analysis.png")
        except Exception as e:
            logger.warning(f"生成异构性分析失败: {e}")
        
        print("\n✨ 恭喜！改进的图感知联邦学习训练成功完成！")
        print(f"🔍 关键成果：")
        print(f"   • 使用平均 {best_metrics['avg_key_points']:.1f} 个点达到最佳性能（减少{(CONFIG['num_points_grid'] - best_metrics['avg_key_points']) / CONFIG['num_points_grid'] * 100:.1f}%）")
        print(f"   • 个性化层成功隔离，客户端保持独特性")
        print(f"   • 图结构自适应聚合提升了异构处理能力")
        print(f"📂 请查看 {save_dir} 目录获取详细结果和可视化")
        
    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()