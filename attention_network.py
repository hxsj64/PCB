"""
多头注意力网络 - 修复动作空间维度问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear transformations and split into heads
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attention_output)

class SpatialEncoder(nn.Module):
    """空间编码器 - 处理2D网格状态"""
    
    def __init__(self, input_dim=4, hidden_dim=128, n_layers=4):
        super(SpatialEncoder, self).__init__()
        
        # CNN特征提取
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64), 
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(hidden_dim)
        ])
        
        # 残差连接
        self.residual_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(n_layers)
        ])
        
        self.residual_norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        # CNN特征提取
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))
        
        # 残差块
        for res_conv, res_bn in zip(self.residual_layers, self.residual_norms):
            residual = x
            x = F.relu(res_bn(res_conv(x)))
            x = x + residual  # 残差连接
        
        return x

class PositionEncoder(nn.Module):
    """位置编码器 - 为每个网格位置添加位置信息"""
    
    def __init__(self, d_model, max_len=2500):  # 50x50网格
        super(PositionEncoder, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x, grid_size):
        batch_size, seq_len, d_model = x.size()
        
        # 为每个位置添加位置编码
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + pos_encoding

class PowerGridTransformer(nn.Module):
    """电力网格Transformer网络 - 修复动作空间维度问题"""
    
    def __init__(self, grid_size=50, input_dim=4, d_model=256, n_head=8, n_layers=6, K=2):
        super(PowerGridTransformer, self).__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        self.K = K  # 设备尺寸
        
        # 计算实际的动作空间大小
        self.action_grid_size = grid_size - K + 1
        self.action_space_size = self.action_grid_size ** 2
        
        print(f"🔧 网络初始化: grid_size={grid_size}, K={K}, action_grid_size={self.action_grid_size}, action_space_size={self.action_space_size}")
        
        # 空间编码器
        self.spatial_encoder = SpatialEncoder(input_dim, d_model//2)
        
        # 将2D特征转换为1D序列
        self.spatial_to_seq = nn.Conv2d(d_model//2, d_model, kernel_size=1)
        
        # 位置编码
        self.position_encoder = PositionEncoder(d_model)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model*4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(d_model//2, d_model)
        
        # 策略头 (Actor) - 输出到动作空间
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        
        # 价值头 (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        
        # 动作空间映射层 - 将网格特征映射到动作空间
        self.action_mapper = nn.Sequential(
            nn.Conv2d(d_model, d_model//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model//2, 1, kernel_size=1),
            nn.Flatten()
        )
        
    def forward(self, state, action_mask=None):
        batch_size = state.size(0)
        
        # 空间特征提取
        spatial_features = self.spatial_encoder(state)  # [B, d_model//2, H, W]
        
        # 转换为序列格式
        seq_features = self.spatial_to_seq(spatial_features)  # [B, d_model, H, W]
        
        # 直接从空间特征计算动作概率
        # 首先将特征图调整到动作空间大小
        if seq_features.size(-1) != self.action_grid_size:
            # 使用自适应池化将特征图调整到正确大小
            seq_features_resized = F.adaptive_avg_pool2d(seq_features, (self.action_grid_size, self.action_grid_size))
        else:
            seq_features_resized = seq_features
        
        # 计算策略logits
        policy_logits = self.action_mapper(seq_features_resized)  # [B, action_space_size]
        
        # 确保输出维度正确
        expected_size = self.action_space_size
        if policy_logits.size(1) != expected_size:
            print(f"⚠️ 警告: 策略输出维度 {policy_logits.size(1)} != 期望维度 {expected_size}")
            # 使用线性层调整到正确维度
            if not hasattr(self, 'size_adjuster'):
                self.size_adjuster = nn.Linear(policy_logits.size(1), expected_size).to(policy_logits.device)
            policy_logits = self.size_adjuster(policy_logits)
        
        # 应用动作掩码
        if action_mask is not None:
            if action_mask.size(1) != policy_logits.size(1):
                print(f"⚠️ 掩码维度不匹配: mask={action_mask.shape}, logits={policy_logits.shape}")
                # 调整掩码大小
                if action_mask.size(1) > policy_logits.size(1):
                    action_mask = action_mask[:, :policy_logits.size(1)]
                else:
                    # 扩展掩码
                    padding = policy_logits.size(1) - action_mask.size(1)
                    action_mask = F.pad(action_mask, (0, padding), value=False)
            
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
        
        # 计算价值 - 使用全局特征
        seq_features_flat = seq_features.flatten(2).transpose(1, 2)  # [B, H*W, d_model]
        
        # 位置编码
        seq_features_encoded = self.position_encoder(seq_features_flat, self.grid_size)
        
        # 全局上下文
        global_context = self.global_pool(spatial_features).flatten(1)  # [B, d_model//2]
        global_context = self.global_fc(global_context).unsqueeze(1)  # [B, 1, d_model]
        
        # 添加全局上下文到序列
        seq_features_with_global = torch.cat([global_context, seq_features_encoded], dim=1)
        
        # Transformer处理
        for transformer_layer in self.transformer_layers:
            seq_features_with_global = transformer_layer(seq_features_with_global)
        
        # 提取全局特征用于价值估计
        global_features = seq_features_with_global[:, 0, :]  # [B, d_model]
        
        # 价值输出
        value = self.value_head(global_features)  # [B, 1]
        
        return policy_logits, value

class POMO_PowerGrid(nn.Module):
    """POMO风格的多起始点优化网络 - 修复维度问题"""
    
    def __init__(self, grid_size=50, input_dim=4, d_model=256, n_head=8, n_layers=6, K=2):
        super(POMO_PowerGrid, self).__init__()
        
        self.transformer = PowerGridTransformer(
            grid_size=grid_size,
            input_dim=input_dim, 
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            K=K
        )
        
        self.grid_size = grid_size
        self.K = K
        self.action_space_size = (grid_size - K + 1) ** 2
        
    def forward(self, state, action_mask=None, deterministic=False):
        """
        前向传播
        Args:
            state: 环境状态 [B, C, H, W]
            action_mask: 有效动作掩码 [B, action_space]
            deterministic: 是否使用确定性策略
        """
        policy_logits, value = self.transformer(state, action_mask)
        
        # 策略分布
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        if deterministic:
            # 确定性动作 (贪心)
            actions = torch.argmax(policy_probs, dim=-1)
        else:
            # 随机采样
            dist = torch.distributions.Categorical(policy_probs)
            actions = dist.sample()
        
        return actions, policy_logits, value, policy_probs
    
    def evaluate_actions(self, state, actions, action_mask=None):
        """评估给定动作的价值和概率"""
        policy_logits, value = self.transformer(state, action_mask)
        
        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value, entropy