"""
PPO训练器 - 修复设备不匹配问题
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

class PPOTrainer:
    def __init__(self, 
                 model, 
                 env, 
                 lr=3e-4,
                 gamma=0.99, 
                 clip_epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=4,
                 mini_batch_size=64,
                 device='cpu'):
        
        self.model = model
        self.env = env
        self.device = device
        
        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        
        # 经验缓存
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def collect_rollout(self, num_steps):
        """收集rollout数据 - 修复设备不匹配问题"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # 确保state在正确的设备上
            if isinstance(state, torch.Tensor):
                # 如果state已经是tensor，确保它在正确的设备上
                if state.device != self.device:
                    state = state.to(self.device)
                state_tensor = state.unsqueeze(0)
            else:
                # 如果state是numpy数组，直接在目标设备上创建tensor
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # 获取有效动作掩码，确保在正确设备上
            action_mask = self.env.get_action_mask()
            if action_mask.device != self.device:
                action_mask = action_mask.to(self.device)
            action_mask = action_mask.unsqueeze(0)
            
            with torch.no_grad():
                try:
                    action, policy_logits, value, _ = self.model(state_tensor, action_mask)
                    
                    # 计算log概率
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    log_prob = dist.log_prob(action)
                    
                except Exception as e:
                    print(f"❌ 模型前向传播失败: {str(e)}")
                    print(f"   state_tensor设备: {state_tensor.device}, 形状: {state_tensor.shape}")
                    print(f"   action_mask设备: {action_mask.device}, 形状: {action_mask.shape}")
                    raise e
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action.item())
            
            # 存储经验 - 将tensor转换为CPU numpy用于存储
            if isinstance(state, torch.Tensor):
                state_to_store = state.cpu().numpy()
            else:
                state_to_store = state
                
            self.states.append(state_to_store)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
            self.rewards.append(reward)
            self.dones.append(done)
            self.action_masks.append(action_mask.squeeze(0).cpu())  # 存储到CPU
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                # 记录episode统计
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # 重置环境
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
        
        return len(self.states)
    
    def compute_returns_and_advantages(self):
        """计算回报和优势函数"""
        returns = []
        advantages = []
        
        # 计算GAE (Generalized Advantage Estimation)
        gae = 0
        next_value = 0
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_value = 0
                gae = 0
            
            delta = self.rewards[i] + self.gamma * next_value - self.values[i]
            gae = delta + self.gamma * 0.95 * gae  # GAE参数lambda=0.95
            
            returns.insert(0, gae + self.values[i])
            advantages.insert(0, gae)
            next_value = self.values[i]
        
        # 标准化优势
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        return returns, advantages
    
    def update_policy(self):
        """更新策略网络"""
        if len(self.states) == 0:
            return
        
        # 计算回报和优势
        returns, advantages = self.compute_returns_and_advantages()
        
        # 转换数据格式 - 确保所有数据都在正确的设备上
        states = []
        for s in self.states:
            if isinstance(s, np.ndarray):
                states.append(torch.tensor(s, dtype=torch.float32, device=self.device))
            else:
                states.append(torch.tensor(s, dtype=torch.float32, device=self.device))
        states = torch.stack(states)
        
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        
        # 确保action_masks在正确设备上
        action_masks = []
        for mask in self.action_masks:
            if isinstance(mask, torch.Tensor):
                action_masks.append(mask.to(self.device))
            else:
                action_masks.append(torch.tensor(mask, dtype=torch.bool, device=self.device))
        action_masks = torch.stack(action_masks)
        
        # PPO更新
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                
                # 前向传播
                try:
                    log_probs, values, entropy = self.model.evaluate_actions(
                        batch_states, batch_actions, batch_action_masks
                    )
                except Exception as e:
                    print(f"❌ 模型评估失败: {str(e)}")
                    print(f"   batch_states设备: {batch_states.device}, 形状: {batch_states.shape}")
                    print(f"   batch_action_masks设备: {batch_action_masks.device}, 形状: {batch_action_masks.shape}")
                    raise e
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs.detach())
                
                # PPO裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_pred_clipped = old_values[batch_indices] + torch.clamp(
                    values.squeeze() - old_values[batch_indices],
                    -self.clip_epsilon, self.clip_epsilon
                )
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_losses_unclipped = (values.squeeze() - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses_clipped, value_losses_unclipped).mean()
                
                # 熵损失
                entropy_loss = entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录统计
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
        
        # 更新训练历史
        if policy_losses:  # 确保有数据
            self.training_history['policy_loss'].append(np.mean(policy_losses))
            self.training_history['value_loss'].append(np.mean(value_losses))
            self.training_history['entropy'].append(np.mean(entropies))
            self.training_history['total_loss'].append(
                self.training_history['policy_loss'][-1] + 
                self.training_history['value_loss'][-1] - 
                self.training_history['entropy'][-1]
            )
        
        # 清空缓存
        self.clear_buffer()
    
    def clear_buffer(self):
        """清空经验缓存"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
    
    def train(self, total_steps, steps_per_update=2048, eval_interval=50, save_interval=100, output_dir="drl_training"):
        """主训练循环"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始PPO训练: 总步数={total_steps}, 每次更新步数={steps_per_update}")
        print(f"设备: {self.device}")
        
        step_count = 0
        update_count = 0
        best_reward = -float('inf')
        
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            while step_count < total_steps:
                try:
                    # 收集经验
                    collected_steps = self.collect_rollout(steps_per_update)
                    step_count += collected_steps
                    
                    # 更新策略
                    self.update_policy()
                    update_count += 1
                    
                    # 记录统计信息
                    if self.episode_rewards:
                        avg_reward = np.mean(list(self.episode_rewards)[-10:])  # 最近10个episode
                        avg_length = np.mean(list(self.episode_lengths)[-10:])
                        
                        self.training_history['episode_rewards'].append(avg_reward)
                        self.training_history['episode_lengths'].append(avg_length)
                        
                        # 更新最佳模型
                        if avg_reward > best_reward:
                            best_reward = avg_reward
                            self.save_model(os.path.join(output_dir, "best_model.pth"))
                        
                        pbar.set_postfix({
                            'Avg Reward': f'{avg_reward:.2f}',
                            'Avg Length': f'{avg_length:.1f}',
                            'Updates': update_count
                        })
                    
                    # 定期评估
                    if update_count % eval_interval == 0:
                        try:
                            eval_reward = self.evaluate(num_episodes=3)  # 减少评估episode数
                            print(f"\nEvaluation after {update_count} updates: {eval_reward:.2f}")
                        except Exception as e:
                            print(f"\n⚠️ 评估失败: {str(e)}")
                    
                    # 定期保存
                    if update_count % save_interval == 0:
                        self.save_model(os.path.join(output_dir, f"checkpoint_{update_count}.pth"))
                        self.plot_training_progress(os.path.join(output_dir, "training_progress.png"))
                    
                    pbar.update(collected_steps)
                    
                except Exception as e:
                    print(f"\n❌ 训练步骤失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
        
        # 保存最终模型和训练历史
        try:
            self.save_model(os.path.join(output_dir, "final_model.pth"))
            self.save_training_history(os.path.join(output_dir, "training_history.json"))
            self.plot_training_progress(os.path.join(output_dir, "final_training_progress.png"))
        except Exception as e:
            print(f"⚠️ 保存最终结果失败: {str(e)}")
        
        print(f"训练完成! 最佳奖励: {best_reward:.2f}")
        return best_reward
    
    def evaluate(self, num_episodes=10, render=False, save_path=None):
        """评估当前策略"""
        self.model.eval()
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while not self.env.done and episode_length < 50:  # 添加最大步数限制
                # 确保state在正确设备上
                if isinstance(state, torch.Tensor):
                    if state.device != self.device:
                        state = state.to(self.device)
                    state_tensor = state.unsqueeze(0)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                action_mask = self.env.get_action_mask()
                if action_mask.device != self.device:
                    action_mask = action_mask.to(self.device)
                action_mask = action_mask.unsqueeze(0)
                
                with torch.no_grad():
                    action, _, _, _ = self.model(state_tensor, action_mask, deterministic=True)
                
                state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward
                episode_length += 1
                
                if render and episode == 0:  # 只渲染第一个episode
                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                        self.env.render(os.path.join(save_path, f"eval_step_{episode_length}.png"))
                    else:
                        self.env.render()
            
            episode_rewards.append(episode_reward)
        
        self.model.train()
        return np.mean(episode_rewards)
    
    def save_model(self, path):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history
            }, path)
        except Exception as e:
            print(f"⚠️ 保存模型失败: {str(e)}")
    
    def load_model(self, path):
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
        except Exception as e:
            print(f"⚠️ 加载模型失败: {str(e)}")
    
    def save_training_history(self, path):
        """保存训练历史"""
        try:
            with open(path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ 保存训练历史失败: {str(e)}")
    
    def plot_training_progress(self, save_path):
        """绘制训练进度"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 奖励曲线
            if self.training_history['episode_rewards']:
                ax1.plot(self.training_history['episode_rewards'])
                ax1.set_title('Episode Rewards')
                ax1.set_xlabel('Update')
                ax1.set_ylabel('Average Reward')
                ax1.grid(True, alpha=0.3)
            
            # Episode长度
            if self.training_history['episode_lengths']:
                ax2.plot(self.training_history['episode_lengths'])
                ax2.set_title('Episode Lengths')
                ax2.set_xlabel('Update')
                ax2.set_ylabel('Average Length')
                ax2.grid(True, alpha=0.3)
            
            # 损失曲线
            if self.training_history['policy_loss']:
                ax3.plot(self.training_history['policy_loss'], label='Policy Loss')
                ax3.plot(self.training_history['value_loss'], label='Value Loss')
                ax3.plot(self.training_history['total_loss'], label='Total Loss')
                ax3.set_title('Training Losses')
                ax3.set_xlabel('Update')
                ax3.set_ylabel('Loss')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 熵
            if self.training_history['entropy']:
                ax4.plot(self.training_history['entropy'])
                ax4.set_title('Policy Entropy')
                ax4.set_xlabel('Update')
                ax4.set_ylabel('Entropy')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 绘制训练进度失败: {str(e)}")