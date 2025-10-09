"""
快速测试DRL模型 - 修复版本
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# 设置随机种子确保可重现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from distance_metrics import ManhattanDistance
from drl_environment import PowerGridRLEnv
from attention_network import POMO_PowerGrid
from ppo_trainer import PPOTrainer

def quick_test():
    """快速测试DRL模型训练"""
    print("🚀 快速测试深度强化学习模型...")
    
    # 小规模配置
    config = {
        'N': 3,  # 3个发电机
        'M': 1,  # 1个变电站
        'D': 10, # 10x10网格
        'K': 2   # 2x2设备尺寸
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 打印动作空间信息
    action_grid_size = config['D'] - config['K'] + 1
    action_space_size = action_grid_size ** 2
    print(f"📐 网格大小: {config['D']}x{config['D']}")
    print(f"📐 设备尺寸: {config['K']}x{config['K']}")
    print(f"📐 动作网格: {action_grid_size}x{action_grid_size}")
    print(f"📐 动作空间大小: {action_space_size}")
    
    # 创建环境
    try:
        env = PowerGridRLEnv(
            N=config['N'],
            M=config['M'],
            D=config['D'], 
            K=config['K'],
            distance_metric=ManhattanDistance,
            device=device
        )
        print(f"✅ 环境创建成功")
        print(f"📍 发电机位置: {env.generator_positions}")
        print(f"📏 环境动作空间大小: {env.action_space_size}")
        
        # 验证动作空间大小
        if env.action_space_size != action_space_size:
            print(f"⚠️ 警告: 动作空间大小不匹配! 环境={env.action_space_size}, 计算={action_space_size}")
        
    except Exception as e:
        print(f"❌ 环境创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试环境基本功能
    print("\n🔧 测试环境基本功能...")
    try:
        state = env.reset()
        print(f"✅ 重置成功，状态形状: {state.shape}")
        
        action_mask = env.get_action_mask()
        valid_actions = torch.sum(action_mask).item()
        print(f"✅ 动作掩码形状: {action_mask.shape}")
        print(f"✅ 有效动作数量: {valid_actions}/{env.action_space_size}")
        
        if valid_actions == 0:
            print("⚠️ 警告: 没有有效动作，可能是配置问题")
            return
        
    except Exception as e:
        print(f"❌ 环境测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型 (小模型) - 传入K参数
    print("\n🧠 创建神经网络模型...")
    try:
        model = POMO_PowerGrid(
            grid_size=config['D'],
            input_dim=4,
            d_model=64,   # 非常小的模型用于快速测试
            n_head=4,
            n_layers=2,
            K=config['K']  # 传入K参数
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功，参数量: {total_params:,}")
        print(f"📏 模型动作空间大小: {model.action_space_size}")
        
        # 验证模型动作空间大小
        if model.action_space_size != env.action_space_size:
            print(f"⚠️ 警告: 模型和环境动作空间大小不匹配! 模型={model.action_space_size}, 环境={env.action_space_size}")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试模型前向传播
    print("\n🔍 测试模型推理...")
    try:
        with torch.no_grad():
            print(f"  输入状态形状: {state.unsqueeze(0).shape}")
            print(f"  输入掩码形状: {action_mask.unsqueeze(0).shape}")
            
            action, policy_logits, value, probs = model(state.unsqueeze(0), action_mask.unsqueeze(0))
            
            print(f"✅ 模型推理成功")
            print(f"  输出动作: {action.item()}")
            print(f"  策略logits形状: {policy_logits.shape}")
            print(f"  价值估计: {value.item():.3f}")
            print(f"  动作概率形状: {probs.shape}")
            print(f"  动作概率范围: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
            print(f"  动作概率总和: {probs.sum().item():.6f}")
        
    except Exception as e:
        print(f"❌ 模型推理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建训练器
    print("\n🏋️ 创建训练器...")
    try:
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=1e-3,  # 更高的学习率用于快速测试
            device=device
        )
        print(f"✅ 训练器创建成功")
        
    except Exception as e:
        print(f"❌ 训练器创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建输出目录
    output_dir = "quick_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 短时间训练
    print("\n🚂 开始训练 (这可能需要几分钟)...")
    try:
        best_reward = trainer.train(
            total_steps=3000,   # 更少的步数
            steps_per_update=256,  # 更小的批次
            eval_interval=5,
            save_interval=10,
            output_dir=output_dir
        )
        
        print(f"✅ 训练完成! 最佳奖励: {best_reward:.2f}")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试训练好的模型
    print("\n🎯 测试训练后的模型...")
    try:
        test_reward = trainer.evaluate(num_episodes=3, render=False)
        print(f"✅ 测试奖励: {test_reward:.2f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 演示一个完整episode
    print("\n🎬 演示完整求解过程...")
    try:
        env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"初始状态: {len(env.generator_positions)}个发电机, 0个变电站")
        
        model.eval()
        while not env.done and step_count < 10:  # 最多10步
            state_tensor = env.state.unsqueeze(0)
            action_mask = env.get_action_mask().unsqueeze(0)
            
            with torch.no_grad():
                action, policy_logits, value, probs = model(state_tensor, action_mask, deterministic=True)
            
            state, reward, done, info = env.step(action.item())
            episode_reward += reward
            step_count += 1
            
            action_x, action_y = env._action_to_position(action.item())
            print(f"步骤 {step_count}: 在位置({action_x}, {action_y})放置变电站, 奖励: {reward:.2f}")
            
            if info.get('invalid_action', False):
                print(f"⚠️ 警告: 无效动作!")
                break
        
        print(f"\n🏁 Episode完成!")
        print(f"  成功完成: {env.done and len(env.substation_positions) == env.M}")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  总步数: {step_count}")
        print(f"  最终变电站位置: {env.substation_positions}")
        
        # 计算最终距离
        if env.substation_positions:
            total_distance = 0
            print(f"\n📏 距离计算:")
            for i, (gx, gy) in enumerate(env.generator_positions):
                min_dist = float('inf')
                closest_sub = None
                for sx, sy in env.substation_positions:
                    dist = ManhattanDistance.calculate((gx, gy), (sx, sy))
                    if dist < min_dist:
                        min_dist = dist
                        closest_sub = (sx, sy)
                total_distance += min_dist
                print(f"  发电机G{i+1}({gx},{gy}) -> 变电站{closest_sub}: {min_dist:.1f}")
            
            print(f"  总曼哈顿距离: {total_distance:.2f}")
            print(f"  平均距离: {total_distance/len(env.generator_positions):.2f}")
        
        # 保存最终状态图
        try:
            env.render(save_path=os.path.join(output_dir, "final_solution.png"))
            print(f"✅ 最终解决方案已保存到: {output_dir}/final_solution.png")
        except Exception as e:
            print(f"⚠️ 保存图片失败: {str(e)}")
        
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 快速测试完成! 所有功能正常工作")
    print(f"📁 结果保存在: {output_dir}/")
    print(f"📊 训练曲线: {output_dir}/training_progress.png")
    print(f"🎯 最终解决方案: {output_dir}/final_solution.png")

if __name__ == "__main__":
    quick_test()