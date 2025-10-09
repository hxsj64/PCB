"""
深度强化学习电力网格优化 - 主训练脚本
"""
import torch
import numpy as np
import random
import argparse
import os
from datetime import datetime

from distance_metrics import DISTANCE_METRICS
from drl_environment import PowerGridRLEnv
from attention_network import POMO_PowerGrid
from ppo_trainer import PPOTrainer

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_config():
    """创建实验配置"""
    configs = [
        {"N": 5, "M": 1, "D": 10, "K": 2, "name": "Small_Grid"},
        {"N": 5, "M": 2, "D": 25, "K": 2, "name": "Medium_Grid"}, 
        {"N": 5, "M": 2, "D": 50, "K": 2, "name": "Large_Grid"},
        {"N": 10, "M": 3, "D": 25, "K": 2, "name": "Complex_Grid"},
        {"N": 10, "M": 4, "D": 50, "K": 2, "name": "Very_Large_Grid"}
    ]
    return configs

def train_single_config(config, distance_name, args):
    """训练单个配置"""
    print(f"\n{'='*60}")
    print(f"训练配置: {config['name']} - {distance_name} 距离")
    print(f"参数: N={config['N']}, M={config['M']}, D={config['D']}, K={config['K']}")
    print(f"{'='*60}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    distance_metric = DISTANCE_METRICS[distance_name]
    env = PowerGridRLEnv(
        N=config['N'],
        M=config['M'], 
        D=config['D'],
        K=config['K'],
        distance_metric=distance_metric,
        device=device
    )
    
    print(f"环境创建成功: {len(env.generator_positions)} 个发电机")
    
    # 创建模型
    model = POMO_PowerGrid(
        grid_size=config['D'],
        input_dim=4,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    
    # 创建训练器
    trainer = PPOTrainer(
        model=model,
        env=env,
        lr=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        device=device
    )
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{config['name']}_{distance_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_info = {
        'config': config,
        'distance_metric': distance_name,
        'model_params': {
            'd_model': args.d_model,
            'n_head': args.n_head,
            'n_layers': args.n_layers
        },
        'training_params': {
            'lr': args.lr,
            'total_steps': args.total_steps,
            'steps_per_update': args.steps_per_update,
            'gamma': args.gamma,
            'clip_epsilon': args.clip_epsilon
        },
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # 开始训练
    try:
        best_reward = trainer.train(
            total_steps=args.total_steps,
            steps_per_update=args.steps_per_update,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            output_dir=output_dir
        )
        
        # 最终评估
        print("\n进行最终评估...")
        final_eval_reward = trainer.evaluate(
            num_episodes=20, 
            render=True, 
            save_path=os.path.join(output_dir, "final_evaluation")
        )
        
        print(f"最终评估奖励: {final_eval_reward:.2f}")
        
        return {
            'config': config,
            'distance_metric': distance_name,
            'best_reward': best_reward,
            'final_eval_reward': final_eval_reward,
            'output_dir': output_dir,
            'success': True
        }
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        return {
            'config': config,
            'distance_metric': distance_name,
            'error': str(e),
            'success': False
        }

def main():
    parser = argparse.ArgumentParser(description='深度强化学习电力网格优化')
    
    # 环境参数
    parser.add_argument('--configs', type=str, nargs='+', default=['Small_Grid', 'Medium_Grid'], 
                       help='要训练的配置')
    parser.add_argument('--distance_metrics', type=str, nargs='+', default=['manhattan'], 
                       help='距离度量类型')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_head', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='Transformer层数')
    
    # 训练参数
    parser.add_argument('--total_steps', type=int, default=500000, help='总训练步数')
    parser.add_argument('--steps_per_update', type=int, default=2048, help='每次更新的步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO裁剪参数')
    parser.add_argument('--value_coef', type=float, default=0.5, help='价值损失系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵损失系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO更新轮数')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='小批次大小')
    
    # 评估和保存
    parser.add_argument('--eval_interval', type=int, default=50, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='保存间隔')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU')
    parser.add_argument('--output_dir', type=str, default='drl_experiments', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取实验配置
    all_configs = create_experiment_config()
    configs_to_run = [c for c in all_configs if c['name'] in args.configs]
    
    if not configs_to_run:
        print("错误: 没有找到匹配的配置")
        return
    
    # 验证距离度量
    available_metrics = list(DISTANCE_METRICS.keys())
    for metric in args.distance_metrics:
        if metric not in available_metrics:
            print(f"错误: 未知的距离度量 '{metric}', 可用的有: {available_metrics}")
            return
    
    print(f"开始深度强化学习训练...")
    print(f"配置数量: {len(configs_to_run)}")
    print(f"距离度量: {args.distance_metrics}")
    print(f"总实验数: {len(configs_to_run) * len(args.distance_metrics)}")
    
    # 创建主输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行所有实验
    results = []
    total_experiments = len(configs_to_run) * len(args.distance_metrics)
    
    for i, config in enumerate(configs_to_run):
        for j, distance_name in enumerate(args.distance_metrics):
            experiment_num = i * len(args.distance_metrics) + j + 1
            print(f"\n进度: [{experiment_num}/{total_experiments}]")
            
            result = train_single_config(config, distance_name, args)
            results.append(result)
    
    # 生成总结报告
    generate_experiment_summary(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("所有实验完成!")
    print(f"结果保存在: {args.output_dir}")
    print(f"{'='*60}")

def generate_experiment_summary(results, output_dir):
    """生成实验总结报告"""
    summary_path = os.path.join(output_dir, "experiment_summary.txt")
    
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("深度强化学习电力网格优化 - 实验总结\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总实验数: {len(results)}\n")
        f.write(f"成功实验: {len(successful_results)}\n")
        f.write(f"失败实验: {len(failed_results)}\n\n")
        
        if successful_results:
            f.write("成功实验结果:\n")
            f.write("-" * 40 + "\n")
            for result in successful_results:
                config = result['config']
                f.write(f"配置: {config['name']} ({config['N']}-{config['M']}-{config['D']})\n")
                f.write(f"距离度量: {result['distance_metric']}\n")
                f.write(f"最佳奖励: {result['best_reward']:.2f}\n")
                f.write(f"最终评估: {result['final_eval_reward']:.2f}\n")
                f.write(f"输出目录: {result['output_dir']}\n\n")
        
        if failed_results:
            f.write("失败实验:\n")
            f.write("-" * 40 + "\n")
            for result in failed_results:
                config = result['config']
                f.write(f"配置: {config['name']} - {result['distance_metric']}\n")
                f.write(f"错误: {result['error']}\n\n")
    
    print(f"实验总结保存到: {summary_path}")

if __name__ == "__main__":
    main()