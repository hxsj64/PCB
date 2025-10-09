"""
专门的GIF生成器 - 从已有模型生成GIF
"""
import os
import glob
from drl_visualizer import DRLTrainingVisualizer
from distance_metrics import DISTANCE_METRICS
import json
import torch

class GIFGenerator:
    def __init__(self, experiments_dir):
        self.experiments_dir = experiments_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def find_trained_models(self):
        """查找所有训练好的模型"""
        models = []
        
        for root, dirs, files in os.walk(self.experiments_dir):
            if 'best_model.pth' in files and 'experiment_info.json' in files:
                info_path = os.path.join(root, 'experiment_info.json')
                model_path = os.path.join(root, 'best_model.pth')
                
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    models.append({
                        'model_path': model_path,
                        'info': info,
                        'exp_dir': root
                    })
                except:
                    continue
        
        return models
    
    def generate_all_gifs(self):
        """为所有找到的模型生成GIF"""
        models = self.find_trained_models()
        
        print(f"🎬 找到 {len(models)} 个训练好的模型")
        
        for i, model_info in enumerate(models):
            print(f"\n📽️ 生成GIF [{i+1}/{len(models)}]")
            
            try:
                info = model_info['info']
                config = info['config']
                distance_name = info['distance_metric']
                
                print(f"   配置: {config['name']} - {distance_name}")
                
                # 创建可视化器
                visualizer = DRLTrainingVisualizer(
                    model_path=model_info['model_path'],
                    env_config=config,
                    distance_metric=DISTANCE_METRICS[distance_name],
                    device=self.device
                )
                
                # 生成决策过程可视化
                gif_dir = os.path.join(model_info['exp_dir'], "solution_gif")
                os.makedirs(gif_dir, exist_ok=True)
                
                episode_data = visualizer.generate_decision_making_visualization(gif_dir)
                
                # 复制GIF到主目录
                source_gif = os.path.join(gif_dir, "decision_making_process.gif")
                if os.path.exists(source_gif):
                    target_gif = os.path.join(self.experiments_dir, f"{config['name']}_{distance_name}_solution.gif")
                    import shutil
                    shutil.copy2(source_gif, target_gif)
                    print(f"   ✅ GIF已保存: {target_gif}")
                else:
                    print(f"   ❌ GIF生成失败")
                
            except Exception as e:
                print(f"   ❌ 处理失败: {str(e)}")

def generate_gifs_from_existing_models(experiments_dir):
    """从现有模型生成GIF"""
    generator = GIFGenerator(experiments_dir)
    generator.generate_all_gifs()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        experiments_dir = sys.argv[1]
    else:
        experiments_dir = "comprehensive_experiments_with_gif"
    
    generate_gifs_from_existing_models(experiments_dir)