"""
优化的因子挖掘训练脚本
"""
import datetime
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from base import GplearnFactor

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedFactorMiner:
    """优化的因子挖掘器"""
    
    def __init__(self):
        self.best_model = None
        self.best_score = float('inf')
        self.training_history = []
        
    def create_factor_model(self, config: dict):
        """创建因子模型"""
        return GplearnFactor(
            data_source=config.get('data_source', 'day'),
            use_custom_functions=config.get('use_custom_functions', True),
            window_sizes=config.get('window_sizes', [3, 5, 10, 20]),
            **config.get('model_params', {})
        )
    
    def optimize_hyperparameters(self, data_start_date, data_end_date, n_trials=5):
        """超参数优化"""
        logger.info("开始超参数优化...")
        
        # 定义不同的配置
        configs = [
            # 配置1: 保守配置
            {
                'data_source': 'day',
                'use_custom_functions': True,
                'window_sizes': [3, 5, 10],
                'model_params': {
                    'population_size': 300,
                    'generations': 20,
                    'tournament_size': 15,
                    'parsimony_coefficient': 0.01,
                    'p_crossover': 0.7,
                    'p_subtree_mutation': 0.1,
                    'p_hoist_mutation': 0.05,
                    'p_point_mutation': 0.1,
                    'verbose': 1
                }
            },
            # 配置2: 激进配置
            {
                'data_source': 'day',
                'use_custom_functions': True,
                'window_sizes': [5, 10, 20],
                'model_params': {
                    'population_size': 800,
                    'generations': 40,
                    'tournament_size': 25,
                    'parsimony_coefficient': 0.005,
                    'p_crossover': 0.8,
                    'p_subtree_mutation': 0.08,
                    'p_hoist_mutation': 0.04,
                    'p_point_mutation': 0.08,
                    'verbose': 1
                }
            },
            # 配置3: 平衡配置
            {
                'data_source': 'day',
                'use_custom_functions': True,
                'window_sizes': [3, 5, 10, 15],
                'model_params': {
                    'population_size': 500,
                    'generations': 30,
                    'tournament_size': 20,
                    'parsimony_coefficient': 0.008,
                    'p_crossover': 0.75,
                    'p_subtree_mutation': 0.1,
                    'p_hoist_mutation': 0.05,
                    'p_point_mutation': 0.1,
                    'verbose': 1
                }
            }
        ]
        
        best_config = None
        
        for i, config in enumerate(configs[:n_trials]):
            logger.info(f"测试配置 {i+1}/{len(configs)}")
            
            try:
                # 创建模型
                factor = self.create_factor_model(config)
                
                # 加载数据
                factor.load_data(data_start_date, data_end_date)
                
                # 数据预处理
                factor.preprocess(target_col="target_feature")
                
                # 分割训练集和验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    factor.X, factor.y, test_size=0.2, random_state=42
                )
                
                # 训练模型
                factor.model.fit(X_train, y_train)
                
                # 评估模型
                y_pred = factor.model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                logger.info(f"配置 {i+1} 结果: MSE={mse:.6f}, R²={r2:.6f}")
                
                # 记录结果
                self.training_history.append({
                    'config': config,
                    'mse': mse,
                    'r2': r2,
                    'expression': str(factor.model._program)
                })
                
                # 更新最佳模型
                if mse < self.best_score:
                    self.best_score = mse
                    self.best_model = factor
                    best_config = config
                    logger.info(f"发现更好的配置，MSE: {mse:.6f}")
                    
            except Exception as e:
                logger.error(f"配置 {i+1} 训练失败: {e}")
                continue
        
        logger.info(f"超参数优化完成，最佳MSE: {self.best_score:.6f}")
        return best_config
    
    def train_final_model(self, data_start_date, data_end_date, config=None):
        """训练最终模型"""
        if config is None:
            # 使用最佳配置
            config = {
                'data_source': 'day',
                'use_custom_functions': True,
                'window_sizes': [3, 5, 10, 15],
                'model_params': {
                    'population_size': 1000,
                    'generations': 50,
                    'tournament_size': 25,
                    'parsimony_coefficient': 0.01,
                    'p_crossover': 0.7,
                    'p_subtree_mutation': 0.1,
                    'p_hoist_mutation': 0.05,
                    'p_point_mutation': 0.1,
                    'verbose': 1
                }
            }
        
        logger.info("开始训练最终模型...")
        
        # 创建模型
        factor = self.create_factor_model(config)
        
        # 加载数据
        factor.load_data(data_start_date, data_end_date)
        
        # 数据预处理
        factor.preprocess(target_col="target_feature")
        
        # 训练模型
        factor.train()
        
        # 保存结果
        output_dir = f"results_optimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        factor.save_results(output_dir)
        
        # 保存训练历史
        if self.training_history:
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(f"{output_dir}/training_history.csv", index=False)
        
        return factor
    
    def analyze_results(self, factor):
        """分析训练结果"""
        logger.info("分析训练结果...")
        
        # 获取预测值
        y_pred = factor.transform()
        y_true = factor.y
        
        # 计算各种指标
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # 计算IC (Information Coefficient)
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        # 计算Rank IC
        rank_ic = np.corrcoef(
            pd.Series(y_true).rank(), 
            pd.Series(y_pred).rank()
        )[0, 1]
        
        logger.info(f"模型性能指标:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  Correlation: {correlation:.6f}")
        logger.info(f"  IC: {ic:.6f}")
        logger.info(f"  Rank IC: {rank_ic:.6f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'correlation': correlation,
            'ic': ic,
            'rank_ic': rank_ic
        }

def main():
    """主函数"""
    logger.info("开始优化的因子挖掘训练")
    
    # 创建优化器
    miner = OptimizedFactorMiner()
    
    # 设置数据时间范围
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 31)  # 使用一年的数据
    
    try:
        # 超参数优化
        best_config = miner.optimize_hyperparameters(start_date, end_date, n_trials=3)
        
        # 训练最终模型
        final_factor = miner.train_final_model(start_date, end_date, best_config)
        
        # 分析结果
        metrics = miner.analyze_results(final_factor)
        
        # 打印最终表达式
        expression = final_factor.get_factor_expression()
        logger.info(f"最终因子表达式: {expression}")
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise

if __name__ == "__main__":
    main() 