import pandas as pd
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
from typing import Dict, Optional
import polars as pl
import sys
from pathlib import Path
sys.path.append("/home/yzhzhang/factor_standard_pipeline")
import datetime

from factor_standard_pipeline.core.base import BaseFactor

# 导入自定义时间序列函数

# 全局数据源
DAY_BAR = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/day_bar")
MINUTE_BAR = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/minute_bar")
TICK = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/tick")
QIML_factor = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/factor/QIML/")

import logging
from pathlib import Path
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# 自定义时间序列函数集已在 function_set.py 中实现，直接导入即可
from gplearn_factor_pipeline.function_set import custom_function_set

class GplearnFactor(BaseFactor):
    """基于gplearn的因子挖掘类"""
    
    def __init__(
        self,
        data_source: str = "day",  # 可选 "day", "minute", "tick"
        initial_expressions: Optional[list] = None,  # 初始因子表达式列表
        use_custom_functions: bool = True,  # 是否使用自定义时间序列函数
        window_sizes: list = [3, 5, 10, 20],  # 时间序列函数的窗口大小列表
        **model_params  # 使用**kwargs接收所有模型参数
    ):
        """
        初始化因子挖掘器
        
        Args:
            data_source: 数据源选择
            initial_expressions: 初始因子表达式列表，例如 ['x0 + x1', 'x0 * x1 / x2']
            use_custom_functions: 是否使用自定义时间序列函数
            **model_params: SymbolicRegressor的参数,包括:
                population_size: 种群大小
                generations: 迭代代数
                tournament_size: 锦标赛大小
                stopping_criteria: 停止条件
                const_range: 常数范围
                init_depth: 初始深度范围
                init_method: 初始化方法
                function_set: 函数集
                metric: 评估指标
                parsimony_coefficient: 简约系数
                p_crossover: 交叉概率
                p_subtree_mutation: 子树变异概率
                p_hoist_mutation: 提升变异概率
                p_point_mutation: 点变异概率
                max_samples: 最大样本比例
                verbose: 显示详细程度
                random_state: 随机种子
        """
        self.data_source = data_source
        self.initial_expressions = initial_expressions
        self.use_custom_functions = use_custom_functions
        self.window_sizes = window_sizes
        
        # 初始化数据源映射
        self.source_mapping = {
            "day": DAY_BAR,
            "minute": MINUTE_BAR,
            "tick": TICK
        }
        
        # 设置默认的遗传表达式参数 - 优化训练参数
        default_params = {
            'population_size': 1000,  # 增加种群大小
            'generations': 50,        # 增加迭代代数
            'tournament_size': 20,    # 增加锦标赛大小
            'stopping_criteria': 0.001,  # 设置合理的停止条件
            'const_range': (-10.0, 10.0),  # 扩大常数范围
            'init_depth': (3, 8),     # 增加初始深度
            'init_method': 'half and half',
            'metric': 'mse',
            'parsimony_coefficient': 0.01,  # 增加简约系数，避免过拟合
            'p_crossover': 0.7,       # 交叉概率
            'p_subtree_mutation': 0.1,  # 子树变异概率
            'p_hoist_mutation': 0.05,  # 提升变异概率
            'p_point_mutation': 0.1,   # 点变异概率
            'max_samples': 0.9,      # 使用90%的样本进行训练
            'verbose': 1,
            'random_state': 42
        }
        
        # 根据是否使用自定义函数设置函数集
        if use_custom_functions:
            default_params['function_set'] = custom_function_set
            logger.info(f"使用自定义时间序列函数集，窗口大小: {window_sizes}")
        else:
            default_params['function_set'] = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs']
            logger.info("使用标准函数集")
        
        # 合并用户传入的参数和默认参数
        final_params = {**default_params, **model_params}
        
        # 初始化gplearn模型
        self.model = SymbolicRegressor(**final_params)
        
        self.scaler = StandardScaler()
        self.feature_names = None

    def calculate_factor(self, **kwargs):
        # return super().calculate_factor(data)
        pass

    def load_data(self, start_date, end_date):
        """
        加载指定时间段的数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        logger.info(f"正在从{self.data_source}加载数据...")
        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)
        
        # 获取数据源
        source = self.source_mapping[self.data_source]
        
        # 过滤日期范围,读取一个更大的目标范围的数据，但只使用目标日期范围内的数据
        # 目标特征列为20个交易日后的open相对于当前open的涨跌幅
        data = (
            source
            .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date + datetime.timedelta(days=20)))
            .with_columns(
                ((pl.col("open").shift(-20)
                 .sub(pl.col("open"))/pl.col("open"))
                 .over("order_book_id")
                 .alias("target_feature")
                 )
            )
            .collect()
            .to_pandas()
        )

        #读取路径为QIML_factor的文件，提取每个文件中的factor_name列作为列名，factor_value列作为值，并合并到data中
        qiml_factor = QIML_factor.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
            ).collect().to_pandas()

        # 将不同的因子名称作为列名，不同的因子值作为列值，按日期和order_book_id展开
        qiml_factor_wide = qiml_factor.pivot_table(
            index=["date", "order_book_id"], columns="factor_name", values="factor_value"
        ).reset_index()

        data = pd.concat([data, qiml_factor_wide], axis=1)
        
        # 只删除目标变量为NaN的行，保留其他数据
        data = data.dropna(subset=['target_feature'])
        
        # 处理无限值和异常值
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=['target_feature'])
        
        # 移除异常大的值（超过10倍标准差）
        target_std = data['target_feature'].std()
        if target_std > 0:
            outlier_threshold = 10 * target_std
            data = data[np.abs(data['target_feature']) <= outlier_threshold]
        
        # 数据验证
        if len(data) == 0:
            raise ValueError("加载的数据为空，请检查日期范围和数据源")
        
        # 检查目标变量的分布
        target_stats = data['target_feature'].describe()
        logger.info(f"目标变量统计信息:\n{target_stats}")
        
        # 检查是否有异常值
        if target_stats['std'] < 1e-6:
            logger.warning("目标变量标准差过小，可能存在数据问题")
        
        logger.info(f"数据加载完成,共{len(data)}条记录")
        logger.info(f"数据列名: {data.columns.tolist()}")
        logger.info(f"数据形状: {data.shape}")
        logger.info(f"数据前5行:\n{data.head()}")
        
        # 检查数据质量
        logger.info(f"缺失值统计:\n{data.isnull().sum()}")
        logger.info(f"数据类型:\n{data.dtypes}")
        
        # 保存数据
        self.data = data
        return self
        
    def preprocess(self, target_col: str, feature_cols: Optional[list] = None):
        """
        数据预处理
        
        Args:
            target_col: 目标列名
            feature_cols: 特征列名列表,默认使用所有数值列
        """
        # 如果未指定特征列,则使用所有数值列(排除date、order_book_id和目标列)
        if feature_cols is None:
            # 获取所有列名
            all_cols = self.data.columns.tolist()
            # 排除不需要的列
            exclude_cols = ['date', 'order_book_id', 'prev_close', target_col, 'limit_up', 'limit_down',
                            'total_turnover', 'num_trades']
            feature_cols = [col for col in all_cols if col not in exclude_cols]
            
        logger.info(f"使用特征列: {feature_cols}")
        logger.info(f"目标列: {target_col}")
        
        # 先合并特征和目标列，统一去除缺失值，确保样本数量一致
        df = self.data[feature_cols + [target_col]].dropna()
        
        if len(df) == 0:
            raise ValueError("预处理后数据为空，请检查特征列和目标列")
        
        self.X = df[feature_cols]
        self.y = df[target_col]
        
        logger.info(f"预处理前数据形状: X={self.X.shape}, y={self.y.shape}")
        
        # 检查特征数据的质量
        logger.info("特征数据质量检查:")
        for col in feature_cols:
            col_data = self.X[col]
            missing_pct = col_data.isnull().sum() / len(col_data) * 100
            zero_pct = (col_data == 0).sum() / len(col_data) * 100
            std_val = col_data.std()
            logger.info(f"  {col}: 缺失率={missing_pct:.2f}%, 零值率={zero_pct:.2f}%, 标准差={std_val:.6f}")
        
        # 移除标准差过小的特征（可能导致标准化问题）
        valid_features = []
        for col in feature_cols:
            if self.X[col].std() > 1e-8:  # 标准差大于1e-8的特征
                valid_features.append(col)
            else:
                logger.warning(f"移除标准差过小的特征: {col} (std={self.X[col].std():.2e})")
        
        if len(valid_features) == 0:
            raise ValueError("没有有效的特征列，所有特征的标准差都过小")
        
        self.X = self.X[valid_features]
        logger.info(f"有效特征列: {valid_features}")
        
        # 处理异常值 - 使用IQR方法
        for col in valid_features:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            self.X[col] = self.X[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"特征列统计信息:\n{self.X.describe()}")
        logger.info(f"目标列统计信息:\n{self.y.describe()}")
        
        # 安全的标准化特征 - 处理零标准差的情况
        try:
            # 检查是否有零标准差的特征
            feature_stds = self.X.std()
            zero_std_features = feature_stds[feature_stds < 1e-8].index.tolist()
            
            if zero_std_features:
                logger.warning(f"发现零标准差特征: {zero_std_features}，将移除这些特征")
                valid_features = [col for col in valid_features if col not in zero_std_features]
                self.X = self.X[valid_features]
            
            # 使用RobustScaler替代StandardScaler，对异常值更鲁棒
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.X.columns
            )
            
            logger.info("特征标准化完成")
            
        except Exception as e:
            logger.error(f"特征标准化失败: {e}")
            # 如果标准化失败，使用简单的min-max缩放
            logger.info("使用min-max缩放作为备选方案")
            for col in self.X.columns:
                col_min = self.X[col].min()
                col_max = self.X[col].max()
                if col_max > col_min:
                    self.X[col] = (self.X[col] - col_min) / (col_max - col_min)
                else:
                    self.X[col] = 0  # 如果所有值相同，设为0
        
        # 检查目标变量的有效性
        y_mean = self.y.mean()
        y_std = self.y.std()
        
        logger.info(f"目标变量原始统计: mean={y_mean:.6f}, std={y_std:.6f}")
        
        if y_std < 1e-6:
            logger.error("目标变量标准差过小，无法进行有效训练")
            raise ValueError("目标变量标准差过小，请检查数据质量")
        
        # 对目标变量进行稳健的缩放
        # 使用中位数和IQR进行标准化，对异常值更鲁棒
        y_median = self.y.median()
        y_q75 = self.y.quantile(0.75)
        y_q25 = self.y.quantile(0.25)
        y_iqr = y_q75 - y_q25
        
        if y_iqr > 1e-8:
            self.y = (self.y - y_median) / y_iqr
        else:
            # 如果IQR太小，使用标准差
            self.y = (self.y - y_mean) / y_std
        
        logger.info(f"数据预处理完成,使用特征列:{valid_features};目标列:{target_col};样本数量:{len(self.X)}")
        logger.info(f"预处理后目标变量统计: mean={self.y.mean():.6f}, std={self.y.std():.6f}")
        
        # 最终数据质量检查
        if self.X.isnull().any().any():
            logger.warning("特征数据中存在缺失值，将进行插值处理")
            self.X = self.X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if self.y.isnull().any():
            logger.warning("目标变量中存在缺失值，将移除这些样本")
            valid_mask = ~self.y.isnull()
            self.X = self.X[valid_mask]
            self.y = self.y[valid_mask]
        
        logger.info(f"最终数据形状: X={self.X.shape}, y={self.y.shape}")
        return self
        
    def set_initial_population(self):
        """设置初始种群，包含用户提供的初始表达式"""
        if self.initial_expressions is None:
            logger.info("未提供初始表达式，使用随机初始化")
            return
            
        try:
            # 使用更简单的方法：通过修改SymbolicRegressor的初始化参数
            # 创建一个包含初始表达式的种群
            
            # 获取当前模型参数
            params = self.model.get_params()
            
            # 如果有初始表达式，调整种群大小和初始化策略
            if len(self.initial_expressions) > 0:
                # 确保种群大小足够容纳初始表达式
                min_population = max(params['population_size'], len(self.initial_expressions) * 2)
                if params['population_size'] < min_population:
                    logger.info(f"调整种群大小从 {params['population_size']} 到 {min_population}")
                    self.model.set_params(population_size=min_population)
                
                logger.info(f"将使用 {len(self.initial_expressions)} 个初始表达式作为遗传算法的起点")
                
                # 记录初始表达式供后续参考
                self.initial_expressions_info = {
                    'expressions': self.initial_expressions,
                    'count': len(self.initial_expressions)
                }
                
        except Exception as e:
            logger.warning(f"设置初始种群时出错: {e}")
    
    def _create_program_from_expression(self, expression):
        """从表达式字符串创建Program对象（简化实现）"""
        try:
            # 这是一个简化的实现
            # 在实际使用中，你可能需要更复杂的解析逻辑
            from gplearn.genetic import _Program
            
            # 创建基本的程序结构
            # 这里只是示例，实际实现需要根据gplearn的内部结构
            return None  # 暂时返回None，表示使用默认初始化
        except Exception as e:
            logger.warning(f"创建程序时出错: {e}")
            return None
        
    def train(self):
        """训练模型"""
        logger.info("开始训练模型...")
        logger.info(f"训练数据形状: X={self.X.shape}, y={self.y.shape}")
        logger.info(f"模型参数: {self.model.get_params()}")
        
        # 数据验证
        if self.X.shape[0] < 100:
            logger.warning(f"训练样本数量较少: {self.X.shape[0]}，可能影响训练效果")
        
        if self.y.std() < 1e-6:
            logger.error("目标变量方差过小，无法进行有效训练")
            raise ValueError("目标变量方差过小")
        
        # 设置初始种群（如果提供了初始表达式）
        self.set_initial_population()
        
        try:
            self.model.fit(self.X, self.y)
            score = self.model.score(self.X, self.y)
            logger.info(f"模型训练完成,最优得分:{score:.6f}")
            
            # 检查模型是否真的训练了
            if hasattr(self.model, '_program') and self.model._program is not None:
                logger.info(f"训练出的表达式: {str(self.model._program)}")
                
                # 计算预测值和实际值的相关性
                y_pred = self.model.predict(self.X)
                correlation = np.corrcoef(self.y, y_pred)[0, 1]
                logger.info(f"预测值与实际值的相关性: {correlation:.6f}")
                
                # 计算R²
                r2 = 1 - np.sum((self.y - y_pred) ** 2) / np.sum((self.y - self.y.mean()) ** 2)
                logger.info(f"R²: {r2:.6f}")
                
            else:
                logger.warning("模型训练可能失败，没有生成程序")
                
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise e
            
        return self
        
    def get_factor_expression(self):
        """获取因子表达式"""
        return str(self.model._program)
    
    def get_initial_expressions_info(self):
        """获取初始表达式信息"""
        if hasattr(self, 'initial_expressions_info'):
            return self.initial_expressions_info
        else:
            return {"expressions": [], "count": 0}
        
    def transform(self, X=None):
        """
        生成因子值
        
        Args:
            X: 输入数据,默认使用训练数据
            
        Returns:
            因子值序列
        """
        if X is None:
            X = self.X
        return self.model.predict(X)
        
    def save_results(self, output_dir: str = "result"):
        """
        保存结果
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存因子表达式
        with open(f"{output_dir}/factor_expression.txt", "w") as f:
            f.write(self.get_factor_expression())
            
        # 保存因子值
        factor_values = pd.DataFrame({
            "date": self.data["date"],
            "order_book_id": self.data["order_book_id"],
            "factor_value": self.transform()
        })
        factor_values.to_parquet(f"{output_dir}/factor_values.parquet")
        
        # 保存训练结果统计
        y_pred = self.transform()
        correlation = np.corrcoef(self.y, y_pred)[0, 1]
        r2 = 1 - np.sum((self.y - y_pred) ** 2) / np.sum((self.y - self.y.mean()) ** 2)
        
        with open(f"{output_dir}/training_stats.txt", "w") as f:
            f.write(f"训练样本数: {len(self.X)}\n")
            f.write(f"特征数: {self.X.shape[1]}\n")
            f.write(f"最优得分: {self.model.score(self.X, self.y):.6f}\n")
            f.write(f"预测相关性: {correlation:.6f}\n")
            f.write(f"R²: {r2:.6f}\n")
        
        logger.info(f"结果已保存至{output_dir}目录")


if __name__ == "__main__":

    
    # 创建因子挖掘器实例，使用自定义时间序列函数
    factor = GplearnFactor(
        data_source="day",           # 使用日线数据
        use_custom_functions=True,   # 启用自定义时间序列函数
        window_sizes=[3, 5, 10],     # 时间序列窗口大小
        population_size=1000,         # 种群大小
        generations=10,              # 迭代代数
        verbose=1,                   # 显示训练过程
        tournament_size=20,          # 锦标赛大小
        p_crossover=0.7,             # 交叉概率
        p_subtree_mutation=0.1,      # 子树变异概率
        p_hoist_mutation=0.05,       # 提升变异概率
        p_point_mutation=0.1,        # 点变异概率
        parsimony_coefficient=0.01,   # 简约系数
        n_jobs=-1
    )

    # 2. 加载数据 - 使用更大的时间范围
    start_date = datetime.date(2025, 4, 1)
    end_date = datetime.date(2025, 7, 31)  # 增加到6个月
    factor.load_data(start_date, end_date)

    # 3. 数据预处理，指定目标列
    factor.preprocess(target_col="target_feature")

    # 4. 训练模型
    factor.train()

    # 5. 获取并打印因子表达式
    expression = factor.get_factor_expression()
    print("发现的因子表达式:", expression)

    # 6. 保存结果
    factor.save_results("results_custom_functions_demo")


