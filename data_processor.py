"""
数据处理模块 - 专门用于处理数据和输出target_feature的描述
"""
import pandas as pd
import polars as pl
import sys
from pathlib import Path
sys.path.append("/home/yzhzhang/factor_standard_pipeline")
import datetime
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# 全局数据源
DAY_BAR = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/day_bar")
MINUTE_BAR = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/minute_bar")
TICK = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/tick")

class DataProcessor:
    """数据处理类 - 专门用于处理数据和输出target_feature的描述"""
    
    def __init__(self, data_source: str = "day"):
        """
        初始化数据处理器
        
        Args:
            data_source: 数据源选择 "day", "minute", "tick"
        """
        self.data_source = data_source
        
        # 初始化数据源映射
        self.source_mapping = {
            "day": DAY_BAR,
            "minute": MINUTE_BAR,
            "tick": TICK
        }
        
        self.data = None
        self.target_feature = None
        
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
        self.target_feature = data['target_feature']
        
        return self
        
    def analyze_target_feature(self):
        """分析target_feature的详细统计信息"""
        if self.target_feature is None:
            logger.error("请先加载数据")
            return None
            
        logger.info("=" * 50)
        logger.info("TARGET_FEATURE 详细分析")
        logger.info("=" * 50)
        
        # 基本统计信息
        logger.info("1. 基本统计信息:")
        stats = self.target_feature.describe()
        logger.info(f"   {stats}")
        
        # 分位数信息
        logger.info("\n2. 分位数信息:")
        percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for p in percentiles:
            value = self.target_feature.quantile(p)
            logger.info(f"   {p*100}%分位数: {value:.6f}")
        
        # 分布特征
        logger.info("\n3. 分布特征:")
        logger.info(f"   均值: {self.target_feature.mean():.6f}")
        logger.info(f"   中位数: {self.target_feature.median():.6f}")
        logger.info(f"   标准差: {self.target_feature.std():.6f}")
        logger.info(f"   偏度: {self.target_feature.skew():.6f}")
        logger.info(f"   峰度: {self.target_feature.kurtosis():.6f}")
        
        # 极值信息
        logger.info("\n4. 极值信息:")
        logger.info(f"   最小值: {self.target_feature.min():.6f}")
        logger.info(f"   最大值: {self.target_feature.max():.6f}")
        logger.info(f"   范围: {self.target_feature.max() - self.target_feature.min():.6f}")
        
        # 零值统计
        zero_count = (self.target_feature == 0).sum()
        zero_ratio = zero_count / len(self.target_feature)
        logger.info(f"   零值数量: {zero_count} ({zero_ratio*100:.2f}%)")
        
        # 正负值统计
        positive_count = (self.target_feature > 0).sum()
        negative_count = (self.target_feature < 0).sum()
        positive_ratio = positive_count / len(self.target_feature)
        negative_ratio = negative_count / len(self.target_feature)
        
        logger.info(f"   正值数量: {positive_count} ({positive_ratio*100:.2f}%)")
        logger.info(f"   负值数量: {negative_count} ({negative_ratio*100:.2f}%)")
        
        # 异常值检测
        logger.info("\n5. 异常值检测:")
        Q1 = self.target_feature.quantile(0.25)
        Q3 = self.target_feature.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.target_feature[(self.target_feature < lower_bound) | (self.target_feature > upper_bound)]
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / len(self.target_feature)
        
        logger.info(f"   IQR: {IQR:.6f}")
        logger.info(f"   下界: {lower_bound:.6f}")
        logger.info(f"   上界: {upper_bound:.6f}")
        logger.info(f"   异常值数量: {outlier_count} ({outlier_ratio*100:.2f}%)")
        
        # 数据质量检查
        logger.info("\n6. 数据质量检查:")
        missing_count = self.target_feature.isnull().sum()
        infinite_count = np.isinf(self.target_feature).sum()
        
        logger.info(f"   缺失值: {missing_count}")
        logger.info(f"   无限值: {infinite_count}")
        
        if missing_count == 0 and infinite_count == 0:
            logger.info("   ✓ 数据质量良好")
        else:
            logger.warning("   ⚠ 数据质量存在问题")
        
        # 样本信息
        logger.info("\n7. 样本信息:")
        logger.info(f"   总样本数: {len(self.target_feature)}")
        logger.info(f"   唯一值数量: {self.target_feature.nunique()}")
        
        # 返回统计结果
        result = {
            'basic_stats': stats,
            'percentiles': {p: self.target_feature.quantile(p) for p in percentiles},
            'distribution': {
                'mean': self.target_feature.mean(),
                'median': self.target_feature.median(),
                'std': self.target_feature.std(),
                'skew': self.target_feature.skew(),
                'kurtosis': self.target_feature.kurtosis()
            },
            'extremes': {
                'min': self.target_feature.min(),
                'max': self.target_feature.max(),
                'range': self.target_feature.max() - self.target_feature.min()
            },
            'counts': {
                'zero': zero_count,
                'positive': positive_count,
                'negative': negative_count,
                'outliers': outlier_count
            },
            'quality': {
                'missing': missing_count,
                'infinite': infinite_count
            }
        }
        
        return result
        
    def save_analysis_report(self, output_file: str = "target_feature_analysis.txt"):
        """保存分析报告到文件"""
        if self.target_feature is None:
            logger.error("请先加载数据")
            return
            
        logger.info(f"保存分析报告到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TARGET_FEATURE 分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计信息
            f.write("1. 基本统计信息:\n")
            f.write(str(self.target_feature.describe()) + "\n\n")
            
            # 分布特征
            f.write("2. 分布特征:\n")
            f.write(f"   均值: {self.target_feature.mean():.6f}\n")
            f.write(f"   中位数: {self.target_feature.median():.6f}\n")
            f.write(f"   标准差: {self.target_feature.std():.6f}\n")
            f.write(f"   偏度: {self.target_feature.skew():.6f}\n")
            f.write(f"   峰度: {self.target_feature.kurtosis():.6f}\n\n")
            
            # 分位数信息
            f.write("3. 分位数信息:\n")
            percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            for p in percentiles:
                value = self.target_feature.quantile(p)
                f.write(f"   {p*100}%分位数: {value:.6f}\n")
            f.write("\n")
            
            # 样本信息
            f.write("4. 样本信息:\n")
            f.write(f"   总样本数: {len(self.target_feature)}\n")
            f.write(f"   唯一值数量: {self.target_feature.nunique()}\n")
            
            zero_count = (self.target_feature == 0).sum()
            positive_count = (self.target_feature > 0).sum()
            negative_count = (self.target_feature < 0).sum()
            
            f.write(f"   零值数量: {zero_count} ({zero_count/len(self.target_feature)*100:.2f}%)\n")
            f.write(f"   正值数量: {positive_count} ({positive_count/len(self.target_feature)*100:.2f}%)\n")
            f.write(f"   负值数量: {negative_count} ({negative_count/len(self.target_feature)*100:.2f}%)\n")
            
        logger.info("分析报告保存完成")


def main():
    """主函数 - 演示如何使用DataProcessor"""
    logger.info("开始数据处理和target_feature分析...")
    
    # 创建数据处理器
    processor = DataProcessor(data_source="day")
    
    # 设置数据时间范围
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 3, 31)
    
    try:
        # 加载数据
        processor.load_data(start_date, end_date)
        
        # 分析target_feature
        result = processor.analyze_target_feature()
        
        # 保存分析报告
        processor.save_analysis_report()
        
        logger.info("数据处理和分析完成！")
        
        return result
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise


if __name__ == "__main__":
    main() 