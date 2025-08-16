"""
数据验证脚本
"""
import polars as pl
import pandas as pd
import numpy as np
import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# 全局数据源
DAY_BAR = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/stock/day_bar")

def validate_data_quality(start_date, end_date):
    """验证数据质量"""
    logger.info("开始验证数据质量...")
    
    # 加载原始数据
    source = DAY_BAR
    data = (
        source
        .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date + datetime.timedelta(days=20)))
        .collect()
        .to_pandas()
    )
    
    logger.info(f"原始数据形状: {data.shape}")
    logger.info(f"数据列名: {data.columns.tolist()}")
    
    # 检查基本统计信息
    logger.info("基本统计信息:")
    logger.info(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")
    logger.info(f"股票数量: {data['order_book_id'].nunique()}")
    logger.info(f"总记录数: {len(data)}")
    
    # 检查缺失值
    logger.info("缺失值统计:")
    missing_stats = data.isnull().sum()
    for col, missing_count in missing_stats.items():
        if missing_count > 0:
            logger.info(f"  {col}: {missing_count} ({missing_count/len(data)*100:.2f}%)")
    
    # 检查open价格
    logger.info("Open价格统计:")
    open_stats = data['open'].describe()
    logger.info(f"  {open_stats}")
    
    # 检查是否有异常值
    if open_stats['std'] < 1e-6:
        logger.warning("Open价格标准差过小，可能存在数据问题")
    
    # 计算目标变量
    logger.info("计算目标变量...")
    data_with_target = (
        pl.from_pandas(data)
        .with_columns(
            ((pl.col("open").shift(-20)
             .sub(pl.col("open"))/pl.col("open"))
             .over("order_book_id")
             .alias("target_feature")
             )
        )
        .to_pandas()
    )
    
    # 检查目标变量
    target_stats = data_with_target['target_feature'].describe()
    logger.info("目标变量统计:")
    logger.info(f"  {target_stats}")
    
    # 检查目标变量的缺失值
    target_missing = data_with_target['target_feature'].isnull().sum()
    logger.info(f"目标变量缺失值: {target_missing} ({target_missing/len(data_with_target)*100:.2f}%)")
    
    # 删除目标变量缺失值后的统计
    data_clean = data_with_target.dropna(subset=['target_feature'])
    logger.info(f"清理后数据形状: {data_clean.shape}")
    
    target_clean_stats = data_clean['target_feature'].describe()
    logger.info("清理后目标变量统计:")
    logger.info(f"  {target_clean_stats}")
    
    # 检查目标变量的分布
    if target_clean_stats['std'] < 1e-6:
        logger.error("目标变量标准差过小，无法进行训练")
        return False
    
    # 检查是否有无限值
    if np.isinf(data_clean['target_feature']).any():
        logger.warning("目标变量包含无限值")
        infinite_count = np.isinf(data_clean['target_feature']).sum()
        logger.info(f"无限值数量: {infinite_count}")
    
    # 检查是否有异常大的值
    target_abs = np.abs(data_clean['target_feature'])
    if target_abs.max() > 10:
        logger.warning("目标变量包含异常大的值")
        logger.info(f"最大绝对值: {target_abs.max()}")
    
    logger.info("数据质量验证完成")
    return True

def test_data_loading():
    """测试数据加载"""
    logger.info("测试数据加载...")
    
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 3, 31)
    
    try:
        success = validate_data_quality(start_date, end_date)
        if success:
            logger.info("数据加载测试成功")
        else:
            logger.error("数据加载测试失败")
    except Exception as e:
        logger.error(f"数据加载测试出错: {e}")
        raise

if __name__ == "__main__":
    test_data_loading() 