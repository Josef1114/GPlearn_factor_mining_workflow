"""
测试修复效果的脚本
"""
import datetime
import logging
from base import GplearnFactor

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def test_data_preprocessing():
    """测试数据预处理"""
    logger.info("开始测试数据预处理...")
    
    try:
        # 创建因子挖掘器实例
        factor = GplearnFactor(
            data_source="day",
            use_custom_functions=True,
            population_size=100,  # 小规模测试
            generations=5,
            verbose=0
        )

        # 加载数据
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2020, 3, 31)
        factor.load_data(start_date, end_date)

        # 数据预处理
        factor.preprocess(target_col="target_feature")
        
        # 检查结果
        logger.info(f"预处理后数据形状: X={factor.X.shape}, y={factor.y.shape}")
        logger.info(f"目标变量统计: mean={factor.y.mean():.6f}, std={factor.y.std():.6f}")
        
        # 验证目标变量是否有效
        if factor.y.std() > 1e-6:
            logger.info("✓ 数据预处理成功，目标变量有效")
            return True
        else:
            logger.error("✗ 数据预处理失败，目标变量无效")
            return False
            
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试修复效果...")
    
    success = test_data_preprocessing()
    
    if success:
        logger.info("✓ 所有测试通过")
    else:
        logger.error("✗ 测试失败")

if __name__ == "__main__":
    main() 