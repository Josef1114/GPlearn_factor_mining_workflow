# 自定义时间序列函数因子挖掘指南

## 概述

本指南介绍如何在 gplearn 遗传算法中集成自定义的时间序列函数，用于因子挖掘。通过添加时间序列特定的函数，可以提高因子挖掘的效果和实用性。

## 功能特性

### 1. 基础数学函数
- `safe_add(x1, x2)`: 安全的加法，处理NaN值
- `safe_sub(x1, x2)`: 安全的减法，处理NaN值
- `safe_mul(x1, x2)`: 安全的乘法，处理NaN值
- `safe_div(x1, x2)`: 安全的除法，处理NaN值和除零
- `safe_log(x)`: 安全的对数，处理负数和零
- `safe_sqrt(x)`: 安全的平方根，处理负数
- `safe_abs(x)`: 绝对值

### 2. 时间序列函数
- `ts_rank_{window}(x)`: 时间序列排名，返回当前值在过去window期内的分位数
- `ts_max_{window}(x)`: 时间序列最大值，返回过去window期内的最大值
- `ts_min_{window}(x)`: 时间序列最小值，返回过去window期内的最小值
- `ts_mean_{window}(x)`: 时间序列均值，返回过去window期内的均值
- `ts_std_{window}(x)`: 时间序列标准差，返回过去window期内的标准差
- `ts_sum_{window}(x)`: 时间序列求和，返回过去window期内的和

## 使用方法

### 基本用法

```python
from base import GplearnFactor
import datetime

# 创建因子挖掘器，使用自定义时间序列函数
factor = GplearnFactor(
    data_source="day",
    use_custom_functions=True,  # 启用自定义函数
    window_sizes=[3, 5, 10, 20],  # 设置时间窗口大小
    population_size=1000,
    generations=20,
    verbose=1
)

# 加载数据
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 1, 31)
factor.load_data(start_date, end_date)
factor.preprocess(target_col="target_open_next")

# 训练模型
factor.train()

# 获取结果
expression = factor.get_factor_expression()
print(f"发现的因子表达式: {expression}")
```

### 高级配置

```python
# 使用不同的窗口大小配置
factor = GplearnFactor(
    data_source="day",
    use_custom_functions=True,
    window_sizes=[2, 3, 5, 8, 13, 21],  # 斐波那契数列窗口
    population_size=2000,
    generations=50,
    tournament_size=30,
    parsimony_coefficient=0.01,  # 增加简约性约束
    verbose=1
)
```

### 对比标准函数集

```python
# 使用标准函数集
factor_standard = GplearnFactor(
    data_source="day",
    use_custom_functions=False,  # 使用标准函数集
    population_size=1000,
    generations=20
)

# 使用自定义函数集
factor_custom = GplearnFactor(
    data_source="day",
    use_custom_functions=True,
    window_sizes=[3, 5, 10],
    population_size=1000,
    generations=20
)
```

## 函数集配置

### 窗口大小选择

不同的窗口大小适用于不同的市场特征：

- **短期窗口 (2-5天)**: 捕捉短期价格波动和动量
- **中期窗口 (5-20天)**: 捕捉趋势变化和均值回归
- **长期窗口 (20天以上)**: 捕捉长期趋势和周期性模式

### 推荐配置

```python
# 短期交易配置
short_term_config = {
    "window_sizes": [2, 3, 5, 8],
    "population_size": 1500,
    "generations": 30
}

# 中期投资配置
medium_term_config = {
    "window_sizes": [5, 10, 20, 30],
    "population_size": 2000,
    "generations": 50
}

# 长期投资配置
long_term_config = {
    "window_sizes": [20, 30, 60, 120],
    "population_size": 2500,
    "generations": 100
}
```

## 性能优化建议

### 1. 计算效率
- 时间序列函数使用滚动窗口计算，对于大数据集可能较慢
- 建议使用适当的数据采样或减少窗口大小来提高速度

### 2. 内存使用
- 大量时间序列函数会增加内存使用
- 监控内存使用情况，必要时减少函数集大小

### 3. 过拟合控制
- 使用 `parsimony_coefficient` 参数控制模型复杂度
- 增加 `tournament_size` 提高选择压力
- 使用交叉验证评估模型泛化能力

## 示例输出

### 函数集分析
```
=== 分析自定义函数集 ===
中等窗口配置:
  窗口大小: [3, 5, 10, 20]
  函数总数: 31
  数学函数数量: 7
  时间序列函数数量: 24
  时间序列函数: ['ts_rank_3', 'ts_max_3', 'ts_min_3', 'ts_mean_3', 'ts_std_3', 'ts_sum_3', ...]
```

### 发现的因子表达式
```
发现的因子表达式: ts_mean_5(x0) * safe_log(ts_std_10(x1)) + ts_rank_3(x2)
```

## 注意事项

1. **数据质量**: 确保输入数据没有过多的缺失值
2. **窗口大小**: 窗口大小不应超过数据长度
3. **计算资源**: 大量时间序列函数会增加计算时间
4. **过拟合**: 监控训练集和验证集性能差异

## 故障排除

### 常见问题

1. **内存不足**: 减少 `population_size` 或 `window_sizes`
2. **计算过慢**: 减少 `generations` 或使用数据采样
3. **函数集过大**: 减少 `window_sizes` 或禁用部分函数类型

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查函数集
factor = GplearnFactor(use_custom_functions=True, window_sizes=[3, 5])
print(f"函数集大小: {len(factor.model.function_set)}")
print(f"函数名称: {[f.__name__ for f in factor.model.function_set]}")
```

## 扩展开发

### 添加新的时间序列函数

```python
def create_ts_momentum(window):
    """创建动量函数"""
    def ts_momentum_func(x):
        if len(x) < window:
            return np.full_like(x, np.nan)
        result = np.full_like(x, np.nan)
        for i in range(window-1, len(x)):
            window_data = x[i-window+1:i+1]
            if not np.any(np.isnan(window_data)):
                result[i] = (window_data[-1] - window_data[0]) / window_data[0]
        return result
    ts_momentum_func.__name__ = f'ts_momentum_{window}'
    return ts_momentum_func
```

### 自定义函数集

```python
def create_custom_function_set_with_momentum(window_sizes=[3, 5, 10]):
    # 基础函数集
    base_functions = create_custom_function_set(window_sizes)
    
    # 添加动量函数
    momentum_functions = [create_ts_momentum(w) for w in window_sizes]
    
    return base_functions + momentum_functions
``` 