import os
from typing import Union, List, Optional
from datetime import datetime, date
from pathlib import Path
from gplearn.functions import make_function
import pandas as pd
import polars as pl
import yaml
import numpy as np

WINDOW = 10

def _validate_input(x):
    """统一的输入验证"""
    if np.all(np.isnan(x)) or np.all(x == 0):
        return False
    return True

def _apply_rolling_function(x, func):
    """通用的滚动窗口函数应用器 - 使用pandas优化"""
    if not _validate_input(x):
        return np.zeros_like(x)
    
    # 转换为pandas Series以使用rolling操作
    series = pd.Series(x)
    result = series.rolling(window=WINDOW, min_periods=1).apply(func, raw=True)
    
    return np.nan_to_num(result.values, nan=0.0, posinf=0.0, neginf=0.0)

def ts_rank_10(x):
    """时间序列排名函数"""
    def rank_func(window_data):
        if len(window_data) < 2:
            return 0
        # 计算当前值在窗口中的排名
        current_val = window_data[-1]
        if np.isnan(current_val):
            return 0
        # 计算排名百分比
        rank = np.sum(window_data <= current_val) - 1
        return rank / (len(window_data) - 1)
    
    return _apply_rolling_function(x, rank_func)

def ts_max_10(x):
    """时间序列最大值函数"""
    return _apply_rolling_function(x, lambda w: np.nanmax(w))

def ts_min_10(x):
    """时间序列最小值函数"""
    return _apply_rolling_function(x, lambda w: np.nanmin(w))

def ts_mean_10(x):
    """时间序列均值函数"""
    return _apply_rolling_function(x, lambda w: np.nanmean(w))

def ts_median_10(x):
    """时间序列中位数函数"""
    return _apply_rolling_function(x, lambda w: np.nanmedian(w))

def ts_std_10(x):
    """时间序列标准差函数"""
    return _apply_rolling_function(x, lambda w: np.nanstd(w))

def ts_sum_10(x):
    """时间序列求和函数"""
    return _apply_rolling_function(x, lambda w: np.nansum(w))

# def inv(x):
    # if not _validate_input(x):
    #     return np.zeros_like(x)
    # # 添加微小偏移避免除以零
    # x_safe = np.where(np.abs(x) < 1e-10, np.sign(x) * 1e-10, x)
    # result = 1 / x_safe
    # return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

# 用 make_function 封装
rank_10_gp = make_function(function=ts_rank_10, name='ts_rank_10', arity=1)
max_10_gp = make_function(function=ts_max_10, name='ts_max_10', arity=1)
min_10_gp = make_function(function=ts_min_10, name='ts_min_10', arity=1)
mean_10_gp = make_function(function=ts_mean_10, name='ts_mean_10', arity=1)
median_10_gp = make_function(function=ts_median_10, name='ts_median_10', arity=1)
std_10_gp = make_function(function=ts_std_10, name='ts_std_10', arity=1)
sum_10_gp = make_function(function=ts_sum_10, name='ts_sum_10', arity=1)
# inv_gp = make_function(function=inv, name='inv', arity=1)

# gplearn 可用的 function_set
custom_function_set = [
    rank_10_gp,
    max_10_gp,
    min_10_gp,
    mean_10_gp,
    median_10_gp,
    std_10_gp,
    sum_10_gp,
    # inv_gp
]
