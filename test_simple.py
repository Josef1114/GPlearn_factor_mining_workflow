import pandas as pd
from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor
import numpy as np
import datetime
import logging
import polars as pl

QIML_factor = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/factor/QIML/")

start_date = datetime.datetime.strptime("2025-01-01", "%Y-%m-%d")
end_date = datetime.datetime.strptime("2025-01-02", "%Y-%m-%d")

#读取路径为QIML_factor的文件，提取每个文件中的factor_name列作为列名，factor_value列作为值，并合并到data中
qiml_factor = QIML_factor.filter(
    (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
).collect().to_pandas()

# 将不同的因子名称作为列名，不同的因子值作为列值，按日期和order_book_id展开
qiml_factor_wide = qiml_factor.pivot_table(
    index=["date", "order_book_id"], columns="factor_name", values="factor_value"
).reset_index()

print(type(qiml_factor_wide))





