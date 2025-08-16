import polars as pl

QIML_factor = pl.scan_parquet("/mnt/nvme1n1/exchange_area/filedb_dev/factor/QIML/")



data = QIML_factor.collect().to_pandas()
print(data.head())
