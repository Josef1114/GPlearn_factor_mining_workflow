# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件数据库基础模块

包含基类和全局变量

在filedb的基础上还是改了一些
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Union, Set

import logging
import polars as pl

from filedb_datac import DB_ROOT_DIR, get_trading_dates

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


MARKET = "cn"
GLOBAL_START_DATE = date(1990, 12, 19)


class PartitionStrategy(Enum):
    """
    分区策略

    查询经常用到的、增量用到的，应该放前面，例如date就比order_book_id要优先
    这个和SQL的最左索引不太一样
    """

    NONE = "none"  # 不分区，如交易日历
    DATE = "date"  # 按日期分区，如日线数据
    DATE_AND_STOCK = "date_and_stock"  # 按日期和股票分区，如逐笔数据
    DATE_AND_INDEX = "date_and_index"  # 按日期和指数分区，如指数成分股数据
    FACTOR_AND_DATE = "factor_and_date"  # 按因子名和日期分区，用于因子数据


@dataclass
class TableConfig:
    """表配置"""

    name: str  # 表名
    partition_strategy: PartitionStrategy  # 分区策略
    schema: Optional[pl.Schema]  # 字段类型映射


@dataclass
class MarketDataConfig(TableConfig):
    """行情数据配置"""

    frequency: str  # 数据频率
    adjust_type: str  # 复权类型


class DataManager:
    """数据管理器"""

    def __init__(self):
        self._trading_calendar_path = DB_ROOT_DIR / "stock" / "trading_calendar.parquet"

    def get_trading_dates(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> List[date]:
        """获取交易日历，优先从本地读取"""
        if self._trading_calendar_path.exists():
            dates = get_trading_dates(start_date, end_date)
            return dates
        else:
            raise FileNotFoundError(
                f"交易日历文件 {self._trading_calendar_path} 不存在"
            )

    def get_missing_dates(
        self, table_path: Path, use_global_start: bool = False
    ) -> List[date]:
        """检查缺失的交易日

        Args:
            table_path: 数据表路径
            use_global_start: 是否使用全局开始日期GLOBAL_START_DATE作为起始日期，默认为False（使用数据文件中最老日期）

        Returns:
            List[date]: 缺失的交易日列表
        """
        if not table_path.exists():
            raise FileNotFoundError(f"数据表 {table_path} 不存在")

        existing_dates = self._get_existing_dates(table_path)

        # 确定起始日期
        if use_global_start:
            start_date = GLOBAL_START_DATE
        else:
            start_date = min(existing_dates) if existing_dates else GLOBAL_START_DATE

        trading_dates = set(self.get_trading_dates(start_date, datetime.now().date()))
        return sorted(trading_dates - existing_dates)

    def _get_existing_dates(self, table_path: Path) -> Set[date]:
        """获取已存在数据的日期集合

        如果是分区parquet，指定table_path下一层必须就是 date 分区文件夹，不能再隔另外的分区层

        只要在再往下任意层级找到parquet文件，就会将该日期标记为有数据

        Args:
            table_path: 数据表路径

        Returns:
            Set[date]: 已存在数据的日期集合
        """
        if not table_path.exists():
            return set()

        existing_dates = set()

        # 根据分区策略获取已存在的日期
        if table_path.suffix == ".parquet":  # 不分区的表
            # 读取parquet文件获取日期范围
            df = pl.scan_parquet(table_path).select("date").collect()
            if not df.is_empty():
                existing_dates.update(df["date"].dt.date().to_list())
        else:  # 分区的表
            # 遍历日期文件夹（Hive风格：date=YYYY-MM-DD） TODO: date不一定在第一层
            for date_str in os.listdir(table_path):
                try:
                    # 解析Hive风格的分区名
                    if date_str.startswith("date="):
                        date = datetime.strptime(
                            date_str.split("=")[1], "%Y-%m-%d"
                        ).date()
                        # 递归检查该日期分区下是否有数据
                        if self._has_data_in_partition(table_path / date_str):
                            existing_dates.add(date)
                except ValueError:
                    continue

        return existing_dates

    def _has_data_in_partition(self, partition_path: Path) -> bool:
        """递归检查分区路径下是否有数据

        只要在任意层级找到parquet文件，就会将该日期标记为有数据

        Args:
            partition_path: 分区路径

        Returns:
            bool: 是否有数据
        """
        if not partition_path.exists():
            return False

        # 如果是文件，检查是否为parquet文件
        if partition_path.is_file():
            return partition_path.suffix == ".parquet"

        # 如果是目录，递归检查其下的所有内容
        for item in os.listdir(partition_path):
            item_path = partition_path / item
            if item_path.is_file() and item_path.suffix == ".parquet":
                return True
            elif item_path.is_dir():
                if self._has_data_in_partition(item_path):
                    return True

        return False


class DataWriter:
    """数据写入器"""

    def write_parquet(
        self,
        df: pl.DataFrame,
        table_path: Union[Path, str],
        partition_by: Optional[List[str]] = None,
    ) -> None:
        """写入parquet文件"""
        if df.is_empty():
            return

        if partition_by:
            df.write_parquet(table_path, partition_by=partition_by)
        else:
            df.write_parquet(table_path)


class DataTable(ABC):
    """数据表基类"""

    def __init__(
        self,
        table_config: TableConfig,
        manager: Optional[DataManager] = None,
        writer: Optional[DataWriter] = None,
    ):
        self.table_config = table_config
        self.manager = manager or DataManager()  # 允许传入manager覆盖
        self.writer = writer or DataWriter()  # 允许传入writer覆盖
        self.table_path = DB_ROOT_DIR
        self.df = pl.LazyFrame()  # 存储当前表格的数据

    def fetch(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> pl.LazyFrame:
        """获取数据并更新self.df"""
        self.df = self._fetch_impl(start_date, end_date)
        self.df = self._format_data(self.df)
        return self.df

    def update_full(self) -> None:
        """全量更新：从起始日期到今天"""
        logger.info(f"开始全量更新 {self.table_config.name}")
        self.df = self.fetch(GLOBAL_START_DATE, datetime.now().date())
        self._write_data()
        logger.info(f"全量更新 {self.table_config.name} 完成")

    def update_missing(self, use_global_start: bool = False) -> None:
        """更新缺失数据：根据交易日历比对，只更新缺失的日期

        Args:
            use_global_start: 是否使用全局开始日期GLOBAL_START_DATE作为起始日期，默认为False（使用数据文件中最老日期）
        """
        missing_dates = self.manager.get_missing_dates(
            self.table_path, use_global_start
        )
        if not missing_dates:
            logger.info(f"No missing data found for {self.table_config.name}")
            return

        logger.info(
            f"开始更新 {self.table_config.name} 的缺失数据，共 {len(missing_dates)} 个交易日"
        )

        for dt in missing_dates:
            logger.info(f"正在更新 {dt} 的数据")
            self.df = self.fetch(dt, dt)
            self._write_data()

        logger.info(f"缺失数据更新完成")

    def update_latest(self) -> None:
        """增量更新：从最新数据日期到今天"""
        if not self.table_path.exists():
            logger.info(f"{self.table_config.name} 路径不存在，执行全量更新")
            self.update_full()
            return

        existing_dates = self.manager._get_existing_dates(self.table_path)
        if not existing_dates:
            logger.info(f"{self.table_config.name} 没有有效数据，执行全量更新")
            self.update_full()
            return

        latest_date = max(existing_dates)

        # 获取最新日期后的交易日
        trading_dates = self.manager.get_trading_dates(
            latest_date + timedelta(days=1), datetime.now().date()
        )

        if not trading_dates:
            logger.info(f"No new trading dates found for {self.table_config.name}")
            return

        logger.info(
            f"开始更新 {self.table_config.name} 从 {trading_dates[0]} 到 {trading_dates[-1]}"
        )
        self.df = self.fetch(trading_dates[0], trading_dates[-1])
        self._write_data()
        logger.info(f"最新数据更新完成")

    # 抽象方法
    @abstractmethod
    def _fetch_impl(
        self, start_date: Union[str, date], end_date: Union[str, date]
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """获取数据的具体实现"""
        pass

    def _get_api_fields(self) -> List[str]:
        """获取用于API请求的字段列表，排除date, datetime 和order_book_id"""
        if not self.table_config.schema:
            return []

        # 排除date和order_book_id字段
        exclude_fields = {"date", "datetime", "order_book_id"}
        return [
            field
            for field in self.table_config.schema.keys()
            if field not in exclude_fields
        ]

    def _format_data(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
        """格式化数据以符合schema规范：
        1. 根据schema转换数据类型
        2. 按schema顺序排序列
        3. 按date和order_book_id排序行
        """
        if isinstance(df, pl.DataFrame):
            if df.is_empty():
                return df
            else:
                df = df.lazy()

        # 1. 类型转换
        if self.table_config.schema:
            df = df.cast(self.table_config.schema)

        # 2. 列排序
        if self.table_config.schema:
            schema_columns = list(self.table_config.schema.keys())
            # 确保所有schema中的列都存在
            existing_columns = [
                col for col in schema_columns if col in df.collect_schema().names()
            ]
            # 获取schema中不存在的列
            extra_columns = [
                col for col in df.collect_schema().names() if col not in schema_columns
            ]
            # 按schema顺序重新排列列
            df = df.select(existing_columns + extra_columns)

        # 3. 行排序
        sort_columns = []
        schema_names = df.collect_schema().names()
        if "date" in schema_names:
            sort_columns.append("date")
        if "index_code" in schema_names:
            sort_columns.append("index_code")
        if "order_book_id" in schema_names:
            sort_columns.append("order_book_id")
        if "datetime" in schema_names:
            sort_columns.append("datetime")
        df = df.sort(sort_columns)

        return df

    def _filter_incomplete_trading_day(
        self, df: Union[pl.DataFrame, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """过滤掉不完整的交易日数据

        对于日内数据（分钟线、tick），如果最新交易日数据不完整（不到15点），则过滤掉该日数据
        """
        if isinstance(df, pl.DataFrame):
            if df.is_empty():
                return df
            else:
                df = df.lazy()

        # 获取最新交易日的数据
        latest_date = df.select(pl.col("date").max()).collect().item()
        latest_day_data = df.filter(pl.col("date") == latest_date)

        last_time = latest_day_data.select(pl.col("datetime").max()).collect().item()
        if last_time.time() < time(hour=15, minute=0):
            logger.warning(
                f"最新交易日 {latest_date} 的数据不完整，最后一条数据时间为 {last_time}，将舍弃该日数据"
            )
            # 只保留完整的数据（排除最后一天）
            return df.filter(pl.col("date") < latest_date)

        return df

    def _write_data(self) -> None:
        """写入数据到parquet文件"""
        if isinstance(self.df, pl.LazyFrame):
            df = self.df.collect()
        elif isinstance(self.df, pl.DataFrame):
            df = self.df
        else:
            raise ValueError(f"self.df必须为pl.LazyFrame或pl.DataFrame")

        if df.is_empty():
            logger.warning(f"{self.table_config.name} 没有数据，跳过写入")
            return

        if self.table_config.partition_strategy == PartitionStrategy.DATE:
            self.writer.write_parquet(df, self.table_path, partition_by=["date"])

        elif self.table_config.partition_strategy == PartitionStrategy.DATE_AND_STOCK:
            self.writer.write_parquet(
                df, self.table_path, partition_by=["date", "order_book_id"]
            )

        elif self.table_config.partition_strategy == PartitionStrategy.DATE_AND_INDEX:
            self.writer.write_parquet(
                df, self.table_path, partition_by=["date", "index_code"]
            )

        elif self.table_config.partition_strategy == PartitionStrategy.FACTOR_AND_DATE:
            self.writer.write_parquet(
                df, self.table_path, partition_by=["factor_name", "date"]
            )

        elif self.table_config.partition_strategy == PartitionStrategy.NONE:
            self.writer.write_parquet(df, f"{self.table_path}.parquet")


class StockDataTable(DataTable):
    """股票数据表"""

    def __init__(self, table_config: TableConfig):
        super().__init__(table_config)
        stock_dir = DB_ROOT_DIR / "stock"
        stock_dir.mkdir(parents=True, exist_ok=True)
        self.table_path = stock_dir / table_config.name


class StockFactorTable(DataTable):
    """股票因子数据表"""

    def __init__(self, table_config: TableConfig):
        super().__init__(table_config)
        factor_dir = DB_ROOT_DIR / "factor"
        factor_dir.mkdir(parents=True, exist_ok=True)
        self.table_path = factor_dir / table_config.name
