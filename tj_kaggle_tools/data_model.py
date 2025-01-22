import polars as pl

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import numpy as np

class DataModel:
    def __init__(
        self,
        df: pl.DataFrame,  # train data 和 eval data
        eval_rate: float = 0.2,  # 20%的数据作为eval data
        numeric_cols=[],  # 直接作为输入的数值列
        te_category_cols=[],  # 需要做target encoding的类别列
        ohe_category_cols=[],  # 需要做one-hot encoding的类别列
        target_col="num_sold",  # 目标列
        target_na_fill=0,  # 目标列的空值填充策略, int/float: 使用具体值填充, "mean" : 均值填充, ("mean", (col1, col2)): 按照col1, col2分组后均值填充
        date_col="date",  # 日期列/天
        time_col="timestamp",  # 时间列/小时/分钟/秒，暂时不用
    ):
        self.df = df
        self.eval_rate = eval_rate
        self.numeric_cols = numeric_cols
        self.te_category_cols = te_category_cols
        self.ohe_category_cols = ohe_category_cols
        self.target_col = target_col
        self.target_na_fill = target_na_fill
        self.date_col = date_col
        self.time_col = time_col

        # Split data into train and eval sets
        self.train_df, self.eval_df = self._split_data()

        # Handle missing values in the target column
        self._handle_missing_target()

        # Apply target encoding to specified category columns
        self._apply_target_encoding()

        # Apply one-hot encoding to specified category columns
        # self._apply_one_hot_encoding()

    def _split_data(self):
        """Split the data into training and evaluation sets."""
        train_df, eval_df = train_test_split(self.df, test_size=self.eval_rate, random_state=42)
        return train_df, eval_df

    def _handle_missing_target(self):
        """Handle missing values in the target column."""
        if isinstance(self.target_na_fill, (int, float)):
            self.train_df = self.train_df.with_columns(
                pl.col(self.target_col).fill_null(value=self.target_na_fill)
            )
            self.eval_df = self.eval_df.with_columns(
                pl.col(self.target_col).fill_null(value=self.target_na_fill)
            )
        elif self.target_na_fill == "mean":
            mean_value = self.train_df[self.target_col].mean()
            self.train_df = self.train_df.fill_null(self.target_col, mean_value)
            self.eval_df = self.eval_df.fill_null(self.target_col, mean_value)
        elif isinstance(self.target_na_fill, tuple) and self.target_na_fill[0] == "mean":
            group_cols = self.target_na_fill[1]
            mean_values = self.train_df.groupby(group_cols).agg(pl.col(self.target_col).mean().alias("mean_value"))
            self.train_df = self.train_df.join(mean_values, on=group_cols, how="left")
            self.train_df = self.train_df.with_column(pl.when(pl.col(self.target_col).is_null()).then(pl.col("mean_value")).otherwise(pl.col(self.target_col)).alias(self.target_col))
            self.eval_df = self.eval_df.join(mean_values, on=group_cols, how="left")
            self.eval_df = self.eval_df.with_column(pl.when(pl.col(self.target_col).is_null()).then(pl.col("mean_value")).otherwise(pl.col(self.target_col)).alias(self.target_col))

    def _apply_target_encoding(self):
        """Apply target encoding to specified category columns using the TargetEncoder library."""
        # Initialize the TargetEncoder with the specified columns
        self.te_encoder = TargetEncoder(cols=self.te_category_cols)
        
        # Fit the encoder on the training data
        self.te_encoder.fit(
            self.train_df.select(self.te_category_cols).to_pandas(),
            self.train_df[self.target_col].to_numpy()
        )
        
        # Transform both train and eval data
        train_encoded = self.te_encoder.transform(
            self.train_df.select(self.te_category_cols).to_pandas()
        )
        eval_encoded = self.te_encoder.transform(
            self.eval_df.select(self.te_category_cols).to_pandas()
        )
        
        # Add the encoded columns to the DataFrames
        for col in self.te_category_cols:
            self.train_df = self.train_df.with_columns(
                pl.Series(f"{col}_target_encoded", train_encoded[col].to_numpy())
            )
            self.eval_df = self.eval_df.with_columns(
                pl.Series(f"{col}_target_encoded", eval_encoded[col].to_numpy())
            )

    def _apply_one_hot_encoding(self):
        """Apply one-hot encoding to specified category columns."""
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        for col in self.ohe_category_cols:
            # Fit on train data
            train_encoded = encoder.fit_transform(self.train_df[col].to_numpy().reshape(-1, 1))
            eval_encoded = encoder.transform(self.eval_df[col].to_numpy().reshape(-1, 1))

            # Create new columns for the encoded data
            for i in range(train_encoded.shape[1]):
                self.train_df = self.train_df.with_column(pl.Series(f"{col}_ohe_{i}", train_encoded[:, i]))
                self.eval_df = self.eval_df.with_column(pl.Series(f"{col}_ohe_{i}", eval_encoded[:, i]))

    def get_train_data(self):
        """Return the processed training data."""
        return self.train_df

    def get_eval_data(self):
        """Return the processed evaluation data."""
        return self.eval_df
