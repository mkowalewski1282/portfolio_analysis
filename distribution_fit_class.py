import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class DistributionFit():
    def __init__(self) -> None:
        pass

    def load_df_from_csv(self, file_path):
        self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    def set_df(self, new_df):
        self.data = new_df

    def fit_distributions_one_stock(self, stock_returns):
        normal_params = stats.norm.fit(stock_returns)
        t_params = stats.t.fit(stock_returns)
        return normal_params, t_params

    def fit_distribution_all_stocks(self):
        self.fitted_params = {}
        for stock in self.data.columns:
            stock_returns = self.data[stock]
            normal_params, t_params = self.fit_distributions_one_stock(stock_returns)
            self.fitted_params[stock] = {"normal": normal_params, "t-student": t_params}

    @staticmethod
    def truncated_cdf(x, dist_cdf, lower_bound, upper_bound):
        cdf_lower_bound = dist_cdf(lower_bound)
        cdf_upper_bound = dist_cdf(upper_bound)
        return (dist_cdf(x) - cdf_lower_bound) / (cdf_upper_bound - cdf_lower_bound)

    @staticmethod
    def get_truncated_boundaries(stock_returns):
        return stock_returns.min(), stock_returns.max()

    @staticmethod
    def truncated_pdf(x, dist_pdf, dist_cdf, lower_bound, upper_bound):
        """
        Compute the truncated PDF for a given distribution in the interval [lower_bound, upper_bound].

        Parameters:
        x: Values for which the truncated PDF is computed.
        dist_pdf: The PDF of the untruncated distribution.
        dist_cdf: The CDF of the untruncated distribution.
        lower_bound: Lower truncation limit.
        upper_bound: Upper truncation limit.

        Returns:
        Truncated PDF values for x.
        """
        cdf_lower_bound = dist_cdf(lower_bound)
        cdf_upper_bound = dist_cdf(upper_bound)
        pdf_x = dist_pdf(x)

        return pdf_x / (cdf_upper_bound - cdf_lower_bound)