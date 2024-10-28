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

    def get_data(self):
        return self.data

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

    def get_fitted_params(self):
        return self.fitted_params

    def truncated_cdf(self, x, dist_cdf, lower_bound, upper_bound):
        cdf_lower_bound = dist_cdf(lower_bound)
        cdf_upper_bound = dist_cdf(upper_bound)
        return (dist_cdf(x) - cdf_lower_bound) / (cdf_upper_bound - cdf_lower_bound)

    def get_truncated_boundaries(self, stock_returns):
        return stock_returns.min(), stock_returns.max()

    def truncated_pdf(self, x, dist_pdf, dist_cdf, lower_bound, upper_bound):
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

    def plot_fitted_distributions(self, stock, stock_returns, normal_params, t_params):
        plt.figure(figsize=(10, 6))
        plt.hist(stock_returns, bins='auto', density=True, alpha=0.6, color='g', label="Return Data")
        x = np.linspace(-1, 2, 100)
        plt.plot(x, stats.norm.pdf(x, *normal_params), 'r-', lw=2, label='Normal Fit')
        plt.plot(x, stats.t.pdf(x, *t_params), 'b-', lw=2, label='t-Student Fit')
        plt.title(f"Fitted Distributions for {stock}")
        plt.legend()
        plt.show()

    def plot_truncated_pdfs(self, stock, stock_returns, normal_params, t_params, with_not_truncated):
        plt.figure(figsize=(10, 6))
        plt.hist(stock_returns, bins='auto', density=True, alpha=0.6, color='g', label="Return Data")

        lower_bound, upper_bound = self.get_truncated_boundaries(stock_returns)
        x = np.linspace(lower_bound, upper_bound, 100)
        # Truncated PDF for Normal distribution
        plt.plot(x, self.truncated_pdf(x, lambda x: stats.norm.pdf(x, *normal_params), lambda x: stats.norm.cdf(x, *normal_params), lower_bound, upper_bound), 'r-', lw=2, label='Truncated Normal PDF')
        # Truncated PDF for t-Student distribution
        plt.plot(x, self.truncated_pdf(x, lambda x: stats.t.pdf(x, *t_params), lambda x: stats.t.cdf(x, *t_params), lower_bound, upper_bound), 'b-', lw=2, label='Truncated t-Student PDF')
        title = f"Truncated PDFs for {stock} in range [{lower_bound}, {upper_bound}]"

        if with_not_truncated:
            x = np.linspace(-1, 2, 100)
            plt.plot(x, stats.norm.pdf(x, *normal_params), 'y-', lw=2, label='Normal Fit')
            plt.plot(x, stats.t.pdf(x, *t_params), 'g-', lw=2, label='t-Student Fit')
            title += ". Not truncated PDFs added."
        plt.title(title)
        plt.legend()
        plt.show()


    def plot_fitted_cdfs(self, stock, stock_returns, normal_params, t_params):
        plt.figure(figsize=(10, 6))
        x = np.linspace(-1, 2, 100)
        plt.plot(x, stats.norm.cdf(x, *normal_params), 'r-', lw=2, label='Normal CDF')
        plt.plot(x, stats.t.cdf(x, *t_params), 'b-', lw=2, label='t-Student CDF')
        sorted_returns = np.sort(stock_returns)
        empirical_cdf = np.arange(1, len(sorted_returns)+1) / len(sorted_returns)
        plt.step(sorted_returns, empirical_cdf, where='post', label='Empirical CDF', color='g')
        plt.title(f"Fitted Distributions for {stock} - CDF")
        plt.legend()
        plt.show()


    def plot_truncated_cdfs(self, stock, stock_returns, normal_params, t_params):
        plt.figure(figsize=(10, 6))
        lower_bound, upper_bound = self.get_truncated_boundaries(stock_returns)
        x = np.linspace(lower_bound, upper_bound, 100)

        # Truncated CDF for Normal distribution
        plt.plot(x, self.truncated_cdf(x, lambda x: stats.norm.cdf(x, *normal_params), lower_bound, upper_bound), 'r-', lw=2, label='Truncated Normal CDF')
        # Truncated CDF for t-Student distribution
        plt.plot(x, self.truncated_cdf(x, lambda x: stats.t.cdf(x, *t_params), lower_bound, upper_bound), 'b-', lw=2, label='Truncated t-Student CDF')
        # Empirical CDF
        sorted_returns = np.sort(stock_returns)
        empirical_cdf = np.arange(1, len(sorted_returns)+1) / len(sorted_returns)
        plt.step(sorted_returns, empirical_cdf, where='post', label='Empirical CDF', color='g')

        plt.title(f"Truncated CDFs for {stock} in range [{lower_bound}, {upper_bound}]")
        plt.legend()
        plt.show()