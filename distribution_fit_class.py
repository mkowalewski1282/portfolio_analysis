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

    def truncated_quantile(self, q, dist_ppf, dist_cdf, lower_bound, upper_bound):
        """
        Compute the quantile for a truncated distribution at the given probability q.

        Parameters:
        q: The quantile level (between 0 and 1).
        dist_ppf: The PPF (percent-point function) or inverse CDF of the untruncated distribution.
        dist_cdf: The CDF of the untruncated distribution.
        lower_bound: Lower truncation limit.
        upper_bound: Upper truncation limit.

        Returns:
        Quantile value at the specified probability q for the truncated distribution.
        """
        cdf_lower_bound = dist_cdf(lower_bound)
        cdf_upper_bound = dist_cdf(upper_bound)

        # Adjusted quantile for the truncated distribution
        adjusted_q = q * (cdf_upper_bound - cdf_lower_bound) + cdf_lower_bound
        return dist_ppf(adjusted_q)

    def get_truncated_boundaries(self, stock_returns):
        return stock_returns.min(), stock_returns.max()

    def get_truncated_quantile_normal_dist(self, quantile, stock):
        stock_returns = self.get_data()[stock]
        lower_bound, upper_bound = self.get_truncated_boundaries(stock_returns)
        normal_params = self.fitted_params[stock]["normal"]
        mean, std = normal_params
        return self.truncated_quantile(quantile,
                                    lambda q: stats.norm.ppf(q, loc=mean, scale=std),
                                    lambda x: stats.norm.cdf(x, loc=mean, scale=std),
                                    lower_bound, upper_bound)

    def get_truncated_quantile_t_dist(self, quantile, stock):
        stock_returns = self.get_data()[stock]
        lower_bound, upper_bound = self.get_truncated_boundaries(stock_returns)
        t_params = self.fitted_params[stock]["t-student"]
        df, loc, scale = t_params
        return self.truncated_quantile(quantile,
                                    lambda q: stats.t.ppf(q, df, loc=loc, scale=scale),
                                    lambda x: stats.t.cdf(x, df, loc=loc, scale=scale),
                                    lower_bound, upper_bound)

    def load_simulated_data_from_csv(self, file_path):
        self.simulated_data = pd.read_csv(file_path)

    # def calculate_returns_from_simulated_quantiles(self):
    #     output_norm = pd.DataFrame(columns=self.get_data().columns)
    #     output_t_student = pd.DataFrame(columns=self.get_data().columns)
    #     for stock in self.simulated_data.columns:
    #         for i, quantile in enumerate(self.simulated_data[stock]):
    #             normal_return = self.get_truncated_quantile_normal_dist(quantile, stock)
    #             t_return = self.get_truncated_quantile_t_dist(quantile, stock)
    #             output_norm.at[i, stock] = normal_return
    #             output_t_student.at[i, stock] = t_return
    #     self.simulated_return_norm = output_norm
    #     self.simulated_return_t_student = output_t_student

    def calculate_returns_from_simulated_quantiles(self):
        output_norm = pd.DataFrame(columns=self.get_data().columns, index=self.simulated_data.index)
        output_t_student = pd.DataFrame(columns=self.get_data().columns, index=self.simulated_data.index)

        for stock in self.simulated_data.columns:
            normal_params = self.fitted_params[stock]["normal"]
            t_params = self.fitted_params[stock]["t-student"]
            mean, std = normal_params
            df, loc, scale = t_params
            stock_returns = self.get_data()[stock]
            lower_bound, upper_bound = self.get_truncated_boundaries(stock_returns)

            cdf_lower_bound = stats.norm.cdf(lower_bound, loc=mean, scale=std)
            cdf_upper_bound = stats.norm.cdf(upper_bound, loc=mean, scale=std)
            normal_adjusted_q = self.simulated_data[stock] * (cdf_upper_bound - cdf_lower_bound) + cdf_lower_bound
            normal_returns = stats.norm.ppf(normal_adjusted_q, loc=mean, scale=std)
            output_norm[stock] = normal_returns

            cdf_lower_bound_t = stats.t.cdf(lower_bound, df, loc=loc, scale=scale)
            cdf_upper_bound_t = stats.t.cdf(upper_bound, df, loc=loc, scale=scale)
            t_adjusted_q = self.simulated_data[stock] * (cdf_upper_bound_t - cdf_lower_bound_t) + cdf_lower_bound_t
            t_returns = stats.t.ppf(t_adjusted_q, df, loc=loc, scale=scale)
            output_t_student[stock] = t_returns
        self.simulated_return_norm = output_norm
        self.simulated_return_t_student = output_t_student

    def get_simulated_return_norm(self):
        return self.simulated_return_norm

    def get_simulated_return_t_student(self):
        return self.simulated_return_t_student

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

    def fit_multivariate_distributions(self, degrees_of_freedom):
        data = self.data.dropna()  # Drop rows with missing values for multivariate fitting
        mean_vector = data.mean().values
        covariance_matrix = data.cov().values

        # Multivariate Normal
        mvn_params = {
            "mean": mean_vector,
            "covariance": covariance_matrix
        }
        # Fit Multivariate t-Student
        t_params = {
            "mean": mean_vector,
            "covariance": covariance_matrix,
            "df": degrees_of_freedom
        }

        self.multivariate_fitted_params = {
            "multivariate_normal": mvn_params,
            "multivariate_t-student": t_params
        }

    def get_multivariate_fitted_params(self):
        return self.multivariate_fitted_params

    def generate_multivariate_normal_samples(self, n):
        params = self.multivariate_fitted_params["multivariate_normal"]
        mean = params["mean"]
        covariance = params["covariance"]
        samples = np.random.multivariate_normal(mean, covariance, size=n)
        return samples

    def generate_multivariate_t_samples(self, n):
        params = self.multivariate_fitted_params["multivariate_t-student"]
        mean = params["mean"]
        covariance = params["covariance"]
        df = params["df"]  # degrees of freedom

        d = len(mean)  # Dimensionality
        g = np.random.gamma(df / 2., 2. / df, size=n)  # Gamma distribution samples for scaling
        z = np.random.multivariate_normal(np.zeros(d), covariance, size=n)  # Multivariate normal samples
        samples = mean + z / np.sqrt(g)[:, None]  # Scale the samples to create t-distribution samples
        return samples

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

        title_to_add = ""
        if with_not_truncated:
            x = np.linspace(-1, 2, 100)
            plt.plot(x, stats.norm.pdf(x, *normal_params), 'y-', lw=2, label='Normal Fit')
            plt.plot(x, stats.t.pdf(x, *t_params), 'g-', lw=2, label='t-Student Fit')
            title_to_add = ". Not truncated PDFs added."

        lower_bound, upper_bound = self.get_truncated_boundaries(stock_returns)
        x = np.linspace(lower_bound, upper_bound, 100)
        # Truncated PDF for Normal distribution
        plt.plot(x, self.truncated_pdf(x, lambda x: stats.norm.pdf(x, *normal_params), lambda x: stats.norm.cdf(x, *normal_params), lower_bound, upper_bound), 'r-', lw=2, label='Truncated Normal PDF')
        # Truncated PDF for t-Student distribution
        plt.plot(x, self.truncated_pdf(x, lambda x: stats.t.pdf(x, *t_params), lambda x: stats.t.cdf(x, *t_params), lower_bound, upper_bound), 'b-', lw=2, label='Truncated t-Student PDF')
        title = f"Truncated PDFs for {stock} in range [{lower_bound}, {upper_bound}]"

        # if with_not_truncated:
        #     x = np.linspace(-1, 2, 100)
        #     plt.plot(x, stats.norm.pdf(x, *normal_params), 'y-', lw=2, label='Normal Fit')
        #     plt.plot(x, stats.t.pdf(x, *t_params), 'g-', lw=2, label='t-Student Fit')
        title += title_to_add
        plt.xlabel("Returns")
        plt.ylabel("Density")
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