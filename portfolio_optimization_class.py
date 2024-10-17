from docplex.mp.model import Model
import pandas as pd


class PortfolioOptimization:
    def __init__(self) -> None:
        self.model = Model(name='EVaR Optimization')

    def set_number_of_scenarios(self, number_of_scenarios):
        self._number_of_scenarios = number_of_scenarios

    def set_number_of_instruments(self, number_of_instruments):
        self._number_of_instruments = number_of_instruments

    def set_tau(self, tau):
        self.tau = tau
        self._beta = (1 - tau) / tau

    def set_maximum_weight(self, maximum_weight):
        self.maximum_weight = maximum_weight

    def load_data_from_csv(self, filepath):
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)

    def set_data(self, new_data):
        self.data = new_data

    def get_whole_data_length(self):
        return len(self.data)

    def slice_windows(self, number_of_quarters, start_quarter_index = 0):
        self.sliced_data = self.data.iloc[start_quarter_index:start_quarter_index + number_of_quarters]
        self._number_of_scenarios = self.sliced_data.shape[0]   # number of rows
        self._number_of_instruments = self.sliced_data.shape[1] # number of columns

        self.R = self.sliced_data.values                                                # Scenarios as rows, instruments as columns
        self.p = [1/self._number_of_scenarios] * self._number_of_scenarios  # Equal probabilities for each scenario

    def build_model(self):
        w = self.model.continuous_var_list(self._number_of_instruments, lb=0, name="w")  # weights, w[j] >= 0    # type: ignore
        y = self.model.continuous_var(name="y")                                          # unbounded y variable
        u = self.model.continuous_var_list(self._number_of_scenarios, lb=0, name="u")    # u[i] >= 0             # type: ignore
        v = self.model.continuous_var_list(self._number_of_scenarios, lb=0, name="v")    # v[i] >= 0             # type: ignore

        # minimize EVaR value (min y)
        self.model.minimize(y)

        # Main constraints for each scenario i
        for i in range(self._number_of_scenarios):
            self.model.add_constraint(
                self.p[i] * y - u[i] + v[i] >= -self.p[i] * self.model.sum(self.R[i, j] * w[j] for j in range(self._number_of_instruments))
            )

        # u_v_constraints
        self.model.add_constraint(self.model.sum(u[i] for i in range(self._number_of_scenarios)) -
                                  self._beta * self.model.sum(v[i] for i in range(self._number_of_scenarios)) >= 0)

        # weights_sum
        self.model.add_constraint(self.model.sum(w[j] for j in range(self._number_of_instruments)) == 1)

        # max_weight
        for j in range(self._number_of_instruments):
            self.model.add_constraint(w[j] <= self.maximum_weight)

    def solve(self, verbose=True):
        self._solution = self.model.solve()
        if verbose:
            if self._solution:
                print(f"Objective value (EVaR): {self.model.get_var_by_name('y').solution_value}")
                print("Weights (w):", [self.model.get_var_by_name(f'w_{j}').solution_value for j in range(self._number_of_instruments)])
            else:
                print("No solution found")

    def get_solution_weights(self):
        return [self.model.get_var_by_name(f'w_{j}').solution_value for j in range(self._number_of_instruments)]

    def get_solution_evar(self):
        return self.model.get_var_by_name('y').solution_value


def optimize_windows(number_of_quarters, tau, maximum_weight, data_path, verbose = False):
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data_length = len(data)
    output_df = pd.DataFrame()
    starts = []
    ends = []
    quarter_numbers = []
    evars = []
    stocks_number = len(data.columns)
    weights_lists = [[] for i in range(stocks_number)]
    for start_index in range(data_length - number_of_quarters):
        # print(start_index)
        optimizer = PortfolioOptimization()
        optimizer.set_tau(tau)
        optimizer.set_maximum_weight(maximum_weight)
        optimizer.set_data(data)
        # print(data)
        optimizer.slice_windows(number_of_quarters, start_index)
        optimizer.build_model()
        optimizer.solve(verbose=verbose)
        weights = optimizer.get_solution_weights()
        evar = optimizer.get_solution_evar()
        starts.append(optimizer.sliced_data.index[0])
        ends.append(optimizer.sliced_data.index[-1])
        quarter_numbers.append(number_of_quarters)
        evars.append(evar)
        weight_index = 0
        for weight in weights:
            weights_lists[weight_index].append(weight)
            weight_index += 1

    output_df["Start"] = starts
    output_df["End"] = ends
    output_df["Number of quarters"] = quarter_numbers
    output_df["EVAR"] = evars
    index = 0
    for weights in weights_lists:
        # print(index)
        output_df[f"w{index}"] = weights
        index += 1

    return output_df

