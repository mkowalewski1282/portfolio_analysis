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
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)

        self._number_of_scenarios = data.shape[0]   # number of rows
        self._number_of_instruments = data.shape[1] # number of columns

        self.R = data.values                                                # Scenarios as rows, instruments as columns
        self.p = [1/self._number_of_scenarios] * self._number_of_scenarios  # Equal probabilities for each scenario

    def build_model(self):
        w = self.model.continuous_var_list(self._number_of_instruments, lb=0, name="w")  # weights, w[j] >= 0
        y = self.model.continuous_var(name="y")                                          # unbounded y variable
        u = self.model.continuous_var_list(self._number_of_scenarios, lb=0, name="u")    # u[i] >= 0
        v = self.model.continuous_var_list(self._number_of_scenarios, lb=0, name="v")    # v[i] >= 0

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

    def solve(self):
        self._solution = self.model.solve()
        if self._solution:
            print(f"Objective value (EVaR): {self.model.get_var_by_name('y').solution_value}")
            print("Weights (w):", [self.model.get_var_by_name(f'w_{j}').solution_value for j in range(self._number_of_instruments)])
        else:
            print("No solution found")

    def get_solution(self):
        return self._solution
