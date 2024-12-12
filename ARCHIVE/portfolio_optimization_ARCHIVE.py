from portfolio_optimization_class import PortfolioOptimization, optimize_windows


optimizer = PortfolioOptimization()
optimizer.set_tau(0.05)
optimizer.set_maximum_weight(0.3)
optimizer.load_data_from_csv(r'data\random_data\n_stocks_per_sector.csv')
optimizer.slice_windows(optimizer.get_whole_data_length(), 0)
optimizer.build_model()
optimizer.solve()

weights = optimizer.get_solution_weights()
evar = optimizer.get_solution_evar()

print(weights)
print(evar)
