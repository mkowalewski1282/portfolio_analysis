# # Line below has to be run only once
# install.packages("copula")
# install.packages("plotly")
# install.packages("MASS")
# install.packages("ks")
# install.packages("QRM")

library(copula)
library(plotly)
library(MASS)
library(ks)
library(QRM)
library(Matrix) 

# file = "data/positive_quarterly_return_rates.csv"
file = "data/random_data/n_stocks_per_sector.csv"
return_rates = read.csv(file)
stocks = colnames(return_rates)[2:ncol(return_rates)]

ecdf_list = lapply(return_rates[stocks], ecdf)

## example plot
plot(ecdf(return_rates$MSFT), main="ECDF of MSFT", xlab="Return", ylab="ECDF")

pseudo_obs = apply(return_rates[stocks], 2, function(x) rank(x)/(length(x)+1))
dim = ncol(pseudo_obs) 



###### CLAYTON
clayton_cop = claytonCopula(dim=dim)
clayton_fit = fitCopula(clayton_cop, pseudo_obs, method="ml")   # maximum likelihood estimation (ml)

# summary(fit)
theta = coef(clayton_fit)
# print(theta)

simulated_data_clayton = rCopula(100000, claytonCopula(theta, dim))
colnames(simulated_data_clayton) = stocks
# head(simulated_data)

# first two stocks -> countour plot
plot(simulated_data_clayton[,1], simulated_data_clayton[,2], main="Simulated Clayton Copula Data",
     xlab="Stock 1", ylab="Stock 2")

x = simulated_data_clayton[, 1]
y = simulated_data_clayton[, 2]


density_estimate <- kde2d(x, y, n = 50)  # n specifies the resolution of the density grid

image(density_estimate, main = "2D Density Plot")
contour(density_estimate, add = TRUE)
persp(density_estimate$x, density_estimate$y, density_estimate$z,
      theta = 30, phi = 20, expand = 0.5, col = "lightblue",
      xlab = "X", ylab = "Y", zlab = "Density", main = "3D Density Plot")


write.csv(simulated_data_clayton, file = "data/copulas_outputs/simulated_clayton_random_22_stocks.csv", row.names = FALSE)


###### GAUSSIAN COPULA

gaussian_cop = normalCopula(dim = dim, dispstr = "un")  # Use "un" for an unstructured copula
fit_gaussian = fitCopula(gaussian_cop, pseudo_obs, method = "ml")  # Maximum Likelihood Estimation

rho = coef(fit_gaussian)

simulated_data_gaussian <- rCopula(1000, normalCopula(rho, dim, dispstr = "un"))
colnames(simulated_data_gaussian) = stocks

# Plot Gaussian copula simulation for the first two stocks
plot(simulated_data_gaussian[,1], simulated_data_gaussian[,2], 
     main = "Simulated Gaussian Copula Data",
     xlab = "Stock 1", ylab = "Stock 2")


write.csv(simulated_data_gaussian, file = "data/copulas_outputs/simulated_gaussian_all_stocks.csv", row.names = FALSE)


###### t - STUDENT COPULA

















































