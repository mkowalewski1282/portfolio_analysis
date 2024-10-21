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

return_rates = read.csv("data/positive_quarterly_return_rates.csv")
stocks = colnames(return_rates)[2:ncol(return_rates)]

ecdf_list = lapply(return_rates[stocks], ecdf)

## example plot
plot(ecdf(return_rates$AAPL), main="ECDF of AAPL", xlab="Return", ylab="ECDF")

pseudo_obs = apply(return_rates[stocks], 2, function(x) rank(x)/(length(x)+1))
dim = ncol(pseudo_obs) 



###### CLAYTON
clayton_cop = claytonCopula(dim=dim)
clayton_fit = fitCopula(clayton_cop, pseudo_obs, method="ml")   # maximum likelihood estimation (ml)

# summary(fit)
theta = coef(clayton_fit)
# print(theta)

simulated_data_clayton = rCopula(1000, claytonCopula(theta, dim))
colnames(simulated_data_clayton) = stocks
# head(simulated_data)

# first two stocks -> countour plot
plot(simulated_data_clayton[,1], simulated_data_clayton[,2], main="Simulated Clayton Copula Data",
     xlab="Stock 1", ylab="Stock 2")

write.csv(simulated_data_clayton, file = "data/copulas_outputs/simulated_clayton_all_stocks.csv", row.names = FALSE)


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

















































