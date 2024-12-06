# TEST

library(copula)
library(plotly)
library(MASS)
library(ks)
library(QRM)
library(Matrix) 

# file = "data/positive_quarterly_return_rates.csv"
# file = "data/random_data/n_stocks_per_sector.csv"
setwd("C:/Studia/Studia/Studia/Praca inżynierska/portfolio_analysis")
file = "data/random_data/1_stocks_per_sector.csv"
return_rates = read.csv(file)
stocks = colnames(return_rates)[2:ncol(return_rates)]

ecdf_list = lapply(return_rates[stocks], ecdf)

## example plot
# plot(ecdf(return_rates$MSFT), main="ECDF of MSFT", xlab="Return", ylab="ECDF") #might not work if MSFT not in samples ofc

pseudo_obs = apply(return_rates[stocks], 2, function(x) rank(x)/(length(x)+1))
dim = ncol(pseudo_obs) 





plot_density <- function(simulated_data) {
  x <- simulated_data[, 1]
  y <- simulated_data[, 2]
  density_estimate <- kde2d(x, y, n = 50)
  par(mfrow = c(1, 2))  # layout
  image(density_estimate, main = "Wykres konturowy", xlab = "u1", ylab = "u2")
  contour(density_estimate, add = TRUE) # image with contour overlay
  persp(density_estimate$x, density_estimate$y, density_estimate$z,
        theta = 30, phi = 20, expand = 0.5, col = "lightblue",
        xlab = "u1", ylab = "u2", zlab = "", main = "Wykres gętości",
        ticktype = "detailed", axes = TRUE)
  par(mfrow = c(1, 1))
}




###### CLAYTON
clayton_cop = claytonCopula(dim=dim)
clayton_fit = fitCopula(clayton_cop, pseudo_obs, method="ml")   # maximum likelihood estimation (ml)

# summary(fit)
theta = coef(clayton_fit)
theta = 0.2 # ONLY FOR THESIS
theta = 0.5 # ONLY FOR THESIS
theta = 5   # ONLY FOR THESIS
# print(theta)
simulated_data_clayton = rCopula(100000, claytonCopula(theta, dim))
colnames(simulated_data_clayton) = stocks
# head(simulated_data)
# first two stocks -> countour plot
plot(simulated_data_clayton[,1], simulated_data_clayton[,2], main="Simulated Clayton Copula Data",
     xlab="Stock 1", ylab="Stock 2")
plot_density(simulated_data_clayton)
#write.csv(simulated_data_clayton, file = "data/copulas_outputs/simulated_clayton_random_22_stocks.csv", row.names = FALSE)


###### GAUSSIAN COPULA

gaussian_cop = normalCopula(dim = dim, dispstr = "un")  # Use "un" for an unstructured copula
fit_gaussian = fitCopula(gaussian_cop, pseudo_obs, method = "ml")  # Maximum Likelihood Estimation
rho = coef(fit_gaussian)
dim = 2    # ONLY FOR THESIS
rho = 0.2  # ONLY FOR THESIS
rho = 0.5  # ONLY FOR THESIS
rho = 0.95 # ONLY FOR THESIS
simulated_data_gaussian <- rCopula(100000, normalCopula(rho, dim, dispstr = "un"))
colnames(simulated_data_gaussian) = stocks
# Plot Gaussian copula simulation for the first two stocks
plot(simulated_data_gaussian[,1], simulated_data_gaussian[,2], 
     main = "Simulated Gaussian Copula Data",
     xlab = "Stock 1", ylab = "Stock 2")
plot_density(simulated_data_gaussian)
#write.csv(simulated_data_gaussian, file = "data/copulas_outputs/simulated_gaussian_22_stocks.csv", row.names = FALSE)


###### t - STUDENT COPULA

library(QRM)

df <- 4  
cor_matrix <- diag(1, dim)  # identity matrix
cor_matrix[lower.tri(cor_matrix)] <- 0.5  # initial value for off-diagonal correlations
# lower triangular values for unstructured "un" copula
rho_vector <- cor_matrix[lower.tri(cor_matrix)]

t_cop <- tCopula(param = rho_vector, dim = dim, df = df, dispstr = "un")
fit_t <- fitCopula(t_cop, pseudo_obs, method = "ml")
params_t <- coef(fit_t)
rho_t <- params_t[1:(length(params_t)-1)]  
df_t <- params_t[length(params_t)]          
 
dim = 2      # ONLY FOR THESIS
rho_t = 0.5  # ONLY FOR THESIS
rho_t = 0.8  # ONLY FOR THESIS
df_t = 1     # ONLY FOR THESIS
df_t = 2     # ONLY FOR THESIS
df_t = 8     # ONLY FOR THESIS

simulated_data_t <- rCopula(100000, tCopula(rho_t, dim, df = df_t, dispstr = "un"))
colnames(simulated_data_t) <- stocks
plot(simulated_data_t[,1], simulated_data_t[,2],
     main = "Simulated t-Student Copula Data",
     xlab = "Stock 1", ylab = "Stock 2")
plot_density(simulated_data_t)

# write.csv(simulated_data_t, file = "data/copulas_outputs/simulated_t_student_22_stocks.csv", row.names = FALSE)


