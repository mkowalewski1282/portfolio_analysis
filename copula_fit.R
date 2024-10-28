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


plot_density <- function(simulated_data) {
  x <- simulated_data[, 1]
  y <- simulated_data[, 2]
  density_estimate <- kde2d(x, y, n = 50)
  par(mfrow = c(1, 2))  # layout
  image(density_estimate, main = "2D Density Plot")
  contour(density_estimate, add = TRUE) # image with contour overlay
  persp(density_estimate$x, density_estimate$y, density_estimate$z,
        theta = 30, phi = 20, expand = 0.5, col = "lightblue",
        xlab = "X", ylab = "Y", zlab = "Density", main = "3D Density Plot")
  par(mfrow = c(1, 1))
}


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

plot_density(simulated_data_clayton)


write.csv(simulated_data_clayton, file = "data/copulas_outputs/simulated_clayton_random_22_stocks.csv", row.names = FALSE)


###### GAUSSIAN COPULA

gaussian_cop = normalCopula(dim = dim, dispstr = "un")  # Use "un" for an unstructured copula
fit_gaussian = fitCopula(gaussian_cop, pseudo_obs, method = "ml")  # Maximum Likelihood Estimation

rho = coef(fit_gaussian)

simulated_data_gaussian <- rCopula(100000, normalCopula(rho, dim, dispstr = "un"))
colnames(simulated_data_gaussian) = stocks

# Plot Gaussian copula simulation for the first two stocks
plot(simulated_data_gaussian[,1], simulated_data_gaussian[,2], 
     main = "Simulated Gaussian Copula Data",
     xlab = "Stock 1", ylab = "Stock 2")

plot_density(simulated_data_gaussian)


write.csv(simulated_data_gaussian, file = "data/copulas_outputs/simulated_gaussian_22_stocks.csv", row.names = FALSE)


###### t - STUDENT COPULA

library(QRM)
# Set degrees of freedom and initialize correlation parameters
df <- 4  # Set the degrees of freedom for the t-Student copula
cor_matrix <- diag(1, dim)  # Start with an identity matrix
cor_matrix[lower.tri(cor_matrix)] <- 0.5  # Example initial value for off-diagonal correlations

# Extract the lower triangular values for unstructured "un" copula
rho_vector <- cor_matrix[lower.tri(cor_matrix)]

# Define the t-Student copula with the extracted correlation vector
t_cop <- tCopula(param = rho_vector, dim = dim, df = df, dispstr = "un")

# Fit the t-Student copula to the pseudo-observations
fit_t <- fitCopula(t_cop, pseudo_obs, method = "ml")

# Extract fitted parameters and degrees of freedom
params_t <- coef(fit_t)
rho_t <- params_t[1:(length(params_t)-1)]  # Extract correlation parameters
df_t <- params_t[length(params_t)]          # Degrees of freedom

# Simulate data from the fitted t-Student copula
simulated_data_t <- rCopula(100000, tCopula(rho_t, dim, df = df_t, dispstr = "un"))
colnames(simulated_data_t) <- stocks

# Plot the simulated data for the first two stocks
plot(simulated_data_t[,1], simulated_data_t[,2],
     main = "Simulated t-Student Copula Data",
     xlab = "Stock 1", ylab = "Stock 2")

# Use the plot_density function to create density plots
plot_density(simulated_data_t)

# Write the simulated data to a CSV file
write.csv(simulated_data_t, file = "data/copulas_outputs/simulated_t_student_22_stocks.csv", row.names = FALSE)




window_length = 32
for(i in 0:(nrow(return_rates) - window_length)){
  library(copula)
  print(i)
  return_rates_8_years = return_rates[(1+i):(window_length+i),]
  stocks = colnames(return_rates_8_years)[2:ncol(return_rates_8_years)]
  ecdf_list = lapply(return_rates_8_years[stocks], ecdf)
  pseudo_obs_8_years = apply(return_rates_8_years[stocks], 2, function(x) rank(x)/(length(x)+1))
  dim_8_years = ncol(pseudo_obs_8_years) 
  
  clayton_cop_8_years = claytonCopula(dim=dim_8_years)
  clayton_fit_8_years = fitCopula(clayton_cop_8_years, pseudo_obs_8_years, method="ml")   # maximum likelihood estimation (ml)
  theta_8_years = coef(clayton_fit_8_years)
  simulated_data_clayton_8_years = rCopula(100000, claytonCopula(theta_8_years, dim_8_years))
  colnames(simulated_data_clayton_8_years) = stocks
  file_name = paste0("data/copulas_outputs/simulated_clayton_random_22_stocks_", i, "_window.csv")
  write.csv(simulated_data_clayton_8_years, file = file_name, row.names = FALSE)
  
  gaussian_cop_8_years = normalCopula(dim = dim_8_years, dispstr = "un")  # Use "un" for an unstructured copula
  init_params <- rep(0, length = dim_8_years * (dim_8_years - 1) / 2)
  fit_gaussian_8_years = fitCopula(gaussian_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params)  # Maximum Likelihood Estimation
  rho_8_years = coef(fit_gaussian_8_years)
  simulated_data_gaussian_8_years <- rCopula(100000, normalCopula(rho_8_years, dim_8_years, dispstr = "un"))
  colnames(simulated_data_gaussian_8_years) = stocks
  file_name = paste0("data/copulas_outputs/simulated_gaussian_22_stocks_", i, "_window.csv")
  write.csv(simulated_data_gaussian_8_years, file = file_name, row.names = FALSE)
  
  library(QRM)
  df_8_years <- 4  
  cor_matrix_8_years <- diag(1, dim_8_years)  
  cor_matrix_8_years[lower.tri(cor_matrix_8_years)] <- 0.5  
  rho_vector_8_years <- cor_matrix_8_years[lower.tri(cor_matrix_8_years)]
  t_cop_8_years <- tCopula(param = rho_vector_8_years, dim = dim_8_years, df = df_8_years, dispstr = "un")
  fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "ml")
  params_t_8_years <- coef(fit_t_8_years)
  rho_t_8_years <- params_t_8_years[1:(length(params_t_8_years)-1)]  
  df_t_8_years <- params_t_8_years[length(params_t_8_years)]          
  simulated_data_t_8_years <- rCopula(100000, tCopula(rho_t_8_years, dim_8_years, df = df_t_8_years, dispstr = "un"))
  colnames(simulated_data_t_8_years) <- stocks
  file_name = paste0("data/copulas_outputs/simulated_t_student_22_stocks_", i, "_window.csv")
  write.csv(simulated_data_t, file = file_name, row.names = FALSE)
  
}
  












































