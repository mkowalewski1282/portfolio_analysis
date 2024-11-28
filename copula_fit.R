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

plot_density <- function(simulated_data) {
  x <- simulated_data[, 1]
  y <- simulated_data[, 2]
  density_estimate <- kde2d(x, y, n = 50)
  par(mfrow = c(1, 2))  # layout
  image(density_estimate, main = "2D Density Plot", xlab = "Stock 1", ylab = "Stock 2")
  contour(density_estimate, add = TRUE) # image with contour overlay
  persp(density_estimate$x, density_estimate$y, density_estimate$z,
        theta = 30, phi = 20, expand = 0.5, col = "lightblue",
        xlab = "Stock 1", ylab = "Stock 2", zlab = "Density", main = "3D Density Plot",
        ticktype = "detailed", axes = TRUE)
  par(mfrow = c(1, 1))
}


## example plot
# plot(ecdf(return_rates$MSFT), main="ECDF of MSFT", xlab="Return", ylab="ECDF") #might not work if MSFT not in samples ofc








# file = "data/positive_quarterly_return_rates.csv"
# file = "data/random_data/n_stocks_per_sector.csv"
for(iter in 2:10){
  setwd("C:/Studia/Studia/Studia/Praca inżynierska/portfolio_analysis")
  file = paste0("data/random_data/1_stocks_per_sector_", iter, "_iter.csv")
  return_rates = read.csv(file)
  stocks = colnames(return_rates)[2:ncol(return_rates)]
  
  ecdf_list = lapply(return_rates[stocks], ecdf)

  pseudo_obs = apply(return_rates[stocks], 2, function(x) rank(x)/(length(x)+1))
  dim = ncol(pseudo_obs) 
  
  
  params_df <- data.frame(
    Window = integer(),
    CopulaType = character(),
    Parameters = character(),
    stringsAsFactors = FALSE
  )
  
  setwd("C:/Studia/Studia/Studia/Praca inżynierska/") # from "C:/Studia/Studia/Studia/Praca inżynierska/portfolio_analysis"
  window_length = 32
  num_of_stocks = 11 # Third batch of stocks
  for(i in 0:(nrow(return_rates) - window_length)){
    library(copula)
    print(i)
    return_rates_8_years = return_rates[(1+i):(window_length+i),]
    stocks = colnames(return_rates_8_years)[2:ncol(return_rates_8_years)]
    ecdf_list = lapply(return_rates_8_years[stocks], ecdf)
    pseudo_obs_8_years = apply(return_rates_8_years[stocks], 2, function(x) rank(x)/(length(x)+1))
    dim_8_years = ncol(pseudo_obs_8_years) 
    
    print("Clayton")
    clayton_cop_8_years = claytonCopula(dim=dim_8_years)
    clayton_fit_8_years = fitCopula(clayton_cop_8_years, pseudo_obs_8_years, method="ml")   # maximum likelihood estimation (ml)
    theta_8_years = coef(clayton_fit_8_years)
    simulated_data_clayton_8_years = rCopula(10000, claytonCopula(theta_8_years, dim_8_years))
    colnames(simulated_data_clayton_8_years) = stocks
    file_name = paste0("copulas_outputs/simulated_clayton_random_", num_of_stocks, "_stocks_", i, "_window_", iter, "_iter.csv")
    write.csv(simulated_data_clayton_8_years, file = file_name, row.names = FALSE)    # PATH CHANGED -> run from wd "Praca inżynierska"
    
    # Append Clayton parameters
    params_df <- rbind(params_df, data.frame(
      Window = i,
      CopulaType = "Clayton",
      Parameters = paste(theta_8_years, collapse = ", ")
    ))
    
    print("Gauss")
    gaussian_cop_8_years = normalCopula(dim = dim_8_years, dispstr = "un")  # Use "un" for an unstructured copula
    init_params <- rep(0, length = dim_8_years * (dim_8_years - 1) / 2)
    fit_gaussian_8_years = fitCopula(gaussian_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params)  # Maximum Likelihood Estimation
    rho_8_years = coef(fit_gaussian_8_years)
    simulated_data_gaussian_8_years <- rCopula(10000, normalCopula(rho_8_years, dim_8_years, dispstr = "un"))
    colnames(simulated_data_gaussian_8_years) = stocks
    file_name = paste0("copulas_outputs/simulated_gaussian_", num_of_stocks, "_stocks_", i, "_window_", iter, "_iter.csv")
    write.csv(simulated_data_gaussian_8_years, file = file_name, row.names = FALSE)         # PATH CHANGED -> run from wd "Praca inżynierska"
    
    # Append Gaussian parameters
    params_df <- rbind(params_df, data.frame(
      Window = i,
      CopulaType = "Gaussian",
      Parameters = paste(rho_8_years, collapse = ", ")
    ))
    
    library(QRM)
    print("t-Student")
    ####### df_8_years <- 4  
    ####### cor_matrix_8_years <- diag(1, dim_8_years)  
    ####### cor_matrix_8_years[lower.tri(cor_matrix_8_years)] <- 0.5  
    ####### rho_vector_8_years <- cor_matrix_8_years[lower.tri(cor_matrix_8_years)]
    ####### t_cop_8_years <- tCopula(param = rho_vector_8_years, dim = dim_8_years, df = df_8_years, dispstr = "un")
    ####### # init_params <- c(rep(0, length = dim_8_years * (dim_8_years - 1) / 2), 4) # Correlations set to 0, df set to 4
    ####### # fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params)
    ####### fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "itau")
    # params_t_8_years <- coef(fit_t_8_years)
    empirical_tau <- cor(pseudo_obs_8_years, method = "kendall")
    rho_init <- tan(pi * empirical_tau[lower.tri(empirical_tau)] / 2)
    rho_init <- pmin(pmax(rho_init, -0.99), 0.99)  # Truncate to avoid boundary issues
    init_params <- c(rho_init, 4)  # Include df = 4
    #init_params <- c(tan(pi * cor(pseudo_obs_8_years, method = "kendall")[lower.tri(diag(dim_8_years))] / 2), 4)  # Initial rho and df = 4
    #t_cop_8_years <- tCopula(param = init_params[1:(length(init_params) - 1)], dim = dim_8_years, df = init_params[length(init_params)], dispstr = "un")
    t_cop_8_years <- tCopula(param = init_params[1:(length(init_params) - 1)],
                             dim = dim_8_years,
                             df = init_params[length(init_params)],
                             dispstr = "un")
    
    fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params, control = list(maxit = 100, reltol = 1e-6))
    params_t_8_years <- fit_t_8_years@copula@parameters
    rho_t_8_years <- params_t_8_years[1:(length(params_t_8_years)-1)]  
    df_t_8_years <- params_t_8_years[length(params_t_8_years)] 
    simulated_data_t_8_years <- rCopula(10000, tCopula(rho_t_8_years, dim_8_years, df = df_t_8_years, dispstr = "un"))
    colnames(simulated_data_t_8_years) <- stocks
    
    ######### print("t-Student")
    ######### df_8_years <- 4  
    ######### cor_matrix_8_years <- diag(1, dim_8_years)  
    ######### cor_matrix_8_years[lower.tri(cor_matrix_8_years)] <- 0.5  
    ######### rho_vector_8_years <- cor_matrix_8_years[lower.tri(cor_matrix_8_years)]
    ######### t_cop_8_years <- tCopula(param = rho_vector_8_years, dim = dim_8_years, df = df_8_years, dispstr = "un")
    ######### init_params <- c(rep(0, length = dim_8_years * (dim_8_years - 1) / 2), 4) # Correlations set to 0, df set to 4
    ######### fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params)
    ######### params_t_8_years <- coef(fit_t_8_years)
    ######### rho_t_8_years <- params_t_8_years[1:(length(params_t_8_years)-1)]  
    ######### df_t_8_years <- params_t_8_years[length(params_t_8_years)]          
    ######### simulated_data_t_8_years <- rCopula(10000, tCopula(rho_t_8_years, dim_8_years, df = df_t_8_years, dispstr = "un"))
    ######### colnames(simulated_data_t_8_years) <- stocks
    
    file_name = paste0("copulas_outputs/simulated_t_student_", num_of_stocks, "_stocks_", i, "_window_", iter, "_iter.csv")  
    write.csv(simulated_data_t_8_years, file = file_name, row.names = FALSE)                # PATH CHANGED -> run from wd "Praca inżynierska"
    
    # Append T-Copula parameters
    params_df <- rbind(params_df, data.frame(
      Window = i,
      CopulaType = "T",
      Parameters = paste(c(rho_t_8_years, df_t_8_years), collapse = ", ")
    ))
    
    write.csv(params_df, paste0("copulas_outputs/copula_parameters_", num_of_stocks, "_stocks_", iter, "_iter.csv"), row.names = FALSE)
  }
}

  









# print("t-Student")
# df_8_years <- 4  
# cor_matrix_8_years <- diag(1, dim_8_years)  
# cor_matrix_8_years[lower.tri(cor_matrix_8_years)] <- 0.5  
# rho_vector_8_years <- cor_matrix_8_years[lower.tri(cor_matrix_8_years)]
# t_cop_8_years <- tCopula(param = rho_vector_8_years, dim = dim_8_years, df = df_8_years, dispstr = "un")
# # init_params <- c(rep(0, length = dim_8_years * (dim_8_years - 1) / 2), 4) # Correlations set to 0, df set to 4
# # fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params)
# fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "itau")
# # params_t_8_years <- coef(fit_t_8_years)
# params_t_8_years <- fit_t_8_years@copula@parameters
# rho_t_8_years <- params_t_8_years[1:(length(params_t_8_years)-1)]  
# df_t_8_years <- params_t_8_years[length(params_t_8_years)] 
# simulated_data_t_8_years <- rCopula(10000, tCopula(rho_t_8_years, dim_8_years, df = df_t_8_years, dispstr = "un"))
# colnames(simulated_data_t_8_years) <- stocks
# 
# 
# 
# 
# empirical_tau <- cor(pseudo_obs_8_years, method = "kendall")
# rho_init <- tan(pi * empirical_tau[lower.tri(empirical_tau)] / 2)
# rho_init <- pmin(pmax(rho_init, -0.99), 0.99)  # Truncate to avoid boundary issues
# init_params <- c(rho_init, 4)  # Include df = 4
# #init_params <- c(tan(pi * cor(pseudo_obs_8_years, method = "kendall")[lower.tri(diag(dim_8_years))] / 2), 4)  # Initial rho and df = 4
# #t_cop_8_years <- tCopula(param = init_params[1:(length(init_params) - 1)], dim = dim_8_years, df = init_params[length(init_params)], dispstr = "un")
# t_cop_8_years <- tCopula(param = init_params[1:(length(init_params) - 1)],
#                          dim = dim_8_years,
#                          df = init_params[length(init_params)],
#                          dispstr = "un")
# 
# fit_t_8_years <- fitCopula(t_cop_8_years, pseudo_obs_8_years, method = "ml", start = init_params, control = list(maxit = 100, reltol = 1e-6))
# 
#


























