# # Line below has to be run only once
# install.packages("copula")
# install.packages("plotly")
install.packages("MASS")

library(copula)
library(plotly)
library(MASS)

return_rates = read.csv("data/positive_quarterly_return_rates.csv")

stocks = colnames(return_rates)[2:ncol(return_rates)]

ecdf_list = lapply(return_rates[stocks], ecdf)


## example plot
plot(ecdf(return_rates$AAPL), main="ECDF of AAPL", xlab="Return", ylab="ECDF")


pseudo_obs = apply(return_rates[stocks], 2, function(x) rank(x)/(length(x)+1))


dim = ncol(pseudo_obs) 
clayton_cop = claytonCopula(dim=dim)
fit = fitCopula(clayton_cop, pseudo_obs, method="ml")   # maximum likelihood estimation (ml)


# summary(fit)

theta = coef(fit)
# print(theta)


simulated_data <- rCopula(1000, claytonCopula(theta, dim))

# first two stocks -> countour plot
plot(simulated_data[,1], simulated_data[,2], main="Simulated Clayton Copula Data",
     xlab="Stock 1", ylab="Stock 2")


fig = plot_ly(x = simulated_data[,1], y = simulated_data[,2], z = kde2d(simulated_data[,1], simulated_data[,2])$z,
              type = "scatter3d", mode = "markers",
              marker = list(size = 3, color = kde2d(simulated_data[,1], simulated_data[,2])$z, colorscale = 'Viridis'))


fig = fig %>% layout(scene = list(xaxis = list(title = "Stock 1 Returns"),
                                   yaxis = list(title = "Stock 2 Returns"),
                                   zaxis = list(title = "Density")),
                      title = "3D Density Plot of Stock Returns (Clayton Copula)")

fig
