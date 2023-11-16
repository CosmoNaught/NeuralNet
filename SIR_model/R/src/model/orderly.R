dir.create("outputs", FALSE, TRUE)

# Define the number of runs and ranges for beta and gamma
n_runs <- 100000  # Set the number of runs
beta_range <- c(0.1, 0.8)  # Define min and max values for beta
gamma_range <- c(0.1, 0.8) # Define min and max values for gamma
n_days <- 100  # Number of days for simulation

betas <- runif(n_runs, beta_range[1], beta_range[2])
gammas <- runif(n_runs, gamma_range[1], gamma_range[2])

# Pre-allocate results matrix
results <- matrix(ncol = n_days + 2, nrow = n_runs) # + 2 cols for beta and gamma
colnames(results) <- c("beta", "gamma", paste("Day", 1:n_days, sep = "_"))

# Load the model once
sir <- odin.dust::odin_dust("sir.R")

# Loop (or apply parallel processing)
for (i in 1:n_runs) {
    pars <- list(beta = betas[i], gamma = gammas[i])
    mod <- sir$new(pars, 0, 1, seed = 1L)
    y <- mod$simulate(seq(0, 4 * n_days, by = 4))
    rownames(y) <- names(mod$info()$index)
    cases_data <- y["cases_inc", , seq_len(n_days) + 1]
    results[i, ] <- c(betas[i], gammas[i], cases_data)
}

# Write results to CSV
write.csv(results, "outputs/epidemic_curves.csv", row.names = FALSE)
