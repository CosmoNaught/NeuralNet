# setwd("D:/NeuralNet/SIR_model/R/src/data/")

# Load necessary libraries
library(ggplot2)
library(tidyr)

source("support.R")

orderly2::orderly_dependency("model", "latest()",
c(results.csv = "outputs/epidemic_curves.csv"))

results <- read.csv("results.csv")

# Create directory for figures
dir.create("figs", FALSE, TRUE)

# List of plotting functions and their arguments
plot_functions <- list(
  plot_curves = list(results)
)

# Generate and save each plot
lapply(names(plot_functions), function(func_name) {
  file_name <- paste0("figs/", func_name, ".png")
  generate_and_save_plots(func_name, plot_functions[[func_name]], file_name)
})