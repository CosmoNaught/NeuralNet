plot_curves <- function(results) {
	# Convert 'results' to a dataframe
	results_df <- as.data.frame(results)

	# Add a run identifier
	results_df$run_id <- seq_len(nrow(results_df))

	# Reshape the dataframe to a long format
	long_data <- pivot_longer(results_df, cols = starts_with("Day"), names_to = "Day", values_to = "Cases")

	# Convert 'Day' to a numeric variable
	long_data$Day <- as.numeric(gsub("Day_", "", long_data$Day))

	# Plot using ggplot2
	ggplot(long_data, aes(x = Day, y = Cases, group = run_id, color = as.factor(run_id))) +
	geom_line() +
	labs(title = "Simulated Epidemics", x = "Day", y = "Number of Cases", color = "Run ID") +
	theme_minimal() +
	theme(legend.position="none")
}

# Function to generate and save plots
generate_and_save_plots <- function(func_name, args, file_name) {
  plot <- do.call(func_name, args)
  ggsave(filename = file_name, plot = plot, bg = "white", width = 15, height = 9, dpi = 200)
}