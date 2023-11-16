path <- setwd("D:/NeuralNet/SIR_model/R/src/")

## Create SIR model and SIR data simulation
# setwd("D:/NeuralNet/SIR_model/R/src/model/") # Step to debug
orderly2::orderly_run("model", root = path)

## Create SIR plots and convert outputs to .csv
# setwd("D:/NeuralNet/SIR_model/R/src/data/") # Step to debug
orderly2::orderly_run("data", root = path)