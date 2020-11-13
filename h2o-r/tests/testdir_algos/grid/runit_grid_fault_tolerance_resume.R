setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

test.grid.resume <- function() {
  iris.hex <- h2o.importFile(path = locate("smalldata/iris/iris.csv"), destination_frame="iris.hex")

  ntrees_opts1 <- c(100, 200)
  ntrees_opts2 <- c(100, 200, 300, 400)
  learn_rate_opts <- c(0.01, 0.02)
  grid_size <- length(ntrees_opts2) * length(learn_rate_opts)
  print(paste("max models", grid_size))
  export_dir <- tempdir()

  hyper_parameters <- list(ntrees = ntrees_opts1, learn_rate = learn_rate_opts)
  baseline_grid <- h2o.grid(
      "gbm", grid_id="grid_ft_resume_test", 
      x=1:4, y=5, training_frame=iris.hex, 
      hyper_params=hyper_parameters,
      export_checkpoints_dir=export_dir,
      checkpoint_frames=TRUE
  )
  grid_id <- baseline_grid@grid_id
  baseline_model_count <- length(baseline_grid@model_ids)
  print(baseline_grid@model_ids)
  
  # Wipe the cloud to simulate cluster restart - the models will no longer be available
  h2o.removeAll()
  
  # Load the Grid back in with all the models checkpointed
  grid <- h2o.loadGrid(paste0(export_dir, "/", grid_id), load_frames=TRUE)
  expect_true(length(grid@model_ids) == baseline_model_count)
  
  # Start the grid search once again, should contain the original models and more
  hyper_parameters2 <- list(ntrees = ntrees_opts2, learn_rate = learn_rate_opts)
  grid <- h2o.grid(
      "gbm", grid_id="grid_ft_resume_test", 
      x=1:4, y=5, training_frame=iris.hex, 
      hyper_params=hyper_parameters2
  )
  expect_true(length(grid@model_ids) == grid_size)
  print(grid@model_ids)
}

doTest("Resume grid search after cluster restart", test.grid.resume)
