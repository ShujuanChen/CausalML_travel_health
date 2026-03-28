tmle_3 <- function(output, model_info) {
  `%||%` <- function(x, y) {
    if (is.null(x) || length(x) == 0 || identical(x, "")) y else x
  }

  library(tmle3)
  library(sl3)
  library(tidyverse)
  library(dplyr)

  lrnr_glm <- make_learner(Lrnr_glm)
  lrnr_glmnet <- make_learner(Lrnr_glmnet)
  lrnr_ranger <- make_learner(Lrnr_ranger)
  lrnr_xgboost <- make_learner(Lrnr_xgboost)
  lrnr_earth <- make_learner(Lrnr_earth)

  sl_ <- make_learner(
    Stack,
    unlist(
      list(lrnr_glm, lrnr_glmnet, lrnr_ranger, lrnr_xgboost, lrnr_earth),
      recursive = TRUE
    )
  )

  Q_learner <- Lrnr_sl$new(
    learners = sl_,
    metalearner = Lrnr_nnls$new(convex = TRUE)
  )
  g_learner <- Lrnr_sl$new(
    learners = sl_,
    metalearner = Lrnr_nnls$new(convex = TRUE)
  )
  learner_list <- list(Y = Q_learner, A = g_learner)

  W <- output[, !(names(output) %in% c("x", "y")), drop = FALSE]
  covariates_mod <- colnames(W)
  output$x <- ifelse(output$x == model_info[["reference_exposure_label"]], 0, 1)

  rr_spec <- tmle_RR(baseline_level = 0, contrast_level = 1)
  nodes_ <- list(W = covariates_mod, A = "x", Y = "y")

  set.seed(10)
  tmle_fit_rr <- tmle3(rr_spec, output, nodes_, learner_list)

  model_dir <- model_info[["output_dir"]] %||% "model"
  if (!dir.exists(model_dir)) {
    dir.create(model_dir, recursive = TRUE)
  }

  model_name <- file.path(
    model_dir,
    paste0(
      "tmle3_",
      model_info[["subset_name"]],
      "_",
      model_info[["strata_name"]],
      ".rds"
    )
  )
  saveRDS(tmle_fit_rr, file = model_name)

  rr_estimate <- tmle_fit_rr$summary$psi_transformed[3]
  rr_se <- tmle_fit_rr$summary$se[3] * rr_estimate
  rr_ci_lower <- tmle_fit_rr$summary$lower_transformed[3]
  rr_ci_upper <- tmle_fit_rr$summary$upper_transformed[3]

  z_score <- tmle_fit_rr$summary$tmle_est[3] / tmle_fit_rr$summary$se[3]
  rr_p <- 2 * (1 - pnorm(abs(z_score)))

  data.frame(
    Method = "TMLE",
    Subset = model_info[["subset_name"]],
    Strata = model_info[["strata_name"]],
    RR_Estimate = rr_estimate,
    RR_SE = rr_se,
    RR_CI_1 = rr_ci_lower,
    RR_CI_2 = rr_ci_upper,
    RR_P = rr_p
  )
}
