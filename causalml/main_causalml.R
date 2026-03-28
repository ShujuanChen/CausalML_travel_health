`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || identical(x, "")) y else x
}

frame_files <- Filter(
  Negate(is.null),
  lapply(sys.frames(), function(frame) frame$ofile %||% NULL)
)
script_dir <- if (length(frame_files) > 0) {
  dirname(normalizePath(frame_files[[length(frame_files)]], winslash = "/", mustWork = FALSE))
} else {
  normalizePath(file.path(getwd(), "causalml"), winslash = "/", mustWork = FALSE)
}

require_env_var <- function(var_name) {
  value <- Sys.getenv(var_name, unset = "")
  if (identical(value, "")) {
    stop(
      sprintf(
        "Missing required environment variable: %s",
        var_name
      ),
      call. = FALSE
    )
  }
  value
}

parse_csv_env <- function(var_name, required = TRUE) {
  value <- Sys.getenv(var_name, unset = "")
  if (identical(value, "")) {
    if (required) {
      stop(
        sprintf(
          "Missing required environment variable: %s",
          var_name
        ),
        call. = FALSE
      )
    }
    return(character())
  }
  trimws(unlist(strsplit(value, ",")))
}

source(file.path(script_dir, "col.R"))
source(file.path(script_dir, "tmle_3.R"))

data_file <- require_env_var("CAUSALML_DATA_FILE")
output_dir <- Sys.getenv(
  "CAUSALML_OUTPUT_DIR",
  unset = file.path(script_dir, "outputs")
)
results_file <- file.path(output_dir, "result.csv")

exposure_builder <- require_env_var("CAUSALML_EXPOSURE_BUILDER")
outcome_builder <- require_env_var("CAUSALML_OUTCOME_BUILDER")
base_covariate_builders <- parse_csv_env("CAUSALML_BASE_COVARIATES")
extended_covariate_builders <- parse_csv_env(
  "CAUSALML_EXTENDED_COVARIATES",
  required = FALSE
)
columns_to_drop <- parse_csv_env("CAUSALML_DROP_COLUMNS", required = FALSE)

reference_exposure_label <- require_env_var("CAUSALML_REFERENCE_LABEL")
comparison_one_label <- require_env_var("CAUSALML_COMPARISON_A_LABEL")
comparison_two_label <- require_env_var("CAUSALML_COMPARISON_B_LABEL")

data <- read.csv(data_file)

for (redundant_col in c("X", "X.1", "X.2")) {
  if (redundant_col %in% names(data)) {
    data[[redundant_col]] <- NULL
  }
}

build_analysis_spec <- function(subset_name, treatment_labels, column_builders) {
  list(
    subset_name = subset_name,
    strata_name = "all",
    treatment_labels = treatment_labels,
    column_builders = column_builders
  )
}

analysis_specs <- list(
  build_analysis_spec(
    subset_name = "comparison_a",
    treatment_labels = c(reference_exposure_label, comparison_one_label),
    column_builders = c(
      exposure_builder,
      outcome_builder,
      base_covariate_builders
    )
  ),
  build_analysis_spec(
    subset_name = "comparison_b",
    treatment_labels = c(reference_exposure_label, comparison_two_label),
    column_builders = c(
      exposure_builder,
      outcome_builder,
      base_covariate_builders,
      extended_covariate_builders
    )
  )
)

sanitize_names <- function(df) {
  for (i in seq_along(colnames(df))) {
    colnames(df)[i] <- gsub(" ", ".", colnames(df)[i], fixed = TRUE)
    colnames(df)[i] <- gsub(",", ".", colnames(df)[i], fixed = TRUE)
    colnames(df)[i] <- gsub("-", ".", colnames(df)[i], fixed = TRUE)
    colnames(df)[i] <- gsub("/", ".", colnames(df)[i], fixed = TRUE)
  }
  df
}

run_analysis <- function(spec, input_data) {
  output <- input_data[1]

  for (builder_name in spec$column_builders) {
    output <- cbind(output, get(builder_name)(input_data))
  }

  output <- sanitize_names(output)
  output <- output[, 2:ncol(output), drop = FALSE]
  output <- na.omit(output)
  output <- subset(output, output[[exposure_builder]] %in% spec$treatment_labels)

  for (drop_name in columns_to_drop) {
    if (drop_name %in% names(output)) {
      output[[drop_name]] <- NULL
    }
  }

  colnames(output)[colnames(output) == exposure_builder] <- "x"
  colnames(output)[colnames(output) == outcome_builder] <- "y"

  tmle_3(
    output = output,
    model_info = list(
      subset_name = spec$subset_name,
      strata_name = spec$strata_name,
      reference_exposure_label = reference_exposure_label,
      output_dir = file.path(output_dir, "models")
    )
  )
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

for (spec in analysis_specs) {
  result <- run_analysis(spec, data)

  if (!file.exists(results_file)) {
    write.csv(result, results_file, row.names = FALSE)
  } else {
    write.table(
      result,
      results_file,
      append = TRUE,
      sep = ",",
      col.names = FALSE,
      row.names = FALSE,
      quote = TRUE
    )
  }
}
