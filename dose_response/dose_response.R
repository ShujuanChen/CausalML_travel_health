`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || identical(x, "")) y else x
}

require_env_var <- function(var_name) {
  value <- Sys.getenv(var_name, unset = "")
  if (identical(value, "")) {
    stop(sprintf("Missing required environment variable: %s", var_name), call. = FALSE)
  }
  value
}

frame_files <- Filter(
  Negate(is.null),
  lapply(sys.frames(), function(frame) frame$ofile %||% NULL)
)
script_dir <- if (length(frame_files) > 0) {
  dirname(normalizePath(frame_files[[length(frame_files)]], winslash = "/", mustWork = FALSE))
} else {
  normalizePath(file.path(getwd(), "dose_response"), winslash = "/", mustWork = FALSE)
}

library(ggplot2)
library(dplyr)
library(ggridges)
library(forcats)
library(viridis)
library(hrbrthemes)
library(splines)

input_file <- require_env_var("DOSE_RESPONSE_INPUT_FILE")
index_column <- require_env_var("DOSE_RESPONSE_INDEX_COLUMN")
effect_column <- require_env_var("DOSE_RESPONSE_EFFECT_COLUMN")
reference_effect <- as.numeric(require_env_var("DOSE_RESPONSE_REFERENCE_EFFECT"))

output_dir <- Sys.getenv(
  "DOSE_RESPONSE_OUTPUT_DIR",
  unset = file.path(script_dir, "outputs")
)
plot_dir_a <- file.path(output_dir, "plot_a")
plot_dir_b <- file.path(output_dir, "plot_b")

analysis_title <- Sys.getenv("DOSE_RESPONSE_TITLE", unset = "Outcome A")
x_label <- Sys.getenv("DOSE_RESPONSE_X_LABEL", unset = "Relative effect")
y_label_grouped <- Sys.getenv("DOSE_RESPONSE_GROUP_LABEL", unset = "Index band")
y_label_curve <- Sys.getenv("DOSE_RESPONSE_CURVE_LABEL", unset = "Relative effect")
index_label <- Sys.getenv("DOSE_RESPONSE_INDEX_LABEL", unset = "Index score")

if (!dir.exists(plot_dir_a)) {
  dir.create(plot_dir_a, recursive = TRUE)
}
if (!dir.exists(plot_dir_b)) {
  dir.create(plot_dir_b, recursive = TRUE)
}

data <- read.csv(input_file)

base_theme <- theme(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.border = element_blank(),
  plot.background = element_rect(color = "white", fill = "white"),
  legend.text = element_text(face = "bold", size = 13),
  plot.title = element_text(face = "bold", hjust = 0),
  axis.title.x = element_text(face = "bold", size = 13),
  axis.title.y = element_text(face = "bold", size = 13),
  axis.line.x = element_line(color = "black"),
  axis.line.y = element_line(color = "black"),
  axis.text.y = element_text(face = "bold", size = 13),
  axis.text.x = element_text(face = "bold", size = 13),
  axis.ticks = element_line(color = "black"),
  legend.position = "bottom",
  legend.title = element_blank()
)

breaks_seq <- seq(5, 50, by = 5)
bin_labels <- paste0(head(breaks_seq, -1), "–", tail(breaks_seq, -1))

data <- data %>%
  mutate(
    index_band = cut(
      .data[[index_column]],
      breaks = breaks_seq,
      right = TRUE,
      include.lowest = TRUE,
      labels = bin_labels
    )
  )

ridge_plot <- ggplot(
  data %>% filter(!is.na(index_band)),
  aes(x = .data[[effect_column]], y = index_band, fill = stat(quantile))
) +
  stat_density_ridges(
    quantile_lines = FALSE,
    calc_ecdf = TRUE,
    geom = "density_ridges_gradient",
    scale = 1.8,
    rel_min_height = 0.01
  ) +
  scale_fill_brewer(
    palette = "PuBu",
    labels = c("0-25%", "25-50%", "50-75%", "75-100%")
  ) +
  labs(
    x = x_label,
    y = y_label_grouped,
    title = analysis_title,
    fill = "Quantile"
  ) +
  theme_minimal() +
  base_theme +
  geom_vline(
    xintercept = reference_effect,
    color = "darkred",
    linetype = "dashed",
    size = 0.7
  ) +
  coord_cartesian(xlim = c(0.88, 1.08)) +
  expand_limits(y = c(NA, max(as.numeric(as.character(data$index_band))) + 2)) +
  theme(
    axis.line.x = element_line(linewidth = 0.6, colour = "black"),
    axis.line.y = element_line(linewidth = 0.6, colour = "black"),
    axis.ticks = element_line(linewidth = 0.6, colour = "black"),
    axis.ticks.length = unit(0.3, "cm"),
    axis.text = element_text(size = 12, colour = "black"),
    axis.title = element_text(size = 13, colour = "black")
  )

ggsave(
  filename = file.path(plot_dir_a, "relative_effect_by_index_band_ridge.png"),
  plot = ridge_plot,
  width = 6.5,
  height = 10,
  dpi = 400
)

summary_by_band <- data %>%
  filter(!is.na(index_band), !is.na(.data[[effect_column]])) %>%
  group_by(index_band) %>%
  summarise(
    n = n(),
    effect_min = min(.data[[effect_column]]),
    effect_max = max(.data[[effect_column]]),
    effect_mean = mean(.data[[effect_column]]),
    effect_median = median(.data[[effect_column]]),
    effect_sd = sd(.data[[effect_column]]),
    effect_se = effect_sd / sqrt(n),
    t_crit = qt(0.975, df = n - 1),
    mean_ci_low = effect_mean - t_crit * effect_se,
    mean_ci_high = effect_mean + t_crit * effect_se,
    pct_below_reference = 100 * mean(.data[[effect_column]] < reference_effect),
    .groups = "drop"
  ) %>%
  arrange(index_band) %>%
  select(-t_crit)

write.csv(
  summary_by_band,
  file = file.path(output_dir, "relative_effect_summary.csv"),
  row.names = FALSE
)

curve_data <- data %>%
  transmute(
    effect_value = .data[[effect_column]],
    index_value = .data[[index_column]]
  ) %>%
  filter(is.finite(effect_value), effect_value > 0, is.finite(index_value))

knots_vec <- quantile(curve_data$index_value, c(.05, .275, .5, .725, .95), na.rm = TRUE)
fit <- lm(
  log(effect_value) ~ ns(index_value, knots = knots_vec[2:4], Boundary.knots = knots_vec[c(1, 5)]),
  data = curve_data
)

anova(fit)

grid <- data.frame(
  index_value = seq(
    min(curve_data$index_value, na.rm = TRUE),
    max(curve_data$index_value, na.rm = TRUE),
    length.out = 600
  )
)

effect_hat <- function(x) {
  exp(predict(fit, newdata = data.frame(index_value = x)))
}

effect_grid <- effect_hat(grid$index_value)
reference_index <- grid$index_value[which.min(abs(effect_grid - reference_effect))]

beta <- coef(fit)
variance_matrix <- vcov(fit)

grid_matrix <- model.matrix(
  ~ ns(index_value, knots = knots_vec[2:4], Boundary.knots = knots_vec[c(1, 5)]),
  data = grid
)
reference_matrix <- model.matrix(
  ~ ns(index_value, knots = knots_vec[2:4], Boundary.knots = knots_vec[c(1, 5)]),
  data = data.frame(index_value = reference_index)
)

eta_grid <- drop(grid_matrix %*% beta)
eta_reference <- drop(reference_matrix %*% beta)
delta <- eta_grid - eta_reference

contrast_matrix <- grid_matrix - matrix(
  drop(reference_matrix),
  nrow = nrow(grid_matrix),
  ncol = ncol(grid_matrix),
  byrow = TRUE
)
contrast_variance <- contrast_matrix %*% variance_matrix
delta_variance <- rowSums(contrast_variance * contrast_matrix)
delta_se <- sqrt(pmax(delta_variance, 0))

prediction_grid <- tibble::tibble(
  index_value = grid$index_value,
  effect_fit = reference_effect * exp(delta),
  effect_low = reference_effect * exp(delta - 1.96 * delta_se),
  effect_high = reference_effect * exp(delta + 1.96 * delta_se)
)

curve_plot <- ggplot(prediction_grid, aes(index_value, effect_fit)) +
  geom_ribbon(
    aes(ymin = effect_low, ymax = effect_high),
    fill = "dodgerblue3",
    alpha = 0.38,
    color = NA
  ) +
  geom_line(color = "dodgerblue3", linewidth = 0.6) +
  geom_hline(yintercept = reference_effect, linetype = 2, color = "grey30") +
  geom_vline(
    xintercept = reference_index,
    linetype = "dashed",
    color = "grey70",
    linewidth = 0.5
  ) +
  geom_point(aes(x = reference_index, y = reference_effect), color = "red", size = 2) +
  labs(
    x = index_label,
    y = y_label_curve,
    title = analysis_title
  ) +
  theme_minimal() +
  geom_hline(yintercept = reference_effect, linetype = 2, color = "grey30") +
  base_theme

ggsave(
  filename = file.path(plot_dir_a, "relative_effect_spline_curve.png"),
  plot = curve_plot,
  width = 8,
  height = 6,
  dpi = 400
)