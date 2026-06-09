#!/usr/bin/env Rscript
# Convert a TEE long-format CSV (written by utils.tee_export) into the .rda the
# totalevalerror package bundles its example data as. Base R only -- no readr or
# dplyr -- so it runs against a bare R install. Dtype coercion mirrors the
# package's own data-raw/make_pilot_data.R.
#
# Usage: Rscript scripts/long_csv_to_rda.R <csv> <rda_out> [name]
#   name = R object name restored by load(); defaults to "tee_long".

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: long_csv_to_rda.R <csv> <rda_out> [name]")
}
csv_path <- args[[1]]
rda_path <- args[[2]]
name <- if (length(args) >= 3) args[[3]] else "tee_long"

# Empty cells (e.g. dropped/parse-failed outcomes) read as NA.
df <- read.csv(csv_path, stringsAsFactors = FALSE, na.strings = c("", "NA"))

character_cols <- c("item_id", "category", "variant_id", "sut_model", "language", "subject")
for (col in intersect(character_cols, names(df))) {
  df[[col]] <- as.character(df[[col]])
}
if ("temperature" %in% names(df)) df$temperature <- as.numeric(df$temperature)
if ("replication" %in% names(df)) df$replication <- as.integer(df$replication)
if ("outcome" %in% names(df)) df$outcome <- as.numeric(df$outcome)

assign(name, df)
save(list = name, file = rda_path)
cat(sprintf("wrote %s (%d rows) as object '%s'\n", rda_path, nrow(df), name))
