---
title: "scivae example"
output: html_notebook
---

## Setup a python environment
If you already have one just skip this!

#### Using VENV
If you don't have Anaconda installed (or know what that is) go with the below option:
```r
virtualenv_create(
      envname = "ml",
      python = NULL,
      packages = "scivae",
      system_site_packages = getOption("reticulate.virtualenv.system_site_packages",
                                       default = FALSE)
    )
```
You only need to do the above once. This creates a new environment for you and then you can install as below:
```r
use_virtualenv("ml", required = TRUE)
```

#### Using Conda
Create a new conda environment called **ml** and install scivae into it.
```r
use_condaenv(condaEnvName, required = TRUE)
```

```r

library(tidyverse)
library(dplyr)
library(reticulate)

# If things fail here it's because you need to the steps above
use_condaenv("ml", required = TRUE) # OR use_virtualenv("ml", required = TRUE)  # depending on how you installed it!
scivae <<- import("scivae")    # Make global

df <- read_csv("iris.csv")
labels <- df$label # Keep for later

# Now we want the dataset not to have the gene ID column (i.e. just to be the numeric values)
df_mat <- df %>% select(!(label))

df_mat <- as.matrix(df_mat)
vae <- scivae$VAE(df_mat, df_mat, labels, "config.json", 'vae_rcm', config_as_str=T)
vae$encode('default')
vae$save()

# Load saved data
vae$load()
# Now let's run the VAE on the data
data <- vae$encode_new_data(df_mat, encoding_type="z", scale=T)

# Add in the columns to the old DF
df$VAE0 <- data[, 1]
df$VAE1 <- data[, 2]
df$VAE2 <- data[, 3]

vis <- scivae$Vis(vae, vae$u, NULL)
cols <- c("sepal_length", "sepal_width")

vis$plot_feature_scatters(df, 'label', columns=cols, show_plt=F, fig_type="png", save_fig=T, vae_data=data,
                                      title="cX DepthshadeTrue latent space")
vis$plot_node_hists(show_plt=F, save_fig=T)
vis$plot_node_hists(show_plt=F, save_fig=T, method="z_mean")
vis$plot_node_hists(show_plt=F, save_fig=T, method="z_log_var")

vis$plot_node_feature_correlation(df, 'label', columns=cols, show_plt=F, save_fig=T, vae_data=data)
vis$plot_node_correlation(show_plt=F, save_fig=T)
vis$plot_input_distribution(df, show_plt=F, save_fig=T)

vis$plot_top_values_by_rank(df, c("VAE0", "VAE1", "VAE2"), cols, "label", num_values=as.integer(10), cluster_rows=F)

```