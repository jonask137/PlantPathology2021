
library(readr)
library(tidyr)
library(base)

## Creating the folders

base_dir <- "Data"
dir.create(base_dir)
train_dir <- file.path(base_dir,"train")
validation_dir <- file.path(base_dir,"validation")
dir.create(train_dir)
dir.create(validation_dir)
labels <- c("scab", "healthy", "frog_eye_leaf_spot", "rust", "complex")
for (label in labels) {
  dir.create(file.path(train_dir,label))
  dir.create(file.path(validation_dir,label))
}

data <- read_csv("Data/plant-pathology-2021-fgvc8/train.csv")
data <- as.data.frame(data)
data_sep_rows <- separate_rows(data, labels, sep= " ")

##partioning
partion_size <- c(.75,.25)

##putting images into val and train folder
train_index <- sample(1:nrow(data_sep_rows), size = partion_size[1]*nrow(data_sep_rows))

train <- data_sep_rows[train_index,]
validation <- data_sep_rows[-train_index,]

for (i in labels) {
  fnames <- train$image[train$labels == i]
  file.copy(file.path("Data/plant-pathology-2021-fgvc8/train_images", fnames),
            file.path("Data/train", i))
  fnames <- validation$image[validation$labels == i]
  file.copy(file.path("Data/plant-pathology-2021-fgvc8/train_images", fnames),
            file.path("Data/validation", i)) 
}
