
library(readr)
library(tidyr)
library(base)

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