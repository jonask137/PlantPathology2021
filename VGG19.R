
# Libraries ----
library(keras)
library(readr)
library(tidyr)
library(base)

# Defining the model ----

#Creating train and validation dir

base_dir <- "Data"

train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")

data <- read_csv("Data/plant-pathology-2021-fgvc8/train.csv")
data <- as.data.frame(data)
data_sep_rows <- separate_rows(data, labels, sep= " ")
partion_size <- c(.75,.25)
train_index <- sample(1:nrow(data_sep_rows), size = partion_size[1]*nrow(data_sep_rows))

train <- data_sep_rows[train_index,]
validation <- data_sep_rows[-train_index,]


## Setting tuning params ----

batch_size <- 16
target_size <- c(300,200)


## Constructing the model ----

conv_base <- application_vgg19(
  weights = "imagenet",
  include_top = FALSE, #Do we want the densely connected layers?
  input_shape = c(300, 200, 3)
)

conv_base

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", 
                input_shape = c(300, 200, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") #We have 5 different categories

model %>% compile(
  loss = "categorical_crossentropy", #we have binary classification
  optimizer = optimizer_rmsprop(lr = 1e-4), #we know its tunable, but we start at some place"
  metrics = c("acc") #we have balanced data, so acc should be ok!
)



## Data generators ----

# All images will be rescaled by 1/255
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)


train_generator <- flow_images_from_directory(
  # This is the target directory
  train_dir,
  # This is the data generator
  train_datagen,
  # All images will be resized to 150x150
  target_size = target_size,
  batch_size = batch_size,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "categorical"
)


validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = target_size,
  batch_size = batch_size,
  class_mode = "categorical"
)

## Making Call back list ----

callbacks_list <- list(
  #Interrupt when there is no more improvement
  callback_early_stopping(monitor = "acc", #We monitor accuracy
                          patience = 3 #Stops when acc does not improve over more than 1 epoch
  ),
  callback_reduce_lr_on_plateau(monitor = "val_loss",
                                factor = 0.1,
                                patience = 10
  )
)



# Compiling the model ----

model %>% compile(
  loss = "categorical_crossentropy", #we have binary classification
  optimizer = optimizer_rmsprop(lr = 1e-4), #we know its tunable, but we start at some place"
  metrics = c("acc"), #we have balanced data, so acc should be ok!
)


# Running the model ----

train_steps_per_epoch <- round(nrow(train)/batch_size)
validation_steps_per_epoch <- round(nrow(validation)/batch_size)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = train_steps_per_epoch,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = validation_steps_per_epoch
)
