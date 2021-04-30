# Libraries ----
library(keras)
library(readr)
library(tidyr)
library(base)
library(doParallel)

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
batch_size <- 32
target_size <- c(150,150)

## Making data generators -----
train_generator <- flow_images_from_directory(train_dir, generator = image_data_generator(),
                                              target_size = target_size, color_mode = "rgb",
                                              class_mode = "categorical", batch_size = batch_size, shuffle = TRUE,
                                              seed = 123)
validation_generator <- flow_images_from_directory(validation_dir, generator = image_data_generator(),
                                                   target_size = target_size, color_mode = "rgb", classes = NULL,
                                                   class_mode = "categorical", batch_size = batch_size, shuffle = TRUE,
                                                   seed = 123)
train_samples = nrow(train)
validation_samples = nrow(validation)

## Constructing the model ----
#base_model <- application_inception_v3(weights = 'imagenet', include_top = FALSE)
base_model <- application_vgg16(weights = 'imagenet', include_top = FALSE)

## add your custom layers
predictions <- base_model$output %>% 
  layer_global_average_pooling_2d(trainable = T) %>% 
  layer_dense(64, trainable = T) %>%
  layer_activation("relu", trainable = T) %>%
  layer_dropout(0.4, trainable = T) %>%
  layer_dense(5, trainable = T) %>%
  layer_activation("softmax", trainable = T)

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)

### Freeze the layer weights in VGG16
for (layer in base_model$layers)
  layer$trainable <- FALSE

## Set call backs ----

## Making Call back list ----

callbacks_list <- list(
  #Interrupt when there is no more improvement
  callback_early_stopping(monitor = "acc", #We monitor accuracy
                          patience = 3 #Stops when acc does not improve over more than 1 epoch
  ),
  callback_reduce_lr_on_plateau(monitor = "val_loss",
                                factor = 0.1,
                                patience = 4
  )
)

################### Section 5 #########################
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.003, decay = 1e-6),  ## play with the learning rate
  metrics = "accuracy"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size)/2, 
  epochs = 20, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size)/2,
  workers = detectCores()-1,
  callbacks = callbacks_list
)

save_model_weights_hdf5(model, "Saved Objects/vgg19_model.h5")
save_model_weights_hdf5(history, "Saved Objects/vgg19_history.h5")
saveRDS(history,"Saved objects/vgg19_history.rds")