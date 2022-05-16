#!/usr/bin/env Rscript
library(reticulate)
library(tensorflow)
library(chk)
library(keras)
library(here)
library(lubridate)
library(stringr)
library(fs)
library(rstudioapi)
library(magick)
library(tibble)
library(dplyr)
library(purrr)
library(sigmoid)
library(tidyr)

current_time <- Sys.time();

reticulate::use_condaenv(condaenv = 'pyrstudio')
#pip install pillow
#Keras::install_keras(method='conda', conda='pyrstudio'

tf_gpus <- tf$config$list_physical_devices('GPU');

chk_not_empty(tf_gpus)

# Does not seem to change anything, tried to remove 'I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory'
# for (gpu in tf_gpus) {
#   tf$config$experimental$set_memory_growth(gpu, TRUE)
# }


latent_dim <- 30
height <- 28
width <- 28
channels <- 1

iterations <- 500000
batch_size <- 400
heartbeat_checkpoint_iteration_modulus <- 100;
images_checkpoint_iteration_modulus <- 100;
model_checkpoint_iteration_modulus <- 2000;



base_scratchpad_folder <- here::here('data', 'scratchpad');
current_time_str <-
  format_ISO8601(current_time, precision='ymdhm') |>
  str_remove_all(pattern='[:-]')
scratchpad_folder <- file.path(base_scratchpad_folder, current_time_str)
fs::dir_create(scratchpad_folder)
images_folder <- file.path(scratchpad_folder, 'images');
fs::dir_create(images_folder)
model_folder <- file.path(scratchpad_folder, 'model');
fs::dir_create(model_folder)




# The Generator
generator_input <- layer_input(shape = c(latent_dim), name='generator_latent_input')
generator_output <- generator_input |>
  layer_dense(units = 7 * 7 * 32, use_bias = FALSE, activation = 'gelu') |>
  layer_reshape(target_shape = c(7, 7, 32)) |>
  layer_conv_2d(filters = 512, kernel_size = 3, use_bias = T,
                padding = "same", activation = 'gelu') |>
  layer_conv_2d_transpose(filters = 256, kernel_size = 3, use_bias = T,
                          strides = 2, padding = "same", activation = 'gelu') |>
  layer_conv_2d_transpose(filters = 256, kernel_size = 3, use_bias = T,
                          strides = 2, padding = "same", activation = 'gelu') |>
  layer_conv_2d(filters = 128, kernel_size = 5, use_bias = T,
                padding = "same", activation = 'gelu') |>
  #  layer_layer_normalization() |>
  layer_conv_2d(filters = channels, kernel_size = 7,
                activation = "tanh", padding = "same") |>
  layer_reshape(target_shape = c(height, width, channels))

generator <- keras_model(generator_input, generator_output, name="generator")
summary(generator)

#The Discriminator
discriminator_input <- layer_input(shape = c(height, width, channels), name='candidate_image')
discriminator_output <-
  discriminator_input |>
  layer_spatial_dropout_2d(rate=0.2, batch_size=batch_size) |>
  layer_conv_2d(filters = 256, kernel_size = 3, padding = "same", activation = 'gelu') |>
  layer_conv_2d(filters = 256, kernel_size = 3, strides = 2, padding = "same", activation = 'gelu') |>
  layer_spatial_dropout_2d(rate=0.2, batch_size=batch_size) |>
  layer_conv_2d(filters = 128, kernel_size = 3, strides = 2, padding = "same", activation = 'gelu') |>
  layer_conv_2d(filters = 4, kernel_size = 3, strides = 1, padding = "same", activation = 'gelu') |>
  layer_flatten() |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 1, activation = "sigmoid")

discriminator <- keras_model(discriminator_input, discriminator_output, name="discriminator")
summary(discriminator)

discriminator_learning_rate <- 0.0008;
discriminator_optimizer <- optimizer_rmsprop(
  learning_rate = discriminator_learning_rate * 1,
#  clipnorm = 1.0,
  decay = discriminator_learning_rate / iterations
)


 compile(discriminator,
  optimizer = discriminator_optimizer,
  jit_compile = TRUE,
  loss = "binary_crossentropy"
)

# The adversarial network
freeze_weights(discriminator) # only relevant for the gan

gan_input <- layer_input(shape = c(latent_dim), name='gan_latent_input')
gan_output <-
  gan_input |>
  generator() |>
  discriminator()
gan <- keras_model(gan_input, gan_output, name="gan")

gan_learning_rate <- 0.0008;
gan_optimizer <- optimizer_rmsprop(
  learning_rate = gan_learning_rate,
  clipnorm = 1.0,
  clipvalue = 1.0,
  decay = gan_learning_rate / iterations
)

compile(gan,
  optimizer = gan_optimizer,
  jit_compile = TRUE,
  loss = "binary_crossentropy"
)


cat(model_to_json(generator), file=file.path(scratchpad_folder, 'generator.json'), sep='\n')
cat(format(generator), file=file.path(scratchpad_folder, 'generator.txt'), sep='\n')
cat(model_to_json(discriminator), file=file.path(scratchpad_folder, 'discriminator.json'), sep='\n')
cat(format(discriminator), file=file.path(scratchpad_folder, 'discriminator.txt'), sep='\n')
cat(model_to_json(gan), file=file.path(scratchpad_folder, 'gan.json'), sep='\n')
cat(format(gan), file=file.path(scratchpad_folder, 'gan.txt'), sep='\n')

# Training
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- 2 * (x_train / 255) - 1
x_test <- 2 * (x_test / 255) - 1

x_train_length <- dim(x_train)[1]


start <- 1

for (step in 1:iterations) {
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim),
                                  nrow = batch_size, ncol = latent_dim)
  generated_images <-  predict_on_batch(generator, random_latent_vectors)
  dim(generated_images) <- c(batch_size, height, width, channels)

  stop <- start + batch_size - 1
  real_images <- x_train[start:stop,,]
  dim(real_images) <- c(batch_size, height, width, channels)
  rows <- nrow(real_images)


  combined_images <- array(0, dim = c(rows * 2, dim(real_images)[-1]))
  combined_images[1:rows,,,] <- generated_images
  combined_images[(rows+1):(rows*2),,,] <- real_images

  orig_labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),
                  matrix(0, nrow = batch_size, ncol = 1))

  labels <- pmin(1,
                 pmax(0,
                      orig_labels +
                        if_else(orig_labels==1, -0.2, 0.2) *
                        array(runif(prod(dim(orig_labels))),
                              dim = dim(orig_labels))
                 ));

  d_loss <- train_on_batch(
    discriminator,
    combined_images,
    labels)

  random_latent_vectors <- matrix(rnorm(2*batch_size * latent_dim),
                                  nrow = 2*batch_size, ncol = latent_dim)

#  misleading_targets <- array(0.1 * runif(batch_size), dim = c(batch_size, 1))
  misleading_targets <- array(0, dim = c(2*batch_size, 1))

  a_loss <- train_on_batch(
    gan,
    random_latent_vectors,
    misleading_targets
  )

  local({
    if (step < 20) {
      var_gen <- generated_images
      dim(var_gen) <- c(batch_size, height*width*channels)
      var_real <- real_images
      dim(var_real) <- c(batch_size, height*width*channels)
      cat(sprintf('sd_g: %0.3e\tsd_r %0.3e\n', mean(var(var_gen)), mean(var(var_real))))
    }
  })

  start <- start + batch_size
  if (start > (nrow(x_train) - batch_size))
    start <- 1

  if (step %% heartbeat_checkpoint_iteration_modulus == 0 ||
      step %% images_checkpoint_iteration_modulus == 0 ||
      step %% model_checkpoint_iteration_modulus == 0 ) {
    cat("\nstep: ", step, "\n");
    cat("discriminator loss:", sprintf("  % 10.4f", d_loss), "\n")
    cat("adversarial loss:", sprintf("  % 10.4f", a_loss), "\n")


    local({
      var_gen <- generated_images
      dim(var_gen) <- c(batch_size, height*width*channels)
      var_real <- real_images
      dim(var_real) <- c(batch_size, height*width*channels)
      cat(sprintf('sd_g: %0.3e\tsd_r %0.3e\n', mean(var(var_gen)), mean(var(var_real))))
    })
  }

  if (step %% images_checkpoint_iteration_modulus == 0) {
    cat("image checkpoint: ", ceiling(step / images_checkpoint_iteration_modulus), "\n");

    #TODO: What is the difference to predict_on_batch?
    b_predict <- predict(discriminator, combined_images)

    image_tibble <-
      tibble(label=orig_labels[,1],
             prob=b_predict[,1],
             image=asplit(combined_images, 1)) |>
      mutate(prob_weight=dbeta(pmin(0.95, pmax(0.05, prob)), 0.4, 0.4)) |> #plot((1:100)/100, dbeta(pmin(0.95, pmax(0.05,(1:100)/100)), 0.4, 0.4))
      group_by(label) |>
      sample_n(size=32, weight=prob_weight) |> 
      ungroup() |>
      mutate(image=map(image,
                        function(im_data) {
                            raw_im <- pmin(1, pmax(0, (im_data + 1.0) / 2));
                            dim(raw_im) <- c(dim(im_data)[1:2], 1);
                            im <- image_read(raw_im) |> image_convert(colorspace='rgb');
                        })) |>
      mutate(image=modify2(image, label,
                       function(orig_im, lab) {
                          im <- orig_im |>
                            image_border(if_else(lab == 1, 'white', 'green'), "12x12")
                       })) |>
      mutate(image=map2(image, prob,
                        function(orig_im, prob) {
                          im <-
                            orig_im |>
                            image_annotate(text=sprintf("%.4f",prob))
                        })) |>
      arrange(prob, desc(label))

    example_image <-
      image_tibble$image |>
      image_join() |>
      image_montage(geometry = '+5+5', tile="8x8") |>
      image_border('white', "12x12") |>
      image_annotate(text=sprintf("step: %i, d_loss:%.4e, a_loss:%.4e",step, d_loss, a_loss))

    image_write(example_image, format='png', file.path(images_folder, sprintf("%07i_e.png", ceiling(step / images_checkpoint_iteration_modulus))))
    if (rstudioapi::isAvailable()) {
      print(example_image)
    }

    gen_image <- (generated_images[1,,,] +1) / 2 * 255
    dim(gen_image) <- c(dim(gen_image)[1:2], 1);
    image_array_save(
      gen_image,
      path = file.path(images_folder, sprintf("%07i_g.png", ceiling(step / images_checkpoint_iteration_modulus))))

    real_image <- (real_images[1,,,]+1) / 2 * 255;
    dim(real_image) <- c(dim(real_image)[1:2], 1);
    image_array_save(
      real_image,
      path = file.path(images_folder, sprintf("%07i_r.png", ceiling(step / images_checkpoint_iteration_modulus))))

  }
  if (step %% model_checkpoint_iteration_modulus == 0) {
    cat("model weight checkpoint: ", ceiling(step / model_checkpoint_iteration_modulus), "\n");
    save_model_weights_hdf5(gan, file.path(model_folder, sprintf("%07i.h5", step)))
  }
}







