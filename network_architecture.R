
#Hyperparameter setting
ini = initializer_ones()
u = 8


library(keras)
options(keras.view_metrics = FALSE)

k_clear_session()


rates <- layer_input(shape = c(wind,N[2]), dtype = 'float32', name = 'rates')
features_median = rates %>%layer_lstm(units = u, name = "feature_median", kernel_initializer = ini, 
                                      recurrent_initializer = ini, bias_initializer =  ini)
features_quantile = rates %>%layer_lstm(units = u,name = "feature_quantile", kernel_initializer = ini, 
                                        recurrent_initializer = ini, bias_initializer =  ini)

var_median =  features_median %>%layer_dropout(0.2)%>% layer_dense(units = N[2], activation = "tanh",  name = "var_median") %>%
  layer_dropout(0.2)%>%layer_reshape(c(1,N[2]))

pre_quantile =  features_quantile %>% layer_dropout(0.2)%>% layer_dense(units = N[2], activation ='relu') %>%
  layer_dropout(0.2)%>%layer_reshape(c(1,N[2]), name = "pre_quantile")

var_quantile = var_median %>% list(pre_quantile) %>% layer_subtract()%>%layer_reshape(c(1,N[2]))

d = layer_dense(units = 1, activation ='linear', kernel_constraint = constraint_nonneg(), bias_constraint = constraint_nonneg())

covar_median = var_median %>%
  layer_reshape(c(N[2],1)) %>%time_distributed(d) %>% layer_reshape(c(1,N[2]), name = "covar_median")  


model <- keras_model(
  inputs = list(rates), 
  outputs = c(var_median, var_quantile, covar_median))

# Optimisation setting
model %>% compile(
  loss = list(tilted_loss_median, tilted_loss_lower, tilted_loss_lower),
  optimizer = optimizer_nadam(),
  metrics = c('mse'))


lr_callback = callback_reduce_lr_on_plateau(monitor = "loss",factor=.90, patience = 50, verbose=1, cooldown = 5, min_lr = 0.00005)
model_callback = callback_model_checkpoint(monitor = "loss",filepath = "rcode/systemic_risk/cp.ckpt", verbose = 1,save_best_only = TRUE, save_weights_only = T)

fit <- model %>% keras::fit(x = list(x_train), 
                            y= list(y_train, y_train, y_train_sys),
                            epochs= 1000,
                            batch_size=16, 
                            verbose=2, 
                            callbacks = list(lr_callback,model_callback),
                            shuffle=T)


# 

fitted_model = model %>% load_model_weights_tf(filepath =  "rcode/systemic_risk/cp.ckpt") 
all_w = get_weights(fitted_model)


#Load optimised weights in the new NN model
k_clear_session()
rates <- layer_input(shape = c(wind,N[2]), dtype = 'float32', name = 'rates')
features_median = rates %>%layer_lstm(units = u,weights = list(all_w[[1]], all_w[[2]], all_w[[3]]), name = "feature_median")
features_quantile = rates %>%layer_lstm(units = u,weights = list(all_w[[4]], all_w[[5]], all_w[[6]]),name = "feature_quantile")

var_median =  features_median %>% layer_dense(units = N[2],weights = list(all_w[[7]], all_w[[8]]), activation = "tanh",  name = "var_median") %>%layer_reshape(c(1,N[2]))
pre_quantile =  features_quantile %>% layer_dense(units = N[2], activation ='relu', weights = list(all_w[[9]], all_w[[10]])) %>%layer_reshape(c(1,N[2]), name = "pre_quantile")
var_quantile = var_median %>% list(pre_quantile) %>% layer_subtract()%>%layer_reshape(c(1,N[2]))

d = layer_dense(units = 1, activation ='linear', kernel_constraint = constraint_nonneg(), bias_constraint = constraint_nonneg())

covar_median = var_median %>%
  layer_reshape(c(N[2],1)) %>%time_distributed(d, weights = list(all_w[[11]], all_w[[12]])) %>% layer_reshape(c(1,N[2]), name = "covar_median")  
covar_quantile = var_quantile %>%
  layer_reshape(c(N[2],1)) %>%time_distributed(d, weights = list(all_w[[11]], all_w[[12]])) %>% layer_reshape(c(1,N[2]), name = "covar_quantile")  

model <- keras_model(
  inputs = list(rates), 
  outputs = c(var_median, var_quantile, covar_median, covar_quantile))

