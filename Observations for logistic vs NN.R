## 
# We compare a logistic regression vs a Neural network aiming to be as similar as possible to
# the logistic. No hidden layer and logistic activation. (I think it's not exactly the same though)
# We use the diamonds dataset from ggplot2.
# This is a balanced dataset
# We use accuracy in validation.
#################
# RESULTS
#################
# Logistic gets close to it's highest value with 30+ observations (100 hits almost the highets point)
# while the neural network requires almost 1000 to get to the same level. At least with this simple
# form, without tuning, just selecting some parameters so it converges.
# NN seems to perform a bit better but might be variance, we should use CV and plot SD.


library(tidyverse)
library(neuralnet)
# diamonds is in ggplot2

# we aim to classify diamonds between Very good and Ideal cut 
# just using the numerical variables
# The dataset can be used to test other models
# example with logistic and RF

df = diamonds %>%
  filter(cut %in% c("Very Good","Premium")) %>%
  mutate(cut = as.character(cut)) %>%
  select_if(function(col) is.numeric(col) | 
              all(col == .$cut)) %>%
  mutate(cut = as.factor(cut))


train_index = sample(1:nrow(df), size = 0.8*nrow(df))

df_train = df[train_index,]
df_test = df[-train_index, ]


nobs = c(10,30,50,100,500,1000,5000,10000,20000)
f <- as.formula("cut ~ carat + depth + table + x + y +z")
# Logistic ----
# premium es 0

set.seed(5) 
results = data.frame()

for (i in nobs){
 
sample_index = sample(1:nrow(df_train),size= i)

#standardize
X_train =  df_train[sample_index,] %>%
  select(-cut)
means <- X_train %>%
  map_dbl(mean)
sd <- X_train %>%
  map_dbl(stats::sd)
scaled <- as.data.frame(scale(X_train, center = means, scale = sd))
df_train_lr = cbind.data.frame(scaled, df_train[sample_index,"cut"]) %>%
  mutate(cut = ifelse(cut =="Premium",0,1))


X_test =  df_test %>%
  select(-cut)
scaled <- as.data.frame(scale(X_test, center = means, scale =sd))
df_test_lr = cbind.data.frame(scaled, df_test[,"cut"])
  
mod = glm(f, data=df_train_lr , family = binomial(link = "logit"))

fit_lr = as.data.frame(predict(mod, df_test_lr, type = "response")) %>%
  setNames("pred") %>% 
  mutate(clase_pred = as.factor(ifelse(pred > 0.5,1,0))) %>%
  cbind.data.frame(df_test_lr$cut) %>%
  rename(cut = 'df_test_lr$cut') %>%
  mutate(clase_test = as.factor(ifelse(cut =="Premium",0,1)))

# Accuracy
val_accuracy = yardstick::metrics(fit_lr, clase_test, clase_pred)[[1,3]]

# Append results
temp = data.frame(classifier = 'logistic', nobs = i, accuracy =val_accuracy)
results = rbind(results, temp)
}



# neural net

set.seed(5)

for (i in nobs){

sample_index = sample(1:nrow(df_train),size= i)
# standardize  
X_train =  df_train[sample_index,] %>%
  select(-cut)
means <- X_train %>%
  map_dbl(mean)
sd <- X_train %>%
  map_dbl(stats::sd)
scaled <- as.data.frame(scale(X_train, center = means, scale = sd))

df_train_nn = cbind.data.frame(scaled, df_train[sample_index,"cut"]) %>%
  mutate(cut = ifelse(cut =="Premium",0,1))


X_test =  df_test %>%
  select(-cut)
scaled <- as.data.frame(scale(X_test, center = means, scale =sd))
df_test_nn = cbind.data.frame(scaled, df_test[,"cut"])




# requires formula in this format

nn <- neuralnet::neuralnet(f,data=df_train_nn,
                           act.fct = "logistic", linear.output = FALSE,
                           threshold = 0.5,
                           learningrate.limit = NULL,
                           learningrate.factor =
                             list(minus = 0.5, plus = 1.2),
                           algorithm = "rprop+",
                           lifesign = "full", 
                           stepmax = 50000)

## Prediction using neural network
fit_nn = as.data.frame(neuralnet::compute(nn,df_test_nn)$net.result) %>%
  setNames("pred") %>% 
  mutate(clase_pred = as.factor(ifelse(pred > 0.5,1,0))) %>%
  cbind.data.frame(df_test_nn$cut) %>%
  rename(cut = 'df_test_nn$cut') %>%
  mutate(clase_test = as.factor(ifelse(cut =="Premium",0,1)))


# metric
val_accuracy = tryCatch(yardstick::metrics(fit_nn, clase_test, clase_pred)[[1,3]],
warning = function(){
  return(0)
},
error = function(cond){
  return(0)
}
)

temp = data.frame(classifier = 'NN logistic',nobs = i, accuracy = val_accuracy)
results = rbind(results, temp)

}

#### plotting results


ggplot(data = results, aes(x= nobs, y =accuracy, color = classifier)) + 
  geom_point(position = position_dodge(width = 0.1)) + 
  scale_x_continuous(breaks = c(0,10,30,50,100,500,1000,5000,10000,20000), trans = "log10") + 
  ggtitle(label = "Comparing observations required to classify a balanced set with Logistic Regression vs Simple Neural Network",
          subtitle = "Less than a couple thousands obs seems insufficient even for a simple model in NN") + 
  xlab("Number of observations") + 
  ylab("Accuracy in validation set")
