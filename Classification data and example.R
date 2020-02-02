
library(tidyverse)
library(pROC)
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


# Logistic ----

# premium es 0
mod = glm(cut ~., data= df, family = binomial(link = "logit"))

summary(mod)
anova(mod)
mod$y

fit = as.data.frame(predict(mod, df, type = "response")) %>%
  setNames("pred") %>%
  mutate(clase_pred = ifelse(pred > 0.5,1,0)) %>%
  cbind.data.frame(df$cut) %>%
  rename(cut = 'df$cut') %>%
  mutate(clase_train = ifelse(cut =="Premium",0,1))

table(fit %>%
        select(clase_train, clase_pred))


diamond_roc = roc(fit$clase_train, fit$pred,
                  plot = TRUE,
                  show.thres=TRUE,
                  print.auc = TRUE,
                  show.thres=TRUE,
                  print.thres = c(0.30,0.35, 0.40, 0.45,0.48, 0.50,0.55, 0.60),
                  print.thres.col = "blue"
                  )


# RandomForest ----

rfmod = randomForest::randomForest(cut ~ ., data = df, importance = TRUE)

fitRF = as.data.frame(predict(rfmod, df)) %>%
  setNames("pred") %>% 
  cbind.data.frame(df$cut) %>%
  rename(cut = 'df$cut') 

table(fitRF %>%
        select(cut, pred))

randomForest::importance(rfmod)
