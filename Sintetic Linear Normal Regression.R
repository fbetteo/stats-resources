# Snippet to generate linear relationships. Variance can be constant or variable. 

library(ggplot2)

set.seed(1)

# parameters ----
n = 1000
b0 = 10
b1 = 3
x1 = runif(n, 1, 20)
constant_variance = 25
unconstant_variance_structure = 1 + 3*x1


# linear regression. Fixed SD. ----
# Equivalents
y = b0 + b1*x1 + rnorm(n, 0, sqrt(constant_variance))
y =  MASS::mvrnorm(n = 1, mu = b0 + b1*x1, Sigma = diag(constant_variance, n , n))

# linear regression. SD depends on X1 ----
y =  MASS::mvrnorm(n = 1, mu = b0 + b1*x1, Sigma = diag(unconstant_variance_structure))

# DF ----
df = data.frame(y, x1)

# check ----
ggplot( data  = df, aes( x = x1, y = y)) + 
  geom_point()

reg= lm(y ~ x1, data = df)
sd(reg$residuals)
summary(reg)

