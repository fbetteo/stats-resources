library(tidyverse)
rm(list = ls())

iris_nolabel = iris %>%
  select(-Species) %>%
  as.matrix()


# PCA
pca = prcomp(iris %>% select(-Species))

head(pca$rotation) # principal components -> Direccion de componentes max varianza

head(pca$x) # scores -> data centrada * rotation = data centrada * componentes principales
# son las nuevas variables combinacion lineal de las originales

# Los loadings son la varianza de los scores
pca$sdev^2

# plot de las dos primeras dimensiones de scores
flag = c(1,1, rep(0, 148)) # marco las 2 primeras obs para ver en el plot
factoextra::fviz_pca_ind(prcomp(iris %>% select(-Species)), label="none",
                                         habillage=flag,
                                        title="",
                                        mean.point=F,
                                        labelsize=3)


factoextra::fviz_pca_ind(prcomp(iris %>% select(-Species)), label="none",
                         habillage=iris$Species,
                         title="",
                         mean.point=F,
                         labelsize=3)

# PCA mediante SVD

# hay que centrar a la media para que X^tX sea numerador de covarianzas
# si dividis por raiz de N te da exactamente igual
# si no te cambia la escala de los loadings pero mismas conclusiones
# pero hay que volver a multiplicar para conseguir los scores creo
# mejor hacerlo sin dividir por raiz de n

center_n = function(x){
  meanx = mean(x)
  return(x-mean(x))
  # n = length(x)
  # return ((x - meanx)/sqrt(n))

}


iris_nolabel_centered = iris %>%
  select(-Species) %>%
  map_df(center_n) %>%
  as.data.frame()


svd_pca = svd(iris_nolabel_centered)

# singular values ^2
svd_pca$d^2/sum(svd_pca$d^2)

# V = principal components
# V es la base de eigenvector orthonormal
# Recordar que en SVD se la traspone 
svd_pca$v
pca$rotation

# Scores -> Data centrada * principal componentes
# Scores -> U*D  (por que?) Se ve cuando vas haciendo la descomposicion
# Scores -> Data centrada * V
head(svd_pca$x)
head(as.matrix(iris_nolabel_centered)%*%pca$rotation)
head(svd_pca$u%*%diag(svd_pca$d))
head(as.matrix(iris_nolabel_centered)%*%svd_pca$v)


# SVD como descomposicion. Sin centrar

svd = svd(iris_nolabel)

# inspeccionando U
svdu = svd$u %>%
  as.data.frame() %>%
  cbind.data.frame(iris$Species) %>%
  arrange(abs(V1)) %>%
  rename(species =`iris$Species` )


svd$d

svdv = svd$v %>%
  as.data.frame()

svdu_t = svdu %>%
  pivot_longer(cols = -species) 

ggplot(data = svdu, aes(x = V1, y = V2)) + 
  geom_point(aes(color = as.factor(species)))

# No hay enormes diferencias en valor absoluto de la primera componente de U entre obs
# pero virginica es la que tiene consistentemente mas.
# Gran valor de singular value para la primera componente

# la primera columna de V (sin transponer, los eigenvectors) tiene mayores val
# absolutos Sepal Length > Petal Length > Sepal Width


