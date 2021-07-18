library(tidyverse)

# buscando estructura en los datos de status de F1

results = read_csv("D:/Data Science/kaggle-Formula1/data/results.csv")
races = read_csv("D:/Data Science/kaggle-Formula1/data/races.csv")
constructor = read_csv("D:/Data Science/kaggle-Formula1/data/constructors.csv")
driver = read_csv("D:/Data Science/kaggle-Formula1/data/drivers.csv")

# Filtro un poco. Igual termine filtrando antes del plot de nuevo, quizas es al pedo esto
df = results %>%
  left_join(races, by = "raceId", suffix = c("","_race")) %>%
  left_join(constructor, by = "constructorId", suffix =c("", "_constructor")) %>%
  left_join(driver, by = "driverId", suffix =c("", "_driver")) %>%
  filter(year > 2005 & (constructorId <= 9 | constructorId == 131 | constructorId > 207) ) %>%
  distinct(raceId, grid,.keep_all = T)

# pongo cada status como una columna (finalizo carrera, colision, +1 lap, etc)
# tambien est? posicion inicial como variable
df_features = df %>%
  select(raceId,grid, statusId) %>%
  mutate(statusId =paste0("s_",statusId)) %>%
  mutate(statusId2 = as.factor(statusId)) %>%
  mutate(dummy = 1) %>%
  pivot_wider( id_cols = c(raceId, grid), names_from = statusId2, values_from = dummy, 
              values_fill = 0) %>%
  select(-raceId)
# si saco grid sale -1 y todo el resto 0 en la right singular. Muy sparse?


# la magia
svd = svd(df_features)

# asocio resultados con observaciones originales
svd2 = cbind.data.frame(df, svd$u) %>%
  setNames(c(names(df), paste0("leftsingular",seq(1, dim(svd$u)[2]))))

dim(svd$u)
svd$d # peso asociado a los singular vectors

# chusmeo qu? 
svdv = svd$v %>%
  as.data.frame() %>%
  cbind.data.frame(id = seq(1, dim(svd$v)[1])) %>%
   arrange(desc(abs(V2))) %>%
  select(id, V2)

# elegir uno. Filtrar por piloto nomas o por escuderia
# muchas obs, es para poder plotear y entender qu? sucede
svd2_filt = svd2 %>% filter(driverId %in% c(1,154))
#svd2_filt = svd2 %>% filter(constructorId %in% c(6,131))

# formato long para facet wrap
svd2_filt_long = pivot_longer(svd2_filt, cols = starts_with("leftsingular"), names_to ="component", values_to = "value") %>%
  filter(as.double(str_remove(component,"leftsingular")) <5) # componentes 1 a 3


ggplot(data = svd2_filt_long, aes( x=  year,y = abs(value), color = surname)) + 
  geom_point(position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.5)) + #jiter dodge porque se pisan valores
# tanto entre pilotos como intra piloto. Esto ultimo me da malaspina (?)
  scale_colour_brewer(palette = "Set1") + 
  facet_wrap(facets = ~ component, nrow = 4)




