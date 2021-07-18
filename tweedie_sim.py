from scipy import stats
import numpy as np
import pandas as pd
from tweedie import tweedie
import seaborn as sns
import statsmodels.api as sm


n = 1000
p = [1.1, 1.5, 1.9]  # constante
mu = [1, 10, 50]  # media
phi = [1, 10, 50]  # var

df1 = pd.DataFrame({'p': p})
df2 = pd.DataFrame({'mu': mu})
df3 = pd.DataFrame({'phi': phi})

# cross join, se puede hacer mejor?
df = df1.assign(key=1).merge(df2.assign(key=1), on="key").merge(
    df3.assign(key=1), on='key').drop('key', axis=1)


def sim_tweedie(df):
    """ simula tweedie y guarda los parametros """
    p = df.p
    mu = df.mu
    phi = df.phi

    res = tweedie(mu=mu, p=p, phi=phi).rvs(n)
    return res, p, mu, phi


def plot_tweedie(data):
    """ Plot distribucion de la muestra y printea los parametros """
    g = sns.displot(data[0])
    g.fig.suptitle(f'p = {data[1]} , mu = {data[2]}, phi = {data[3]} ')
    #g.set_title(f'p = {data[1]} , mu = {data[2]}, phi = {data[3]} ')
    return g


a = df.apply(sim_tweedie, axis=1)

plot_tweedie(a[0])

for i in a:
    plot_tweedie(i)

# simulo una muestra de medias tomadas de tweedie y plot grafico
means = []
for a in range(1, 1000):
    t = tweedie(mu=60, p=1.2, phi=50).rvs(5000).mean()
    means.append(t)

t_s = pd.Series(means)
sns.displot(t_s)


########
CONVERSION = 0.1
SCALE = 50
DIF = 0.05  # percent
N = 2000  # we want to find out (compras)
ITERATIONS = 1000
# el N total (busquedas) es (N / conversion)

means1 = np.zeros(ITERATIONS)
means2 = np.zeros(ITERATIONS)


def generate_samples(n, conversion, iterations, scale, dif):
    no_purchase = np.random.binomial(
        1, 1 - conversion, (round(n/conversion), iterations))  # zeros
    y1 = (1-no_purchase)*np.random.exponential(scale,
                                               no_purchase.shape)  # revenue1
    y2 = (1-no_purchase)*np.random.exponential(scale *
                                               (1+dif), no_purchase.shape)  # revenue2
    means1 = np.mean(y1, axis=0)
    means2 = np.mean(y2, axis=0)
    return y1, y2, means1, means2


y1, y2, m1, m2 = generate_samples(N, CONVERSION, ITERATIONS, SCALE, DIF)

tstats = []
pvalues = []

for a, b in zip(y1.T, y2.T):
    tstats_t, pvalues_t = stats.ttest_ind(a, b, equal_var=False)
    tstats.append(tstats_t)
    pvalues.append(pvalues_t)


ALPHA = 0.57
POWER = 0.8


def calculate_power(pvalues, alpha):
    return pd.Series([1 for p in pvalues if p < alpha]).sum()/len(pvalues)


def calculate_significance(pvalues, power):
    return pd.Series(pvalues).quantile(power)


calculate_power(pvalues, ALPHA)
calculate_significance(pvalues, POWER)

sns.distplot(pvalues)

for a, b in zip(y1.T, y2.T):
    print(a, b)
y1
for a, b in zip(y1, y2):
    print(a, b)

y1_df

for x, y in y1.T, y2.T:
    print(x, y)

np.dstack((y1, y2)).shape


a, b = stats.ttest_ind(y1[:, 0], y2[:, 0], equal_var=False)
a
b
vars(a)

y1.apply(sum)

# remplazar exponential con lo que corresponda
for i in range(0, ITERATIONS):
    no_purchase = np.random.binomial(
        1, 1 - CONVERSION, round(N/CONVERSION))  # los zeros
    y1 = (1-no_purchase)*np.random.exponential(SCALE,
                                               len(no_purchase))  # revenue1
    y2 = (1-no_purchase)*np.random.exponential(SCALE *
                                               (1+DIF), len(no_purchase))  # revenue2
    means1[i] = np.mean(y1)
    means2[i] = np.mean(y2)

y1.shape

stats.ttest_ind(y1, y2, equal_var=False)

a = np.expand_dims(m1, axis=1)
b = np.expand_dims(m2, axis=1)
df = np.concatenate([a, b], axis=1)


sns.displot(y1)
sns.displot(df, kde=True)
sns.displot(means1, kde=True)
sns.displot(means2, kde=True)


x1 = np.random.exponential(SCALE, (N, ITERATIONS))


a = np.random.binomial(1, 0.5, (5, 2))
a
y1 = (1-a)*np.random.exponential(SCALE, a.shape)  # revenue1
np.mean(y1, axis=0)
a.shape
