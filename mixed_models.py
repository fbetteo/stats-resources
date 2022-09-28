# https://www.pythonfordatascience.org/mixed-effects-regression-python/#test_with_python

import pandas as pd
import researchpy as rp  # generates academia tables
import statsmodels.api as sm
import scipy.stats as stats

import statsmodels.formula.api as smf

df = pd.read_csv("http://www-personal.umich.edu/~bwest/rat_pup.dat", sep="\t")
df.info()
df

rp.codebook(df)  # quick summary of variables

rp.summary_cont(df.groupby(["treatment", "sex"])
                ["weight"])  # aggregated summary


boxplot = df.boxplot(["weight"], by=["treatment", "sex"],
                     figsize=(16, 9),
                     showmeans=True,
                     notch=True)

boxplot.set_xlabel("Categories")
boxplot.set_ylabel("Weight")

######

model = smf.mixedlm("weight ~ litsize + C(treatment) + C(sex, Treatment('Male')) + C(treatment):C(sex, Treatment('Male'))",
                    df,
                    groups="litter").fit()

model.summary()


# Random Intercept Model w/out Interaction Term

model = smf.mixedlm(
    "weight ~ litsize + C(treatment) + C(sex, Treatment('Male'))", df, groups="litter").fit()

model.summary()


# Random Slope Model: Random intercepts and slopes are independent

model2 = smf.mixedlm("weight ~ litsize + C(treatment) + C(sex)", df, groups="litter",
                     vc_formula={"sex": "0 + C(sex)"}).fit()

model2.summary()
