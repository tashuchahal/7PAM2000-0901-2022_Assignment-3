#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Loading Data

# In[1]:


# Importing Modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting environment to ignore future warnings
import warnings
warnings.simplefilter('ignore')


# In[2]:


# Loading dataset
data = pd.read_csv("data.csv", skiprows=4).iloc[:, :-1]
data.head()


# # Getting Data of Interest

# In[3]:


# FIlling missing values
data.fillna(0, inplace=True)


# In[4]:


fields = ['Population, total',
          'Renewable electricity output (% of total electricity output)',
        'Electricity production from oil sources (% of total)',
       'Electricity production from nuclear sources (% of total)',
       'Electricity production from natural gas sources (% of total)',
       'Electricity production from hydroelectric sources (% of total)',
       'Electricity production from coal sources (% of total)']

df = data[data["Indicator Name"].isin(fields)]


# In[5]:


# Dropping records that are not for Countries
not_country = ['Euro area', 'IDA blend',
    'Middle East & North Africa (excluding high income)',
    'Africa Western and Central',
    'Middle East & North Africa (IDA & IBRD countries)',
    'Central Europe and the Baltics',
    'Middle East & North Africa (IDA & IBRD countries)'
    'Middle East & North Africa', "Arab World",
    'Europe & Central Asia (excluding high income)',
    'Africa Eastern and Southern', 'Low income',
    'Latin America & Caribbean (excluding high income)',
    'Europe & Central Asia (IDA & IBRD countries)',
    'Heavily indebted poor countries (HIPC)', 'European Union',
    'Latin America & the Caribbean (IDA & IBRD countries)',
    'Latin America & Caribbean', 'Pre-demographic dividend',
    'Fragile and conflict affected situations',
    'Least developed countries: UN classification',
    'Sub-Saharan Africa (excluding high income)', 'Sub-Saharan Africa',
    'Sub-Saharan Africa (IDA & IBRD countries)', 'IDA only',
    'Europe & Central Asia', 'IDA total',
    'Post-demographic dividend', 'High income', 'OECD members',
    'South Asia (IDA & IBRD)', 'South Asia',
    'East Asia & Pacific (IDA & IBRD countries)',
    'East Asia & Pacific (excluding high income)', 'East Asia & Pacific',
    'Late-demographic dividend', 'Upper middle income',
    'Lower middle income', 'Early-demographic dividend', 'IBRD only',
    'Middle income', 'Low & middle income', 'IDA & IBRD total', 'World']

df = df[~df["Country Name"].isin(not_country)]


# In[6]:


# Dropping Unnecessary Issues
df.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)


# # Analysis

# In[7]:


avg_population = df[df["Indicator Name"] == "Population, total"].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_oil_elec = df[df["Indicator Name"] == 'Electricity production from oil sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_nuc_elec = df[df["Indicator Name"] == 'Electricity production from nuclear sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_gas_elec = df[df["Indicator Name"] == 'Electricity production from natural gas sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_hyd_elec = df[df["Indicator Name"] == 'Electricity production from hydroelectric sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_coal_elec = df[df["Indicator Name"] == 'Electricity production from coal sources (% of total)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
avg_renew_elec = df[df["Indicator Name"] == 'Renewable electricity output (% of total electricity output)'].drop(["Indicator Name"], axis=1).set_index("Country Name").mean(axis=1)
