# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:32:25 2018

@author: Lani L. Mateo

Working Directory:
/Users/DELL/OneDrive - Education First/Desktop/PyCourses

Purpose:
To explore World Bank 2014 data focusing on Central Africa 1 Region and identify
ways to:

1. Flag anomalies in the data set.
2. Present trends and outliers.
3. Deliver insights and recommendations to further improve data in the region.

"""
# Step 1: Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Import dataset
wb = pd.read_excel ('world_data_hult_regions(1).xlsx')
wb_ext_env = pd.read_excel ('worldbank_sharks_full_env.xlsx')

# Step 3: Correct typo in data set
wb = wb.replace('Central Aftica 1','Central Africa 1')
wb.rename(columns={'CO2_emissions_per_capita)':'CO2_emissions_per_capita'}, 
                 inplace=True)

# Step 4: Subset wb for 'Central Africa 1':
wb_CA1 = wb[wb.Hult_Team_Regions ==
           	'Central Africa 1'].sort_values('country_name')

# Step 5: Identify missing values:
for col in wb:
    print(col)
    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    if wb[col].isnull().astype(int).sum() > 0:
        wb['m_'+col] = wb[col].isnull().astype(int)
        
# Step 6: Get info about the full dataset:
print(wb.head())
print(wb.describe())
print(wb.info())
print(wb.Hult_Team_Regions.unique())

# Step 7: Define unique Data Frames
wb_mean  = pd.DataFrame.copy(wb)
wb_median = pd.DataFrame.copy(wb_mean)
wb_dropped = pd.DataFrame.copy(wb_median)

# Step 8: Create Loops for mean, median or define dropped
# Loop for wb_mean
for col in wb_mean:
    """ Impute missing values using the mean of each column """
    if wb_mean.iloc[:,6].isnull().astype(int).sum() > 0:
        col_mean = wb_mean.iloc[:,6].mean()
        wb_mean.iloc[:,6:] = wb_mean.iloc[:,6:].fillna(col_mean).round(2)

# Loop for wb_median
for col in wb_median:
    """ Impute missing values using the median of each column """
    if wb_median.iloc[:,6].isnull().astype(int).sum() > 0:
        col_median = wb_median.iloc[:,6].median()
        wb_median.iloc[:,6:] = wb_median.iloc[:,6:].fillna(col_median).round(2)

# Dropna() for wb_dropped
wb_dropped = wb_dropped.dropna().round(2)

print(wb)

##############################################################################
# Step 9: Analyze Environment Indicators - Lani Mateo
# A. Corrections in dataframe
"""
Marce, please add the following code under the correction of typo errors:
    
wb.rename(columns = {'CO2_emissions_per_capita)':'CO2_emissions_per_capita'}, 
                 inplace = True)
"""
# B1. Exploring CO2 emissions per capita and Air Pollution
print (wb_CA1.CO2_emissions_per_capita.describe())
print (wb_CA1.avg_air_pollution.describe())

# B2. Which indicators correlate with the 2 in Central Africa 1 Region?
wb_CA1_corr = wb_CA1.corr().round(2).sort_values(
        by ='CO2_emissions_per_capita',ascending = False)

print (wb_CA1_corr)

plt.figure(figsize=(15,15))
sns.set(font_scale=1.5)
sns.heatmap(wb_CA1_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.1)

plt.title ('Correlations (Central Africa 1 Region)')

"""
Based on the correlations, prioritize exploration of CO2 emissions with
>>> exports_pct_gdp (0.61).

HIV incidence and Homicides per 100k have 0.63 and 0.62 correlations
respectively but there isn't a story to tell unless otherwise, because these
are highly correlated to characteristics of urbanization. However, there is
weak correlation between CO2 emissions per capita and urban population and
urban growth percentage. It must be a spurious correlation but taking note of
it here just in case some story reveals itself in the future.
    
"""
# C1. Exploration of trends in CO2 emissions compared to other variables
'In the interest of time, I will focus exploratory analysis on CO2 emissions.'

# C2. Environment variable codes
"""
Studies show that key boosters of CO2 emissions are economic development and
energy efficiency. % of exports from GDP would be good indicator of economic
development. In terms of energy efficiency, maybe exploring access to electricty
in urban areas is a good start but need to explore other factors outside
of the data set like % share of agriculture industry in GDP since it is
the top 3 contributor in CO2 emissions.

Other variables than can be explored against CO2 emissions
'gdp_usd'
'urban_population_pct'
'urban_population_growth_pct'
'access_to_electricity_urban' 
'average_air_pollution'

'Exports of goods and services(% of GDP)'
'GDP(current US$)'
'Urban Population(% of total)'
'Urban population growth(annual%)'
'Access to electricity(% of population)'
'CO2 emissions per capita'

"""
# D. How to present analysis. 
# D1. Definition of X and Y axis, size of markers 
# Global variables for all plots D2a, D2b and D2c
X = 'exports_pct_gdp' # Replace to check other variables for X axis
Y = 'CO2_emissions_per_capita' # Replace to check other variables for Y axis
S = 'urban_population_growth_pct' # Replace to modify size of markers in scatterplot
T = 'Exports of goods and services(% of GDP)' # Replace to modify title
H = 'income_group'

# D2. Choose which bests represents your analysis. 
# D2a. Using lmplot
sns.set(font_scale=1.5)
A1 = sns.lmplot(x = X,
                y = Y,
                hue = "income_group",
                height=8,
                aspect= 1.5,
                fit_reg = True,
                scatter_kws={"s": 400},
                palette = 'plasma',
                data = wb_CA1)
A1._legend.set_title('Income Group')
plt.text (x = 47.69, y= 4.96, s = 'World')
#plt.plot([0, 4.96], [1.5, 0], linewidth=2) # If you want to draw a line on the chart
plt.title ('Global CO2 emissions per capita and Exports % GDP') # Manually update
#plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.show()

help(plt.text)

# D2b. Using scatterplot
plt.figure(figsize=(14,8))
A2 = sns.scatterplot(x = X,
                     y = Y,
                     sizes = (80,500),                                    
                     hue = 'income_group',
                     size = S,
                     legend = 'brief',
                     palette = 'plasma',
                     data = wb_CA1)
A2.text(47.69, 4.7+(-.05),#change these values based on x,y coordinates of marker
            'Equatorial Guinea',
            horizontalalignment='left',
            size='medium',
            color='black',
            weight='medium')
plt.legend(loc=0, prop={'size': 15})
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.show()

# D2c. Using regplot
plt.figure(figsize=(14,8))
A3 = sns.regplot(x = X,
                 y = Y,
                 marker = 'o',
                 scatter_kws = {'s':500},
                 data = wb_CA1) 
A3.text(47.69, 4.7+(-.05),# change these values based on x,y coordinates of marker
           'Equatorial Guinea',# change based on annotation for specific marker
            horizontalalignment='left',
            size='medium',
            color='black',
            weight='light')    
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.show()


# D3. Using swarmplots
plt.figure(figsize=(14,8))
sns.stripplot(x = X,
              y = Y,
              data = wb,
              size = 10,
              orient = 'v')
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.show()


# D4. Using boxplot
plt.figure(figsize=(10,4))
plt.boxplot(x = Y,
            data = wb_CA1,
            vert = False,
            patch_artist = True,
            meanline = True,
            showmeans = True)
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('CO2 emissions per capita') # Use if indicators are abbreviated or mispelled
plt.show()



