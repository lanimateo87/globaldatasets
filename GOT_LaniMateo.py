# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:45:22 2019
@author: Lani L. Mateo
Purpose: 
    1. To predict which characters in the series will live or die
    2. Give data-driven recommendations on how to survive in Game of Thrones.
    3. Show feature engineering, variable selection, and model development.
"""

###############################################################################
# SET UP MODULES AND FILES
###############################################################################

# Modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score    # AUC value
from sklearn.metrics import confusion_matrix # confusion matrix

# Data set and data dictionary

file = 'GOT_character_predictions.xlsx'
got = pd.read_excel(file)

file1 = 'GOT_data_dictionary.xlsx'
got_dict = pd.read_excel(file1)

pd.options.display.max_columns = 50

###############################################################################
# GENERAL INFO ABOUT THE DATA
###############################################################################

got.columns # Refer to got_dict for description of each attribute
got.shape
got.info()

"""
Lani:
    There are 1946 GOT characters with 26 attributes.
    There's a lot of null values.
    Check all null values and see if there are unnecessary attributes for
    this analysis. 
    
    8 variables are nominal, 11 integer and 7 float
"""

# Identify attributes with null values

got.isnull().sum() # No. of null values per attribute
got.isnull().sum().sum() # Total number of null values
got.describe()

"""
Lani:
    There are 20606 null values.
    The following attributes have huge number of null values.

    title                         
    culture                      
    dateOfBirth                   
    mother                        
    father                        
    heir                          
    house                          
    spouse                        
    isAliveMother                 
    isAliveFather                 
    isAliveHeir                   
    isAliveSpouse                 
    age                              

    2 characters have negative age - Rhaego and Doreah.
"""

###############################################################################
# DATA REDUCTION, FLAGGING MISSING VALUES AND IMPUTATION
###############################################################################

# Flag missing values

for col in got:
    if got[col].isnull().astype(int).sum() > 0:
        got['m_'+col] = got[col].isnull().astype(int)
        
# Drop unnecessary variables
"""
Lani:
    The dateofBirth is not really a date for all observation and 
    ranges from -28 to 298299. This isn't right.
    We can't use this attribute in this analysis
    Let's drop it.
"""

got = got.drop(['dateOfBirth'], axis=1)

# Impute null values and binary/numeric with 0

got.iloc[:,:] = got.iloc[:,:].fillna(0)

# Verify if data is clean and ready for detailed exploratory data analysis

got.info()
got.isnull().sum().sum() # To verify if all null values are imputed

"""
Lani:  
    All null values are imputed.
    Do some GOT research, explore data further and develop hypothesis.
 """

# Create data type list to reference variables later

nominal = ['name','title','mother','father','heir','house','spouse','culture']

numeric = ['S.No','dateOfBirth','age','numDeadRelations','popularity']

binary = ['book1_A_Game_Of_Thrones', 'book2_A_Clash_Of_Kings',
    'book3_A_Storm_Of_Swords', 'book4_A_Feast_For_Crows',
    'book5_A_Dance_with_Dragons', 'isAliveMother', 'isAliveFather',
    'isAliveHeir', 'isAliveSpouse', 'isMarried', 'isNoble','isAlive',
    'm_title','m_culture','m_dateOfBirth','m_mother','m_father','m_heir',
    'm_house','m_spouse','m_age']

###############################################################################
# DETAILED EXPLORATORY DATA ANALYSIS
###############################################################################

"""
Lani:
    Most of the variables in our data are nominal and binary.
    We can check if there is any significant correlation between variables.
    
    Independent variable: 'isAlive'
"""

# Check for correlation 
got_corr = got.iloc[:,:-13].corr()

plt.figure(figsize=(13,10))
sns.set(style = "white", font_scale= 1)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(got_corr,
            annot = True,
            fmt = ".2f",
            cmap = cmap,
            linecolor = 'white',
            linewidths = 0.05,)

"""
Lani:
    None of the numeric variables have strong association to isAlive.
    numDeadRelations and popularity has strong association at 0.61
"""

# Nobility count in catplot
got.groupby('isNoble')[['isAlive']].sum()/got.isNoble.count()

sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'isNoble',
                 data = got)
plt.show()

"""
651 (33%) of characters are alive and noble and 800 (41%) are not.
There are less people from noble background who are alive but there is no
significant difference for those who are dead.
Maybe use probability for this variable in the model.
"""

# Gender count in catplot
got.groupby('male')[['isAlive']].sum()/got.male.count()
sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'male',
                 data = got)
plt.show()

"""
838 (43%) males are alive while 613 (32%) are women.
There are more males whether alive or dead.
"""

# Appearance in each book
got.groupby('book1_A_Game_Of_Thrones')[['isAlive']].sum()/got.book1_A_Game_Of_Thrones.count()
got.groupby('book2_A_Clash_Of_Kings')[['isAlive']].sum()/got.book2_A_Clash_Of_Kings.count()
got.groupby('book3_A_Storm_Of_Swords')[['isAlive']].sum()/got.book3_A_Storm_Of_Swords.count()
got.groupby('book4_A_Feast_For_Crows')[['isAlive']].sum()/got.book4_A_Feast_For_Crows.count()
got.groupby('book5_A_Dance_with_Dragons')[['isAlive']].sum()/got.book5_A_Dance_with_Dragons.count()


"""
Alive and present in each book
Book 1 - 238 (12%) 
Book 2 - 516 (26%) 
Book 3 - 700 (36%) 
Book 4 - 971 (50%) 
Book 5 - 587 (30%)
"""

# Book Presence - In how many books is the character present?
got['book1to5'] = got['book1_A_Game_Of_Thrones'] + got[
        'book2_A_Clash_Of_Kings'] + got[
        'book3_A_Storm_Of_Swords'] + got[
        'book4_A_Feast_For_Crows'] + got[
        'book5_A_Dance_with_Dragons']
got.book1to5.describe()

"""
To give weight to characters present in all book.
"""
got['out_allbook'] = 0
for val in enumerate(got.loc[ : ,'book1to5']):
    if val[1] == 5:
        got.loc[val[0], 'out_allbook'] = 1

got.groupby('book1to5')[['isAlive']].sum()/got.book1to5.count()

# Book Presence count in catplot
sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'book1to5',
                 data = got)
plt.show()

"""
Alive
530 (27%) of the characters appeared in a book at least once and are still alive.
There are people who did not appear in any of the books but are alive.
"""

# numFamily - How many family affiliation does a character have?
got['numFamily'] = got['isAliveMother'] + got[
        'isAliveFather'] + got[
                'isAliveHeir'] + got[
                        'isAliveSpouse']

got.groupby('numFamily')[['isAlive']].sum()/got.numFamily.count()
got.numFamily.describe()

# numFamily count in catplot
sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'numFamily',
                 data = got)
plt.show()

"""
Most who are alive, 1287 (66%), do not have any family relations
162 (8%) have one
1 have 2
1 have 3
"""

# isAliveMother count in catplot
got.groupby('isAliveMother')[['isAlive']].sum()/got.isAliveMother.count()

sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'isAliveMother',
                 data = got)
plt.show()

"""
Most (74%) who are alive no longer have a mother.
"""

# isAliveFather count in catplot
got.groupby('isAliveFather')[['isAlive']].sum()/got.isAliveFather.count()

sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'isAliveFather',
                 data = got)
plt.show()

"""
Most (74%) who are alive no longer have a father.
"""

# isAliveSpouse count in catplot
got.groupby('isAliveSpouse')[['isAlive']].sum()/got.isAliveSpouse.count()

sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'isAliveSpouse',
                 data = got)
plt.show()

"""
Most (66%) who are alive no longer have a spouse.
"""

# isAliveHeir count in catplot
got.groupby('isAliveHeir')[['isAlive']].sum()/got.isAliveHeir.count()

sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'isAliveHeir',
                 data = got)
plt.show()

"""
Most (74%) who are alive no longer have an heir.
"""

# isMarried count in catplot
got.groupby('isMarried')[['isAlive']].sum()/got.isMarried.count()

sns.catplot(x = 'isAlive',
                 kind = 'count',
                 hue = 'isMarried',
                 data = got)
plt.show()

"""
Most (65%) who are alive are not married.
"""

###############################################################################
# DETAILED EXPLORATORY DATA ANALYSIS
###############################################################################

"""Lani: 
    Exploration of the above variables doesn't give enough information to
    form a model. Explore how number of appearance in books, family
    affiliations, culture, house and title affect odds of survival.
    We can explore and create new variables for these.
"""

# prob_book1 - probability of survival based on appearance in Book 1

got_book1 = got.groupby('book1_A_Game_Of_Thrones')[['isAlive']].sum()
got_book1['prob_book1'] = got_book1/got_book1.isAlive.sum()
got['prob_book1'] = got['book1_A_Game_Of_Thrones'].map(got_book1['prob_book1'])
got.prob_book1.isnull().sum() # check

# prob_book2 - probability of survival based on appearance in the Book 2

got_book2 = got.groupby('book2_A_Clash_Of_Kings')[['isAlive']].sum()
got_book2['prob_book2'] = got_book2/got_book2.isAlive.sum()
got['prob_book2'] = got['book2_A_Clash_Of_Kings'].map(got_book2['prob_book2'])
got.prob_book2.isnull().sum() # check

# prob_book3 - probability of survival based on appearance in the Book 3

got_book3 = got.groupby('book3_A_Storm_Of_Swords')[['isAlive']].sum()
got_book3['prob_book3'] = got_book3/got_book3.isAlive.sum()
got['prob_book3'] = got['book3_A_Storm_Of_Swords'].map(got_book3['prob_book3'])
got.prob_book3.isnull().sum() # check)

# prob_book4 - probability of survival based on appearance in the Book 4

got_book4 = got.groupby('book4_A_Feast_For_Crows')[['isAlive']].sum()
got_book4['prob_book4'] = got_book4/got_book4.isAlive.sum()
got['prob_book4'] = got['book4_A_Feast_For_Crows'].map(got_book4['prob_book4'])
got.prob_book4.isnull().sum() # check

# prob_book5 - probability of survival based on appearance in the Book 5

got_book5 = got.groupby('book5_A_Dance_with_Dragons')[['isAlive']].sum()
got_book5['prob_book5'] = got_book5/got_book5.isAlive.sum()
got['prob_book5'] = got['book5_A_Dance_with_Dragons'].map(got_book5['prob_book5'])
got.prob_book5.isnull().sum() # check

# prob_book1to5 - probability of survival based on appearance in the 5 books

got_book1to5 = got.groupby('book1to5')[['isAlive']].sum()
got_book1to5['prob_book1to5'] = got_book1to5/got_book1to5.isAlive.sum()
got['prob_book1to5'] = got['book1to5'].map(got_book1to5['prob_book1to5'])
got.prob_book1to5.isnull().sum() # check

# prob_numFamily - probability of survival based on number of family relations

got_numFamily = got.groupby('numFamily')[['isAlive']].sum()
got_numFamily['prob_numFamily'] = got_numFamily/got_numFamily.isAlive.sum()
got['prob_numFamily'] = got['numFamily'].map(got_numFamily['prob_numFamily'])
got.prob_numFamily.isnull().sum() # check

# prob_culture - probability of survival based on cultural affiliation

got_culture = got.groupby('culture')[['isAlive']].sum()
got_culture['prob_culture'] = got_culture/got_culture.isAlive.sum()
got['prob_culture'] = got['culture'].map(got_culture['prob_culture'])
got.prob_culture.isnull().sum() # check

# prob_house - probability of survival based on house affiliation

got_house = got.groupby('house')[['isAlive']].sum()
got_house['prob_house'] = got_house/got_house.isAlive.sum()
got['prob_house'] = got['house'].map(got_house['prob_house'])
got.prob_house.isnull().sum() # check

# prob_title - probability of survival based on title

got_title = got.groupby('title')[['isAlive']].sum()
got_title['prob_title'] = got_title/got_title.isAlive.sum()
got['prob_title'] = got['title'].map(got_title['prob_title'])
got.prob_title.isnull().sum() # check

# prob_nobility - probability of survival based on nobility

got_nobility = got.groupby('isNoble')[['isAlive']].sum()
got_nobility['prob_nobility'] = got_nobility/got_nobility.isAlive.sum()
got['prob_nobility'] = got['isNoble'].map(got_nobility['prob_nobility'])
got.prob_nobility.isnull().sum() # check

# prob_marriage - probability of survival based on marriage

got_marriage = got.groupby('isMarried')[['isAlive']].sum()
got_marriage['prob_marriage'] = -1*got_marriage/got_marriage.isAlive.sum()
got['prob_marriage'] = got['isMarried'].map(got_marriage['prob_marriage'])
got.prob_marriage.isnull().sum() # check

# prob_gender - probability of survival based on gender

got_gender = got.groupby('male')[['isAlive']].sum()
got_gender['prob_gender'] = got_gender/got_gender.isAlive.sum()
got['prob_gender'] = got['male'].map(got_gender['prob_gender'])
got.prob_gender.isnull().sum() # check

# prob_alivemother - probability of survival based on mother

got_alivemother = got.groupby('isAliveMother')[['isAlive']].sum()
got_alivemother['prob_alivemother'] = got_alivemother/got_alivemother.isAlive.sum()
got['prob_alivemother'] = got['isAliveMother'].map(got_alivemother['prob_alivemother'])
got.prob_alivemother.isnull().sum() # check

# prob_alivefather - probability of survival based on fother

got_alivefather = got.groupby('isAliveFather')[['isAlive']].sum()
got_alivefather['prob_alivefather'] = got_alivefather/got_alivefather.isAlive.sum()
got['prob_alivefather'] = got['isAliveFather'].map(got_alivefather['prob_alivefather'])
got.prob_alivefather.isnull().sum() # check

# prob_aliveheir - probability of survival based on heir

got_aliveheir = got.groupby('isAliveHeir')[['isAlive']].sum()
got_aliveheir['prob_aliveheir'] = got_aliveheir/got_aliveheir.isAlive.sum()
got['prob_aliveheir'] = got['isAliveHeir'].map(got_aliveheir['prob_aliveheir'])
got.prob_aliveheir.isnull().sum() # check

# prob_alivespouse - probability of survival based on spouse

got_alivespouse = got.groupby('isAliveSpouse')[['isAlive']].sum()
got_alivespouse['prob_alivespouse'] = -1*got_alivespouse/got_alivespouse.isAlive.sum()
got['prob_alivespouse'] = got['isAliveSpouse'].map(got_alivespouse['prob_alivespouse'])
got.prob_aliveheir.isnull().sum() # check

# prob_alivefamily
got['prob_alivefamily'] = (got['prob_alivespouse'] + got[
                'prob_aliveheir'] + got[
                'prob_alivefather'] + got[
                'prob_alivemother'])/4

# Cleaned excel file
got_df = got.select_dtypes(['number'])
got_df.isnull().sum() # check
got_df.to_csv('got_clean.csv')

###############################################################################
# MODEL : RANDOM FOREST IN SCI-KIT LEARN
###############################################################################

# Data preparation
got_data = got.loc[:,['prob_title',
                      'prob_culture',
                      'prob_book1to5',
                      'prob_house',                 
                      'prob_book1',
                      'prob_book4',
                      'popularity',
                      'male',
                      'numDeadRelations',
                      'age']]
got_target = got.loc[:,'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)

# Full forest using gini

full_forest_gini = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 10,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Full forest using entropy

full_forest_entropy = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 10,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Fitting the models

full_gini_fit = full_forest_gini.fit(X_train, y_train)
full_entropy_fit = full_forest_entropy.fit(X_train, y_train)

# Are our predictions the same for each model?

pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))
full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()

# Scoring the gini model

print('Gini Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Gini Testing Score:', full_gini_fit.score(X_test, y_test).round(4))

# Scoring the entropy model

print('Entropy Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Entropy Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

# Saving score objects

gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)

###############################################################################
# Feature importance function
###############################################################################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

plot_feature_importances(full_gini_fit,
                         train = X_train,
                         export = False)

plot_feature_importances(full_entropy_fit,
                         train = X_train,
                         export = False)

###############################################################################
# Repreparing Data
###############################################################################

got_data = got.loc[:,['prob_title',
                      'prob_culture',
                      'prob_book1to5',
                      'prob_house',                 
                      'prob_book1',
                      'prob_book4',
                      'popularity',
                      'male',
                      'numDeadRelations',
                      'age']]
got_target = got.loc[:,'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target.values.ravel(),
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)

###############################################################################
# Running with Gini
###############################################################################

# Full forest using gini
sig_forest_gini = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Fitting the models

sig_fit = sig_forest_gini.fit(X_train, y_train)

# Scoring the gini model

print('Training Score', sig_fit.score(X_train, y_train).round(4))
print('Testing Score:', sig_fit.score(X_test, y_test).round(4))

# Saving score objects

gini_sig_train = sig_fit.score(X_train, y_train)
gini_sig_test  = sig_fit.score(X_test, y_test)

###############################################################################
# Parameter tuning with GridSearchCV
###############################################################################

# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]

param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}

# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)

# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)

# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))

###############################################################################
# Building Random Forest Model Based on Best Parameters
###############################################################################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 15,
                                    n_estimators = 1000,
                                    warm_start = True)

rf_optimal.fit(X_train, y_train)

rf_optimal_pred = rf_optimal.predict(X_test)

print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))

rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)

###############################################################################
# Gradient Boosted Machines
###############################################################################

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 1000,
                                  max_depth = None,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )

gbm_basic_fit = gbm_3.fit(X_train, y_train)
gbm_basic_predict = gbm_basic_fit.predict(X_test)

# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))

gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)

###############################################################################
# Applying GridSearchCV
###############################################################################

# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']

param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}

# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)

# Creating a GridSearchCV object
gbm_grid_cv = RandomizedSearchCV(gbm_grid,
                                 param_grid,
                                 cv = 3,
                                 n_iter = 1000,
                                 scoring = 'roc_auc')

# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))

###############################################################################
# Building GBM Model Based on Best Parameters
################################################################################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.2,
                                      max_depth = 2,
                                      n_estimators = 1000,
                                      random_state = 508)

gbm_optimal.fit(X_train, y_train)
gbm_optimal_score = gbm_optimal.score(X_test, y_test)
gbm_optimal_pred = gbm_optimal.predict(X_test)

# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))

gbm_optimal_train = gbm_optimal.score(X_train, y_train)
gmb_optimal_test  = gbm_optimal.score(X_test, y_test)

###############################################################################
# SAVE RESULTS
###############################################################################

# Saving best model scores
model_scores_df = pd.DataFrame({'RF_Score': [full_forest_cv.best_score_],
                                'GBM_Score': [gbm_grid_cv.best_score_]})

model_scores_df.to_excel("Ensemble_Model_Results.xlsx")

# Save model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': rf_optimal_pred,
                                     'GBM_Predicted': gbm_optimal_pred})

model_predictions_df.to_excel("Ensemble_Model_Predictions.xlsx")

###############################################################################
# AUC
###############################################################################
auc = roc_auc_score(y_test, gbm_optimal_pred)
cv = cross_val_score(gbm_optimal, got_data, got_target, cv = 3)
print(auc.round(3))
print(pd.np.mean(cv).round(3))

cv_gbm_3 = cross_val_score(gbm_optimal,
                           got_data,
                           got_target,
                           cv = 3)


print(cv_gbm_3)
print(pd.np.mean(cv_gbm_3).round(3))

print('\nAverage: ',
      pd.np.mean(cv_gbm_3).round(3),
      '\nMinimum: ',
      min(cv_gbm_3).round(3),
      '\nMaximum: ',
      max(cv_gbm_3).round(3))
