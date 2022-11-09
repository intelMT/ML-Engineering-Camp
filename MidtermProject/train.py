#!/usr/bin/env python
# coding: utf-8

# import openml
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Using the openml dataset ID:44062 `Brazilian Houses to Rent` <br>
# 
# Original raw data with column descriptions:  <br>
# https://www.kaggle.com/datasets/rubenssjr/brasilian-houses-to-rent?select=houses_to_rent_v2.csv <br>
# 
# The data used, clean: `Brazilian_houses v.8 on OpenML` <br>
# https://www.openml.org/search?type=data&status=active&sort=date&id=44062

### NEXT COMMENTED LINES ARE USED TO DOWNLOAD DATA FROM OPENML. ###

# dataset = openml.datasets.get_dataset(44062, download_data=False)
# X, y, categorical_indicator, attribute_names = dataset.get_data(
#     dataset_format="array", target=dataset.default_target_attribute
# )
# df = pd.DataFrame(X, columns=attribute_names)
# df.to_csv("data/brazilian_houses_openml_v8.csv")

df = pd.read_csv("data/brazilian_houses_openml_v8.csv", index_col=0)

df.columns = df.columns.str.replace("[()]", "", regex=True)
df.columns = df.columns.str.lower()

cat_columns = ['city', 'animal', 'furniture']
num_columns = ['area', 'rooms', 'bathroom', 'parking_spaces', 'hoa_brl','rent_amount_brl', 
               'property_tax_brl', 'fire_insurance_brl']
cat_columns, num_columns

# Target variable is the rent which log1p-transformed. 
actual_total_rent = np.exp(df.target) - 1 

df.target.hist(bins=35)
plt.xlabel("Log1p transformed rent.")
plt.ylabel("Counts")
plt.show()

print(df.info())

# Look at only numerical values
print(df.loc[:,num_columns].describe())

# Lets see how categorical variables are distributed.
# * City takes values from 0 to 4, as integers, 5 cities.
# * Furniture and Animal describe binary features: is the house furnitured or does owner NOT allow animals?


print("The percentage of houses that reject animals/pets in the house: {}%".format(np.round(df.animal.mean()*100,1)))


print("The percentage of furnitured houses: {}%".format(np.round(df.furniture.mean()*100,1)))

# We see that it is coded inversely to intuition. 75.6% of houses should be without furniture. (as verifiable on original raw data in Kaggle)

# reversing the code to mean furnitured and animal-permitted.
df.furniture = (df.furniture ==0).astype(int)
print("The percentage of furnitured houses: {}%".format(np.round(df.furniture.mean()*100,1)))
df.animal = (df.animal == 0).astype(int)
print("The percentage of pet-friendly houses: {}%".format(np.round(df.animal.mean()*100,1)))

# Some EDA based on cat. descriptors
print(df.groupby('city').mean())
print(df.groupby('animal').mean())
print(df.groupby('furniture').mean())
print(df.pivot_table(values='target', index='city', columns='rooms').round(2))

# We see that rents increase with increasing number of rooms. <br>
# City-4 is the most expensive city. then City-3 comes second most expensive city to rent. Others seem comparable.

# We can leave the animal and furniture categorical variables as they are since they are binary. We need to OHE the city variable.


# see the percentages of city data in whole dataset.
# print(df['city'].value_counts())

# From Kaggle csv2 file: I inferred:
city_num2name = {4: 'Sao Paulo', 3: 'Rio de Janeiro', 0:'Belo Horizonte', 2:'Porto Alegre', 1:'Campinas'}
city_name2num = {val:key for key,val in city_num2name.items()}

# city_name2num 

df['city'] = df['city'].map(city_num2name)

sns.barplot(data=df, x='city', y='furniture')
plt.show()

# We see that in city-0 (Belo Horizente) and city-1 (Campinas) have lower portion of their houses furnitured. 

sns.barplot(data=df, x='city', y='animal')
plt.show()

# Porte Alegre has the least animal tolerance.

df.corrwith(df["target"]).round(2)

sns.heatmap(df.corr() > 0.9)
plt.show()

df['rent_amount_brl'].corr(df['fire_insurance_brl'])

# We can see that rent_amount and fire_insurance are highly correlated, causing multicollinearity. <br>
# We can remove one or both of them, I prefer to delete `rent_amounth_brl` to prevent multicollinearity. <br>
# Also, the task is to predict total rent amount, majority of which is this column.

sns.heatmap(df.corr().round(2))
plt.show()

del df['rent_amount_brl']

# Training the data

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['city'])
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42, stratify=df_full_train['city'])

df_train.shape, df_val.shape, df_test.shape

y_train = df_train.pop('target')
y_val = df_val.pop('target')
y_test = df_test.pop('target')
df_train.shape, df_val.shape, df_test.shape

print("Creating DictVectorizer...")
# Turn city into 5 columns for 5 cities
dv = DictVectorizer(sparse=False)

train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)


# X_train.shape, X_val.shape, X_test.shape

# Base model: Linear regression
print("Training begins for LinearRegression...")

mm_scaler = MinMaxScaler()
model = LinearRegression()

X_train_s = mm_scaler.fit_transform(X_train)
X_val_s = mm_scaler.transform(X_val)
X_test_s = mm_scaler.transform(X_test)

model.fit(X_train_s, y_train)

print("Linear Regression training R2 score")
print(model.score(X_train_s, y_train).round(2))

feat_importances = {name:val for name, val in zip(dv.get_feature_names_out() , model.coef_)}

print("Feature Importances based on MinMaxScaler")
print(sorted(feat_importances.items(), key=lambda x: abs(x[1]), reverse=True))

""" The Homeowners association tax (`hoa_brl`), `fire_insurance_brl` seems to be dominant factor in linear regression with 
min-max scaling. The features `property_tax_brl`, `rooms`, `bathroom` and `area`  are the next important features. 
This may change with scaling method applied, i.e. standard scaler. <br>
For this task, robust scaler results in the feature importances order that makes more sense since it is robust to outliers, 
as shown below. The R2 scores do not change.
"""
rb_scaler = RobustScaler()
model = LinearRegression()

X_train_s = rb_scaler.fit_transform(X_train)
X_val_s = rb_scaler.transform(X_val)
X_test_s = rb_scaler.transform(X_test)

model.fit(X_train_s, y_train)
feat_importances = {name:val for name, val in zip(dv.get_feature_names_out() , model.coef_)}

print("Feature Importances based on RobustScaler")
print(sorted(feat_importances.items(), key=lambda x: abs(x[1]), reverse=True))

# For this this simple linear regression is meaningless to split data into train/test/val because there is no hyperparam.
# Lets see other scores anyways.
print("Linear Regression Validation score: {:.2f}".format(model.score(X_val_s, y_val)))
print("Linear Regression Test score: {:.2f}".format(model.score(X_test_s, y_test)))

# XGBoost Model Training and Tuning

print("Training begins for XGBoost...")

dtrain = xgb.DMatrix(X_train, label=y_train)

xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)

y_train_pred = model.predict(dtrain)
mean_squared_error(y_train, y_train_pred)

dval = xgb.DMatrix(X_val)
dtest = xgb.DMatrix(X_test)

y_val_pred = model.predict(dval)
y_test_pred = model.predict(dtest)

print("XGBoost Validation MSE: ")
print(mean_squared_error(y_val, y_val_pred))

print("XGBoost Test MSE: ")
print(mean_squared_error(y_test, y_test_pred))

# The results look good. For Xgboost, we can optionally remove the `rent_amount_brl` column from data as it contains far too much of the target variable even though target variable is log-1p-transformed.

# Root mean square error of training for the actual value in BRL
# np.sqrt(mean_squared_error(np.expm1(y_train_pred), np.expm1(y_train.values)))

# dv.get_feature_names_out()

# # it is at index 12.
# ind = list(np.arange(0,14))
# ind.remove(12)
# print(ind)

# X_train = X_train[:,ind]
# X_test = X_test[:,ind]
# X_val = X_val[:,ind]

# X_train.shape, X_val.shape, X_test.shape

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(xgb_params, dtrain, num_boost_round=175)
y_train_pred = model.predict(dtrain)
print("XGBoost Training MSE:")
print(mean_squared_error(y_train, y_train_pred).round(4))

dval = xgb.DMatrix(X_val)
dtest = xgb.DMatrix(X_test)

y_val_pred = model.predict(dval)
y_test_pred = model.predict(dtest)

print("XGBoost Validation MSE:")
print(mean_squared_error(y_val, y_val_pred).round(4))
print("XGBoost Test MSE:")
print(mean_squared_error(y_test, y_test_pred).round(4))

# np.expm1(y_train_pred), np.expm1(y_train.values)

print("Tuning eta...")
eta_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
results = []

for eta in eta_values:    
    xgb_params = {
        'eta': eta, 
        'max_depth': 3,
        'min_child_weight': 1,

        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=175)
    y_val_pred = model.predict(dval)
    results.append(mean_squared_error(y_val, y_val_pred))


plt.plot(eta_values, results)
plt.xlabel("Eta")
plt.ylabel("MSE of Validation")
plt.show()

# Using 'eta' == 0.2 as best value from previous graph

print("Tuning max_depth...")
results = []
max_depth = [2,3,4,5,6,7,8,9,10]

for depth in max_depth:    
    xgb_params = {
        'eta': 0.2, 
        'max_depth': depth,
        'min_child_weight': 1,

        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=175)
    y_val_pred = model.predict(dval)
    results.append(mean_squared_error(y_val, y_val_pred))

plt.plot(max_depth, results)
plt.xlabel("Max Depth")
plt.ylabel("MSE of Validation")
plt.show()

# Using 'max_depth' == 7 as best value from previous graph
print("Tuning min_child_weights...")
results = []
min_child_weights = [1,2,3,4,5,6,7,8,9,10]

for mcw in min_child_weights:    
    xgb_params = {
        'eta': 0.2, 
        'max_depth': 7,
        'min_child_weight': mcw,

        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=175)
    y_val_pred = model.predict(dval)
    results.append(mean_squared_error(y_val, y_val_pred))


plt.plot(min_child_weights, results)
plt.xlabel("Minimum Child Weight")
plt.ylabel("MSE of Validation")
plt.show()


# Using 'min_child_weight' == 1 as best value from previous graph

print("Tuning for best number of boosting rounds...")
train_results = []
val_results = []
n_boost_rounds = [30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]

xgb_params = {
    'eta': 0.2, 
    'max_depth': 7,
    'min_child_weight': 2,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

for nboost in n_boost_rounds:   
    model = xgb.train(xgb_params, dtrain, num_boost_round=nboost)
    y_train_pred = model.predict(dtrain)
    train_results.append(mean_squared_error(y_train, y_train_pred))
    y_val_pred = model.predict(dval)
    val_results.append(mean_squared_error(y_val, y_val_pred))


# Lets see when we start to overfit...
# plt.plot(n_boost_rounds, train_results, color='g', label="Training")
plt.plot(n_boost_rounds, val_results, color='r', label="Validation")
plt.xlabel("Number of boosting rounds")
plt.ylabel("MSE")
plt.legend()
plt.show()

print("Best n_boost value")
print(n_boost_rounds[np.argmin(val_results)])

# Fine tune n_boost. Not needed much so skipped...

# train_results = []
# val_results = []
# n_boost_rounds = np.arange(130,171)

# xgb_params = {
#     'eta': 0.3, 
#     'max_depth': 4,
#     'min_child_weight': 3,

#     'objective': 'reg:squarederror',
#     'eval_metric': 'rmse',

#     'nthread': 8,
#     'seed': 1,
#     'verbosity': 1,
# }

# for nboost in n_boost_rounds:   
#     model = xgb.train(xgb_params, dtrain, num_boost_round=nboost)
#     y_train_pred = model.predict(dtrain)
#     train_results.append(mean_squared_error(y_train, y_train_pred))
#     y_val_pred = model.predict(dval)
#     val_results.append(mean_squared_error(y_val, y_val_pred))

# # Lets see when we start to overfit...
# # plt.plot(n_boost_rounds, train_results, color='g')
# plt.plot(n_boost_rounds, val_results, color='r')
# plt.show()

print("Now training on the full train dataset.")

y_full_train = df_full_train.pop('target')
df_full_train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.transform(df_full_train_dicts)

dFullTrain = xgb.DMatrix(X_full_train, label=y_full_train)

xgb_params = {
    'eta': 0.2, 
    'max_depth': 7,
    'min_child_weight': 2,

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
 
model = xgb.train(xgb_params, dFullTrain, num_boost_round=250)

y_full_train_pred = model.predict(dFullTrain)
y_test_pred = model.predict(dtest)
print("Final Training MSE:")
print(mean_squared_error(y_full_train, y_full_train_pred).round(5))
print("Test MSE:")
print(mean_squared_error(y_test, y_test_pred).round(5))


# We can see that we improved the results with model tuning.

print("Saving the final xgboost model and dict_vectorizer...")

with open("bin/xgboost_brazil_house.bin", 'wb') as f:
    pickle.dump(model, f)


with open("bin/dictvec.bin", 'wb') as f:
    pickle.dump(dv, f)

print("Saved.")

print("Starting training Random Forest Regressor")

rf_model = RandomForestRegressor(n_estimators=200,
                                 max_depth=10,
                                 min_samples_leaf=3,
                                 random_state=19)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
print("RF Regressor: Training MSE: ")
mean_squared_error(y_train, y_train_pred).round(4)


y_val_pred = rf_model.predict(X_val)
print("RF Regressor: Validation MSE: ")
mean_squared_error(y_val, y_val_pred).round(4)

y_test_pred = rf_model.predict(X_test)
print("RF Regressor: Test MSE: ") 
mean_squared_error(y_test, y_test_pred).round(4)


error_list = []
depth_range= np.arange(1,21)
for d in depth_range:
    rf_model = RandomForestRegressor(n_estimators=200,
                                max_depth=d,
                                min_samples_leaf=3,
                                random_state=19)
    rf_model.fit(X_train, y_train)
    y_val_pred = rf_model.predict(X_val)
    error_list.append(mean_squared_error(y_val, y_val_pred))


plt.plot(depth_range, error_list)
plt.xlabel("Max Depth of Trees")
plt.ylabel("MSE of Validation")
plt.ylim([0., 0.03])
plt.show()


print("Best max depth: ")
print(depth_range[np.argmin(error_list)])
# I think any value between 10 and 20 gives results close... I will go with lowest (complexity): 10.

error_list = []
min_sample_leaves = np.arange(1,21)
for msl in min_sample_leaves:
    rf_model = RandomForestRegressor(n_estimators=200,
                                max_depth=10,
                                min_samples_leaf=msl,
                                random_state=19)
    rf_model.fit(X_train, y_train)
    y_val_pred = rf_model.predict(X_val)
    error_list.append(mean_squared_error(y_val, y_val_pred))


plt.plot(min_sample_leaves, error_list)
plt.xlabel("Min Samples of Leaf")
plt.ylabel("MSE of Validation")
# plt.ylim([0., 0.03])
plt.show()

print("Best min_samples_leaf == 1")

train_error = []
val_error = []
n_estimators = np.arange(30,451,30)
for N in n_estimators:
    rf_model = RandomForestRegressor(n_estimators=N,
                                max_depth=10,
                                min_samples_leaf=1,
                                random_state=19)
    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    train_error.append(mean_squared_error(y_train, y_train_pred))
    y_val_pred = rf_model.predict(X_val)
    val_error.append(mean_squared_error(y_val, y_val_pred))


# Lets see again when we start to overfit...
# plt.plot(n_estimators, train_error, color='g', label='Train')
plt.plot(n_estimators, val_error, color='r', label='Validation')
plt.xlabel("Number of estimators")
plt.ylabel("MSE")
plt.legend()
plt.show()

print("Best number of trees to form the Random Forest:")
print(n_estimators[np.argmin(val_error)])

rf_model = RandomForestRegressor(n_estimators=270,
                            max_depth=10,
                            min_samples_leaf=1,
                            random_state=19)
rf_model.fit(X_full_train, y_full_train)
y_full_train_pred = rf_model.predict(X_full_train)
print("RF Full Train MSE :")
print(mean_squared_error(y_full_train, y_full_train_pred).round(4))

y_test_pred = rf_model.predict(X_test)
print("RF Test MSE :")
print(mean_squared_error(y_test, y_test_pred).round(4))


rf_model.get_params()

print("Saving the RandomForestRegressor model...")

with open("bin/sklearn_rf_brazil_house.bin", 'wb') as f:
    pickle.dump(rf_model, f)

print("Saved successfully, all training is complete!")