import pandas as pd
import sklearn
import tensorflow
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf

def load_housing_data(housing_path):
    return pd.read_csv(housing_path)

housing = load_housing_data('train.csv')
Housing_test = load_housing_data('test.csv')
housing.info()

corr_matrix = housing.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)

Housing_Y = housing.iloc[:,-1]
Housing_X = housing.drop("SalePrice",axis = 1)

Housing_X.drop(['ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','MSSubClass','EnclosedPorch'],axis =1,inplace=True)
Housing_X.drop(['Alley','SaleType','GarageCond','GarageFinish','Functional'],axis = 1,inplace =True)
Housing_X.drop(['GarageQual','HeatingQC','Heating','BsmtFinType2','BsmtFinType2','BsmtExposure','BsmtCond','Foundation','MasVnrType','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','Condition2','Condition1','LotConfig','LandContour'],axis=1,inplace=True)
Housing_X.drop(['FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

           


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(Housing_X.iloc[:, 1:3])
Housing_X.iloc[:, 1:3] = imputer.transform(Housing_X.iloc[:, 1:3])

imputer = imputer.fit(Housing_X.iloc[:, 10:15])
Housing_X.iloc[:, 10:15] = imputer.transform(Housing_X.iloc[:, 10:15])

imputer = imputer.fit(Housing_X.iloc[:, 19:22])
Housing_X.iloc[:, 19:22] = imputer.transform(Housing_X.iloc[:, 19:22])

imputer = imputer.fit(Housing_X.iloc[:, 24:32])
Housing_X.iloc[:, 24:32] = imputer.transform(Housing_X.iloc[:, 24:32])

imputer = imputer.fit(Housing_X.iloc[:, 33:35])
Housing_X.iloc[:, 33:35] = imputer.transform(Housing_X.iloc[:, 33:35])

imputer = imputer.fit(Housing_X.iloc[:, 36:39])
Housing_X.iloc[:, 36:39] = imputer.transform(Housing_X.iloc[:, 36:39])

imputer = imputer.fit(Housing_X.iloc[:, 40:42])
Housing_X.iloc[:, 40:42] = imputer.transform(Housing_X.iloc[:, 40:42])

Housing_X['BsmtQual'] = Housing_X['BsmtQual'].replace(['Ex','Gd','TA','Fa','Po','NA'],[100,95,85,75,65,0])
Housing_X.loc[Housing_X['BsmtQual'].isnull(), 'BsmtQual'] = 0
Housing_X.loc[Housing_X['GarageType'].isnull(), 'GarageType'] = 'NA'
Housing_X.loc[Housing_X['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'NA'
Housing_X.loc[Housing_X['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

ordinal_encoder = OrdinalEncoder()
Housing_X.iloc[:,0:1] = ordinal_encoder.fit_transform(Housing_X.iloc[:,0:1])
Housing_X.iloc[:,3:10] = ordinal_encoder.fit_transform(Housing_X.iloc[:,3:10])
Housing_X.iloc[:,15:19] = ordinal_encoder.fit_transform(Housing_X.iloc[:,15:19])
Housing_X.iloc[:,22:24] = ordinal_encoder.fit_transform(Housing_X.iloc[:,22:24])
Housing_X.iloc[:,32:33] = ordinal_encoder.fit_transform(Housing_X.iloc[:,32:33])
Housing_X.iloc[:,35:36] = ordinal_encoder.fit_transform(Housing_X.iloc[:,35:36])
Housing_X.iloc[:,39:40] = ordinal_encoder.fit_transform(Housing_X.iloc[:,39:40])
Housing_X.iloc[:,42:43] = ordinal_encoder.fit_transform(Housing_X.iloc[:,42:43])

cat_encoder = OneHotEncoder()
cat_col_1 = Housing_X['MSZoning']
cat_col_1 = cat_col_1.to_frame()
cat_col_1 = cat_encoder.fit_transform(cat_col_1)
cat_col_1 = pd.DataFrame(cat_col_1.toarray())
cat_col_1.drop(0,axis = 1,inplace=True)


cat_col_2 = Housing_X['LotShape']
cat_col_2 = cat_col_2.to_frame()
cat_col_2 = cat_encoder.fit_transform(cat_col_2)
cat_col_2 = pd.DataFrame(cat_col_2.toarray())
cat_col_2.drop(0,axis = 1,inplace=True)


cat_col_3 = Housing_X['LandSlope']
cat_col_3 = cat_col_3.to_frame()
cat_col_3 = cat_encoder.fit_transform(cat_col_3)
cat_col_3 = pd.DataFrame(cat_col_3.toarray())
cat_col_3.drop(0,axis = 1,inplace=True)

cat_col_4 = Housing_X['Neighborhood']
cat_col_4 = cat_col_4.to_frame()
cat_col_4 = cat_encoder.fit_transform(cat_col_4)
cat_col_4 = pd.DataFrame(cat_col_4.toarray())
cat_col_4.drop(0,axis = 1,inplace=True)

cat_col_5 = Housing_X['BldgType']
cat_col_5 = cat_col_5.to_frame()
cat_col_5 = cat_encoder.fit_transform(cat_col_5)
cat_col_5 = pd.DataFrame(cat_col_5.toarray())
cat_col_5.drop(0,axis = 1,inplace=True)

cat_col_6 = Housing_X['ExterQual']
cat_col_6 = cat_col_6.to_frame()
cat_col_6 = cat_encoder.fit_transform(cat_col_6)
cat_col_6 = pd.DataFrame(cat_col_6.toarray())
cat_col_6.drop(0,axis = 1,inplace=True)

cat_col_7 = Housing_X['CentralAir']
cat_col_7 = cat_col_7.to_frame()
cat_col_7 = cat_encoder.fit_transform(cat_col_7)
cat_col_7 = pd.DataFrame(cat_col_7.toarray())
cat_col_7.drop(0,axis = 1,inplace=True)

cat_col_8 = Housing_X['KitchenQual']
cat_col_8 = cat_col_8.to_frame()
cat_col_8 = cat_encoder.fit_transform(cat_col_8)
cat_col_8 = pd.DataFrame(cat_col_8.toarray())
cat_col_8.drop(0,axis = 1,inplace=True)

cat_col_9 = Housing_X['GarageType']
cat_col_9 = cat_col_9.to_frame()
cat_col_9 = cat_encoder.fit_transform(cat_col_9)
cat_col_9 = pd.DataFrame(cat_col_9.toarray())
cat_col_9.drop(0,axis = 1,inplace=True)

cat_col_10 = Housing_X['PavedDrive']
cat_col_10 = cat_col_10.to_frame()
cat_col_10 = cat_encoder.fit_transform(cat_col_10)
cat_col_10 = pd.DataFrame(cat_col_10.toarray())
cat_col_10.drop(0,axis = 1,inplace=True)

cat_col_11 = Housing_X['SaleCondition']
cat_col_11 = cat_col_11.to_frame()
cat_col_11 = cat_encoder.fit_transform(cat_col_11)
cat_col_11 = pd.DataFrame(cat_col_11.toarray())
cat_col_11.drop(0,axis = 1,inplace=True)

cat_col_12 = Housing_X['Street']
cat_col_12 = cat_col_12.to_frame()
cat_col_12 = cat_encoder.fit_transform(cat_col_12)
cat_col_12 = pd.DataFrame(cat_col_12.toarray())
cat_col_12.drop(0,axis = 1,inplace=True)

cat_col_13 = Housing_X['Utilities']
cat_col_13 = cat_col_13.to_frame()
cat_col_13 = cat_encoder.fit_transform(cat_col_13)
cat_col_13 = pd.DataFrame(cat_col_13.toarray())
cat_col_13.drop([0,1],axis = 1,inplace=True)

cat_col_14 = Housing_X['HouseStyle']
cat_col_14 = cat_col_14.to_frame()
cat_col_14 = cat_encoder.fit_transform(cat_col_14)
cat_col_14 = pd.DataFrame(cat_col_14.toarray())
cat_col_14.drop([0,1],axis = 1,inplace=True)

cat_col_15 = Housing_X['ExterCond']
cat_col_15 = cat_col_15.to_frame()
cat_col_15 = cat_encoder.fit_transform(cat_col_15)
cat_col_15 = pd.DataFrame(cat_col_15.toarray())
cat_col_15.drop(0,axis = 1,inplace=True)

cat_col_16 = Housing_X['BsmtQual']
cat_col_16 = cat_col_16.to_frame()
cat_col_16 = cat_encoder.fit_transform(cat_col_16)
cat_col_16 = pd.DataFrame(cat_col_16.toarray())
cat_col_16.drop(0,axis = 1,inplace=True)


cat_col_17 = Housing_X['BsmtFinType1']
cat_col_17 = cat_col_17.to_frame()
cat_col_17 = cat_encoder.fit_transform(cat_col_17)
cat_col_17 = pd.DataFrame(cat_col_17.toarray())
cat_col_17.drop(0,axis = 1,inplace=True)


cat_col_18 = Housing_X['Electrical']
cat_col_18 = cat_col_18.to_frame()
cat_col_18 = cat_encoder.fit_transform(cat_col_18)
cat_col_18 = pd.DataFrame(cat_col_18.toarray())
cat_col_18.drop([0,1],axis = 1,inplace=True)

X_Train = pd.concat([cat_col_1,cat_col_2,cat_col_3,cat_col_4,cat_col_5,cat_col_6,cat_col_7,cat_col_8,cat_col_9,cat_col_10,cat_col_11,cat_col_12,cat_col_13,cat_col_14,cat_col_15,cat_col_16,cat_col_17,cat_col_18,Housing_X.iloc[:, 1:3],Housing_X.iloc[:, 10:15],Housing_X.iloc[:, 19:22],Housing_X.iloc[:, 24:32],Housing_X.iloc[:,33:35],Housing_X.iloc[:, 36:39],Housing_X.iloc[:, 40:42]],axis=1)
X_S = MinMaxScaler()
Y_S = MinMaxScaler()
X_Train = X_S.fit_transform(X_Train)
Housing_Y = Housing_Y.to_frame()
Housing_Y = Y_S.fit_transform(Housing_Y)

X_Train_F = X_Train[:,80:106]

N,D = X_Train_F.shape
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(D,)),
  tf.keras.layers.Dense(4000,activation ='relu'),
  tf.keras.layers.Dense(4000,activation ='relu'),
  tf.keras.layers.Dense(3000,activation ='relu'),
  tf.keras.layers.Dense(3000,activation ='relu'),
  tf.keras.layers.Dense(2000,activation ='relu'),
  tf.keras.layers.Dense(2000,activation ='relu'),
  tf.keras.layers.Dense(1000,activation ='relu'),
  tf.keras.layers.Dense(768,activation ='relu'),
  tf.keras.layers.Dense(512,activation ='relu'),
  tf.keras.layers.Dense(256,activation ='relu'),
  tf.keras.layers.Dense(1, activation='relu')
])
from keras import optimizers

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['mse','mae'])



r = model.fit(X_Train_F,Housing_Y,epochs=2000,verbose=1)

Housing_test.drop(['ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','MSSubClass','EnclosedPorch'],axis =1,inplace=True)
Housing_test.drop(['Alley','SaleType','GarageCond','GarageFinish','Functional'],axis = 1,inplace =True)
Housing_test.drop(['GarageQual','HeatingQC','Heating','BsmtFinType2','BsmtFinType2','BsmtExposure','BsmtCond','Foundation','MasVnrType','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','Condition2','Condition1','LotConfig','LandContour'],axis=1,inplace=True)
Housing_test.drop(['FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)       


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(Housing_test.iloc[:, 1:3])
Housing_test.iloc[:, 1:3] = imputer.transform(Housing_test.iloc[:, 1:3])

imputer = imputer.fit(Housing_test.iloc[:, 10:15])
Housing_test.iloc[:, 10:15] = imputer.transform(Housing_test.iloc[:, 10:15])

imputer = imputer.fit(Housing_test.iloc[:, 19:22])
Housing_test.iloc[:, 19:22] = imputer.transform(Housing_test.iloc[:, 19:22])

imputer = imputer.fit(Housing_test.iloc[:, 24:32])
Housing_test.iloc[:, 24:32] = imputer.transform(Housing_test.iloc[:, 24:32])

imputer = imputer.fit(Housing_test.iloc[:, 33:35])
Housing_test.iloc[:, 33:35] = imputer.transform(Housing_test.iloc[:, 33:35])

imputer = imputer.fit(Housing_test.iloc[:, 36:39])
Housing_test.iloc[:, 36:39] = imputer.transform(Housing_test.iloc[:, 36:39])

imputer = imputer.fit(Housing_test.iloc[:, 40:42])
Housing_test.iloc[:, 40:42] = imputer.transform(Housing_test.iloc[:, 40:42])



Housing_test['BsmtQual'] = Housing_test['BsmtQual'].replace(['Ex','Gd','TA','Fa','Po','NA'],[100,95,85,75,65,0])
Housing_test.loc[Housing_test['BsmtQual'].isnull(), 'BsmtQual'] = 0
Housing_test.loc[Housing_test['GarageType'].isnull(), 'GarageType'] = 'NA'
Housing_test.loc[Housing_test['MSZoning'].isnull(), 'MSZoning'] = 'RL'
Housing_test.loc[Housing_test['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'
Housing_test.loc[Housing_test['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'NA'
Housing_test.loc[Housing_test['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
Housing_test.loc[Housing_test['Utilities'].isnull(), 'Utilities'] = 'AllPub'


ordinal_encoder = OrdinalEncoder()
Housing_test.iloc[:,0:1] = ordinal_encoder.fit_transform(Housing_test.iloc[:,0:1])
Housing_test.iloc[:,3:10] = ordinal_encoder.fit_transform(Housing_test.iloc[:,3:10])
Housing_test.iloc[:,15:19] = ordinal_encoder.fit_transform(Housing_test.iloc[:,15:19])
Housing_test.iloc[:,22:24] = ordinal_encoder.fit_transform(Housing_test.iloc[:,22:24])
Housing_test.iloc[:,32:33] = ordinal_encoder.fit_transform(Housing_test.iloc[:,32:33])
Housing_test.iloc[:,35:36] = ordinal_encoder.fit_transform(Housing_test.iloc[:,35:36])
Housing_test.iloc[:,39:40] = ordinal_encoder.fit_transform(Housing_test.iloc[:,39:40])
Housing_test.iloc[:,42:43] = ordinal_encoder.fit_transform(Housing_test.iloc[:,42:43])

cat_encoder = OneHotEncoder()
cat_col_1_t = Housing_test['MSZoning']
cat_col_1_t = cat_col_1_t.to_frame()
cat_col_1_t = cat_encoder.fit_transform(cat_col_1_t)
cat_col_1_t = pd.DataFrame(cat_col_1_t.toarray())
cat_col_1_t.drop(0,axis = 1,inplace=True)


cat_col_2_t = Housing_test['LotShape']
cat_col_2_t = cat_col_2_t.to_frame()
cat_col_2_t = cat_encoder.fit_transform(cat_col_2_t)
cat_col_2_t = pd.DataFrame(cat_col_2_t.toarray())
cat_col_2_t.drop(0,axis = 1,inplace=True)


cat_col_3_t = Housing_test['LandSlope']
cat_col_3_t = cat_col_3_t.to_frame()
cat_col_3_t = cat_encoder.fit_transform(cat_col_3_t)
cat_col_3_t = pd.DataFrame(cat_col_3_t.toarray())
cat_col_3_t.drop(0,axis = 1,inplace=True)

cat_col_4_t = Housing_test['Neighborhood']
cat_col_4_t = cat_col_4_t.to_frame()
cat_col_4_t = cat_encoder.fit_transform(cat_col_4_t)
cat_col_4_t = pd.DataFrame(cat_col_4_t.toarray())
cat_col_4_t.drop(0,axis = 1,inplace=True)

cat_col_5_t = Housing_test['BldgType']
cat_col_5_t = cat_col_5_t.to_frame()
cat_col_5_t = cat_encoder.fit_transform(cat_col_5_t)
cat_col_5_t = pd.DataFrame(cat_col_5_t.toarray())
cat_col_5_t.drop(0,axis = 1,inplace=True)

cat_col_6_t = Housing_test['ExterQual']
cat_col_6_t = cat_col_6_t.to_frame()
cat_col_6_t = cat_encoder.fit_transform(cat_col_6_t)
cat_col_6_t = pd.DataFrame(cat_col_6_t.toarray())
cat_col_6_t.drop(0,axis = 1,inplace=True)

cat_col_7_t = Housing_test['CentralAir']
cat_col_7_t = cat_col_7_t.to_frame()
cat_col_7_t = cat_encoder.fit_transform(cat_col_7_t)
cat_col_7_t = pd.DataFrame(cat_col_7_t.toarray())
cat_col_7_t.drop(0,axis = 1,inplace=True)

cat_col_8_t = Housing_test['KitchenQual']
cat_col_8_t = cat_col_8_t.to_frame()
cat_col_8_t = cat_encoder.fit_transform(cat_col_8_t)
cat_col_8_t = pd.DataFrame(cat_col_8_t.toarray())
cat_col_8_t.drop(0,axis = 1,inplace=True)

cat_col_9_t = Housing_test['GarageType']
cat_col_9_t = cat_col_9_t.to_frame()
cat_col_9_t = cat_encoder.fit_transform(cat_col_9_t)
cat_col_9_t = pd.DataFrame(cat_col_9_t.toarray())
cat_col_9_t.drop(0,axis = 1,inplace=True)

cat_col_10_t = Housing_test['PavedDrive']
cat_col_10_t = cat_col_10_t.to_frame()
cat_col_10_t = cat_encoder.fit_transform(cat_col_10_t)
cat_col_10_t = pd.DataFrame(cat_col_10_t.toarray())
cat_col_10_t.drop(0,axis = 1,inplace=True)

cat_col_11_t = Housing_test['SaleCondition']
cat_col_11_t = cat_col_11_t.to_frame()
cat_col_11_t = cat_encoder.fit_transform(cat_col_11_t)
cat_col_11_t = pd.DataFrame(cat_col_11_t.toarray())
cat_col_11_t.drop(0,axis = 1,inplace=True)

cat_col_12_t = Housing_test['Street']
cat_col_12_t = cat_col_12_t.to_frame()
cat_col_12_t = cat_encoder.fit_transform(cat_col_12_t)
cat_col_12_t = pd.DataFrame(cat_col_12_t.toarray())
cat_col_12_t.drop(0,axis = 1,inplace=True)

cat_col_13_t = Housing_test['Utilities']
cat_col_13_t = cat_col_13_t.to_frame()
cat_col_13_t = cat_encoder.fit_transform(cat_col_13_t)
cat_col_13_t = pd.DataFrame(cat_col_13_t.toarray())
cat_col_13_t.drop(0,axis = 1,inplace=True)

cat_col_14_t = Housing_test['HouseStyle']
cat_col_14_t = cat_col_14_t.to_frame()
cat_col_14_t = cat_encoder.fit_transform(cat_col_14_t)
cat_col_14_t = pd.DataFrame(cat_col_14_t.toarray())
cat_col_14_t.drop(0,axis = 1,inplace=True)

cat_col_15_t = Housing_test['ExterCond']
cat_col_15_t = cat_col_15_t.to_frame()
cat_col_15_t = cat_encoder.fit_transform(cat_col_15_t)
cat_col_15_t = pd.DataFrame(cat_col_15_t.toarray())
cat_col_15_t.drop(0,axis = 1,inplace=True)

cat_col_16_t = Housing_test['BsmtQual']
cat_col_16_t = cat_col_16_t.to_frame()
cat_col_16_t = cat_encoder.fit_transform(cat_col_16_t)
cat_col_16_t = pd.DataFrame(cat_col_16_t.toarray())
cat_col_16_t.drop(0,axis = 1,inplace=True)


cat_col_17_t = Housing_test['BsmtFinType1']
cat_col_17_t = cat_col_17_t.to_frame()
cat_col_17_t = cat_encoder.fit_transform(cat_col_17_t)
cat_col_17_t = pd.DataFrame(cat_col_17_t.toarray())
cat_col_17_t.drop(0,axis = 1,inplace=True)


cat_col_18_t = Housing_test['Electrical']
cat_col_18_t = cat_col_18_t.to_frame()
cat_col_18_t = cat_encoder.fit_transform(cat_col_18_t)
cat_col_18_t = pd.DataFrame(cat_col_18_t.toarray())
cat_col_18_t.drop(0,axis = 1,inplace=True)

X_test = pd.concat([cat_col_1_t,cat_col_2_t,cat_col_3_t,cat_col_4_t,cat_col_5_t,cat_col_6_t,cat_col_7_t,cat_col_8_t,cat_col_9_t,cat_col_10_t,cat_col_11_t,cat_col_12_t,cat_col_13_t,cat_col_14_t,cat_col_15_t,cat_col_16_t,cat_col_17_t,cat_col_18_t,Housing_test.iloc[:, 1:3],Housing_test.iloc[:, 10:15],Housing_test.iloc[:, 19:22],Housing_test.iloc[:, 24:32],Housing_test.iloc[:, 33:35],Housing_test.iloc[:, 36:39],Housing_test.iloc[:, 40:42]],axis=1)
X_test = X_S.transform(X_test)
X_test_f = X_test[:,80:106]


y_pred= model.predict(X_test_f)

y_pred = Y_S.inverse_transform(y_pred)