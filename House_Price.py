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

Housing_X.drop(['BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr'],axis =1,inplace=True)
Housing_X.drop(['Alley','PoolQC','Fence','MiscFeature','SaleType','GarageCond','GarageFinish','Functional','Electrical'],axis = 1,inplace =True)
Housing_X.drop(['GarageQual','HeatingQC','Heating','BsmtFinType2','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','Foundation','ExterCond','MasVnrType','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','HouseStyle','Condition2','Condition1','LotConfig','Utilities','LandContour'],axis=1,inplace=True)
Housing_X.drop(['FireplaceQu','Street'],axis=1,inplace=True)            


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(Housing_X.iloc[:, 1:3])
Housing_X.iloc[:, 1:3] = imputer.transform(Housing_X.iloc[:, 1:3])

imputer = imputer.fit(Housing_X.iloc[:, 7:11])
Housing_X.iloc[:, 7:11] = imputer.transform(Housing_X.iloc[:, 7:11])

imputer = imputer.fit(Housing_X.iloc[:, 13:15])
Housing_X.iloc[:, 13:15] = imputer.transform(Housing_X.iloc[:, 13:15])

imputer = imputer.fit(Housing_X.iloc[:, 16:21])
Housing_X.iloc[:, 16:21] = imputer.transform(Housing_X.iloc[:, 16:21])

imputer = imputer.fit(Housing_X.iloc[:, 22:24])
Housing_X.iloc[:, 22:24] = imputer.transform(Housing_X.iloc[:, 22:24])

imputer = imputer.fit(Housing_X.iloc[:, 25:28])
Housing_X.iloc[:, 25:28] = imputer.transform(Housing_X.iloc[:, 25:28])

imputer = imputer.fit(Housing_X.iloc[:, 29:31])
Housing_X.iloc[:, 29:31] = imputer.transform(Housing_X.iloc[:, 29:31])

Housing_X['BsmtQual'] = Housing_X['BsmtQual'].replace(['Ex','Gd','TA','Fa','Po','NA'],[100,95,85,75,65,0])
Housing_X.loc[Housing_X['BsmtQual'].isnull(), 'BsmtQual'] = 0
Housing_X.loc[Housing_X['GarageType'].isnull(), 'GarageType'] = 'NA'

ordinal_encoder = OrdinalEncoder()
Housing_X.iloc[:,0:1] = ordinal_encoder.fit_transform(Housing_X.iloc[:,0:1])
Housing_X.iloc[:,3:7] = ordinal_encoder.fit_transform(Housing_X.iloc[:,3:7])
Housing_X.iloc[:,11:12] = ordinal_encoder.fit_transform(Housing_X.iloc[:,11:12])
Housing_X.iloc[:,15:16] = ordinal_encoder.fit_transform(Housing_X.iloc[:,15:16])
Housing_X.iloc[:,21:22] = ordinal_encoder.fit_transform(Housing_X.iloc[:,21:22])
Housing_X.iloc[:,24:25] = ordinal_encoder.fit_transform(Housing_X.iloc[:,24:25])
Housing_X.iloc[:,28:29] = ordinal_encoder.fit_transform(Housing_X.iloc[:,28:29])
Housing_X.iloc[:,31:32] = ordinal_encoder.fit_transform(Housing_X.iloc[:,31:32])

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

X_Train = pd.concat([cat_col_1,cat_col_2,cat_col_3,cat_col_4,cat_col_5,cat_col_6,cat_col_7,cat_col_8,cat_col_9,cat_col_10,cat_col_11,Housing_X.iloc[:, 1:3],Housing_X.iloc[:, 7:11],Housing_X.iloc[:, 13:15],Housing_X.iloc[:, 16:21],Housing_X.iloc[:, 22:24],Housing_X.iloc[:, 25:28],Housing_X.iloc[:, 29:31]],axis=1)
X_S = MinMaxScaler()
Y_S = MinMaxScaler()
X_Train = X_S.fit_transform(X_Train)
Housing_Y = Housing_Y.to_frame()
Housing_Y = Y_S.fit_transform(Housing_Y)

N,D = X_Train.shape
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(D,)),
  tf.keras.layers.Dense(4000,activation ='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01)),
  tf.keras.layers.Dense(4000,activation ='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01)),
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



r = model.fit(X_Train,Housing_Y,epochs=200,verbose=1)

Housing_test.drop(['BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr'],axis =1,inplace=True)
Housing_test.drop(['Alley','PoolQC','Fence','MiscFeature','SaleType','GarageCond','GarageFinish','Functional','Electrical'],axis = 1,inplace =True)
Housing_test.drop(['GarageQual','HeatingQC','Heating','BsmtFinType2','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','Foundation','ExterCond','MasVnrType','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','HouseStyle','Condition2','Condition1','LotConfig','Utilities','LandContour'],axis=1,inplace=True)
Housing_test.drop(['FireplaceQu','Street'],axis=1,inplace=True)            


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(Housing_test.iloc[:, 1:3])
Housing_test.iloc[:, 1:3] = imputer.transform(Housing_test.iloc[:, 1:3])

imputer = imputer.fit(Housing_test.iloc[:, 7:11])
Housing_test.iloc[:, 7:11] = imputer.transform(Housing_test.iloc[:, 7:11])

imputer = imputer.fit(Housing_test.iloc[:, 13:15])
Housing_test.iloc[:, 13:15] = imputer.transform(Housing_test.iloc[:, 13:15])

imputer = imputer.fit(Housing_test.iloc[:, 16:21])
Housing_test.iloc[:, 16:21] = imputer.transform(Housing_test.iloc[:, 16:21])

imputer = imputer.fit(Housing_test.iloc[:, 22:24])
Housing_test.iloc[:, 22:24] = imputer.transform(Housing_test.iloc[:, 22:24])

imputer = imputer.fit(Housing_test.iloc[:, 25:28])
Housing_test.iloc[:, 25:28] = imputer.transform(Housing_test.iloc[:, 25:28])

imputer = imputer.fit(Housing_test.iloc[:, 29:31])
Housing_test.iloc[:, 29:31] = imputer.transform(Housing_test.iloc[:, 29:31])



Housing_test['BsmtQual'] = Housing_test['BsmtQual'].replace(['Ex','Gd','TA','Fa','Po','NA'],[100,95,85,75,65,0])
Housing_test.loc[Housing_test['BsmtQual'].isnull(), 'BsmtQual'] = 0
Housing_test.loc[Housing_test['GarageType'].isnull(), 'GarageType'] = 'NA'
Housing_test.loc[Housing_test['MSZoning'].isnull(), 'MSZoning'] = 'RL'
Housing_test.loc[Housing_test['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'

ordinal_encoder = OrdinalEncoder()
Housing_test.iloc[:,0:1] = ordinal_encoder.fit_transform(Housing_test.iloc[:,0:1])
Housing_test.iloc[:,3:7] = ordinal_encoder.fit_transform(Housing_test.iloc[:,3:7])
Housing_test.iloc[:,11:12] = ordinal_encoder.fit_transform(Housing_test.iloc[:,11:12])
Housing_test.iloc[:,15:16] = ordinal_encoder.fit_transform(Housing_test.iloc[:,15:16])
Housing_test.iloc[:,21:22] = ordinal_encoder.fit_transform(Housing_test.iloc[:,21:22])
Housing_test.iloc[:,24:25] = ordinal_encoder.fit_transform(Housing_test.iloc[:,24:25])
Housing_test.iloc[:,28:29] = ordinal_encoder.fit_transform(Housing_test.iloc[:,28:29])
Housing_test.iloc[:,31:32] = ordinal_encoder.fit_transform(Housing_test.iloc[:,31:32])

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

X_test = pd.concat([cat_col_1_t,cat_col_2_t,cat_col_3_t,cat_col_4_t,cat_col_5_t,cat_col_6_t,cat_col_7_t,cat_col_8_t,cat_col_9_t,cat_col_10_t,cat_col_11_t,Housing_test.iloc[:, 1:3],Housing_test.iloc[:, 7:11],Housing_test.iloc[:, 13:15],Housing_test.iloc[:, 16:21],Housing_test.iloc[:, 22:24],Housing_test.iloc[:, 25:28],Housing_test.iloc[:, 29:31]],axis=1)
X_test = X_S.transform(X_test)


y_pred= model.predict(X_test)

y_pred = Y_S.inverse_transform(y_pred)