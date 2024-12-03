import pandas as pd
import numpy as np
from sklearn import preprocessing
# Step1.資料的預處理(Preprocessing data)
# Step1-1 開啟檔案
train_data = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
    
print("\n\n訓練集資料：\n\n",train_data)
print("\n\n測試集資料：\n\n",test_data)

# Step1-2 觀察資料
train_data.info()

train_data.describe()

train_data.shape

train_data.columns
features = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for i in range(len(features)):
    print("第",i,"行不重複的元素數量：", len(train_data[features[i]].unique()))
    if (len(train_data[features[i]].unique()) > 100):
        print("捨去第",i,"個特徵：",features[i])


# Step1-3 處理資料的NAN
# 檢查每一列中的NaN值
print("各列的NaN值數量：")
print(train_data.isnull().sum())

# 顯示包含NaN值的具體行
print("\n包含NaN值的數據：")
print(train_data[train_data.isnull().any(axis=1)])

# 計算每列NaN值的百分比
print("\n各列NaN值的百分比：")
print((train_data.isnull().sum() / len(train_data)) * 100)


train_data['Facilities_cost'] = train_data['RoomService'] + train_data['FoodCourt'] + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck']


abandon_features = ['PassengerId','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Name']
#Cabin取side,p/s
train_data = train_data.drop(abandon_features,axis=1)

train_data['VIP']=np.where(train_data['CryoSleep']==0,1,0).astype(bool)
train_data['CryoSleep'] = train_data['CryoSleep'].fillna(False).astype(bool)
train_data['Facilities_cost'] = train_data['Facilities_cost'].fillna(1495)

# 移除包含缺失值的列
train_data = train_data.dropna(subset=['HomePlanet', 'Cabin', 'Destination'],axis=0)


print("\n移除缺失值後的訓練資料集：")
print(train_data.head())
print("\n資料集大小：")
print(train_data.shape)


# Step1-4 特徵挑選及資料正規化與編碼
# HomePlanet 使用 one-hot encoding
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
HomePlanet_onehot = onehot_encoder.fit_transform(train_data[['HomePlanet']])
HomePlanet_onehot_df = pd.DataFrame(HomePlanet_onehot, columns=onehot_encoder.get_feature_names_out(['HomePlanet']))

print(train_data.index)
print(HomePlanet_onehot_df.index)
#index重置對齊
train_data.reset_index(drop=True, inplace=True)
HomePlanet_onehot_df.reset_index(drop=True, inplace=True)

# 將原本的 HomePlanet 欄位移除,並加入 one-hot encoding 後的欄位
train_data = train_data.drop('HomePlanet', axis=1)
train_data = pd.concat([train_data, HomePlanet_onehot_df], axis=1)

print("\n進行 one-hot encoding 後的資料集：")
print(train_data.head())

# Destination 使用 one-hot encoding
Destination_onehot = onehot_encoder.fit_transform(train_data[['Destination']])
Destination_onehot_df = pd.DataFrame(Destination_onehot, columns=onehot_encoder.get_feature_names_out(['Destination']))

#index重置對齊
train_data.reset_index(drop=True, inplace=True)
Destination_onehot_df.reset_index(drop=True, inplace=True)

# 將原本的 Destination 欄位移除,並加入 one-hot encoding 後的欄位
train_data = train_data.drop('Destination', axis=1)
train_data = pd.concat([train_data, Destination_onehot_df], axis=1)

print("\n對 Destination 進行 one-hot encoding 後的資料集：")
print(train_data.head())

# CryoSleep 使用 label encoding
label_encoder = preprocessing.LabelEncoder()
train_data['CryoSleep'] = label_encoder.fit_transform(train_data['CryoSleep'])

print("\n對 CryoSleep 進行 label encoding 後的資料集：")
print(train_data.head())

# 使用 '/' 將 Cabin 欄位分為三欄
train_data[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = train_data['Cabin'].str.split('/', expand=True)

# 對 Cabin_Deck 進行 Label Encoding
train_data['Cabin_Deck'] = label_encoder.fit_transform(train_data['Cabin_Deck'])

# 對 Cabin_Side 進行 Label Encoding
train_data['Cabin_Side'] = label_encoder.fit_transform(train_data['Cabin_Side'])

train_data = train_data.drop('Cabin',axis=1)

print("\n對 Cabin_Deck 和 Cabin_Side 進行 Label Encoding 後的資料集：")
print(train_data.head())

# VIP 缺失值補False
train_data['VIP'] = train_data['VIP'].fillna(False)

# VIP 使用 label encoding
train_data['VIP'] = label_encoder.fit_transform(train_data['VIP'])

print("\n對 VIP 進行 label encoding 後的資料集：")
print(train_data.head())

# Age 缺失值補29歲
train_data['Age'] = train_data['Age'].fillna(29)

print("\n對 Age 補值後的資料集：")
print(train_data.head())

# Transported 使用 label encoding
train_data['Transported'] = label_encoder.fit_transform(train_data['Transported'])

print("\n對 Transported 進行 label encoding 後的資料集：")
print(train_data.head())

#Step1-5 分成特徵與標籤
train_data_feature = train_data.drop('Transported',axis=1)
train_data_label = train_data['Transported']


# Step2.模型選擇與建立(Model choose and build)     
from sklearn.model_selection import train_test_split

train_feature, val_feature, train_label, val_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

#1.SVM
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(train_feature, train_label)     

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("支持向量機(Support Vector Machines)模型準確度(訓練集):",svm_model.score(train_feature, train_label))
print ("支持向量機(Support Vector Machines)模型準確度(測試集):",svm_model.score(val_feature, val_label))
svm_model_acc = svm_model.score(val_feature, val_label)


#2.KNN
from sklearn.neighbors import KNeighborsClassifier

KNeighbors_model = KNeighborsClassifier(n_neighbors=2)
KNeighbors_model.fit(train_feature, train_label)

print ("最近的鄰居(Nearest Neighbors)模型準確度(訓練集)：",KNeighbors_model.score(train_feature, train_label))
print ("最近的鄰居(Nearest Neighbors)模型準確度(測試集)：",KNeighbors_model.score(val_feature, val_label))
KNeighbors_model_acc = KNeighbors_model.score(val_feature, val_label)

#3.Decision Tree
from sklearn import tree

DecisionTree_model = tree.DecisionTreeClassifier()
DecisionTree_model.fit(train_feature, train_label)

print ("決策樹(Decision Trees)模型準確度(訓練集)：",DecisionTree_model.score(train_feature, train_label))
print ("決策樹(Decision Trees)模型準確度(測試集)：",DecisionTree_model.score(val_feature, val_label))
DecisionTree_model_acc = DecisionTree_model.score(val_feature, val_label)

#4.Random Forest
from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=10)
RandomForest_model.fit(train_feature, train_label)

print ("隨機森林(Forests of randomized trees)模型準確度(訓練集)：",RandomForest_model.score(train_feature, train_label))
print ("隨機森林(Forests of randomized trees)模型準確度(測試集)：",RandomForest_model.score(val_feature, val_label))
RandomForest_model_model_acc = RandomForest_model.score(val_feature, val_label)


#5
from sklearn.neural_network import MLPClassifier

MLP_model = MLPClassifier(solver='lbfgs', 
                                   alpha=1e-4,
                                   hidden_layer_sizes=(6, 2), 
                                   )
MLP_model.fit(train_feature, train_label)

print ("神經網路(Neural Network models)模型準確度(訓練集)：",MLP_model.score(train_feature, train_label))
print ("神經網路(Neural Network models)模型準確度(測試集)：",MLP_model.score(val_feature, val_label))
MLP_model_acc = MLP_model.score(val_feature, val_label)

#6.Gaussian Process
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

GaussianProcess_model = GaussianProcessClassifier()
GaussianProcess_model.fit(train_feature, train_label)

print ("高斯過程(GaussianProcess)模型準確度(訓練集)：",GaussianProcess_model.score(train_feature, train_label))
print ("高斯過程(GaussianProcess)模型準確度(測試集)：",GaussianProcess_model.score(val_feature, val_label))
GaussianProcess_model_acc = GaussianProcess_model.score(val_feature, val_label)

#7.AdaBoost
from sklearn.ensemble import AdaBoostClassifier

AdaBoost_model = AdaBoostClassifier(n_estimators=10)
AdaBoost_model.fit(train_feature, train_label)

print ("AdaBoost模型準確度(訓練集)：",AdaBoost_model.score(train_feature, train_label))
print ("AdaBoost模型準確度(測試集)：",AdaBoost_model.score(val_feature, val_label))
AdaBoost_model_acc = AdaBoost_model.score(val_feature, val_label)

#8.Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

GradientBoosting_model = GradientBoostingClassifier(n_estimators=10)
GradientBoosting_model.fit(train_feature, train_label)

print ("Gradient Boosting模型準確度(訓練集)：",GradientBoosting_model.score(train_feature, train_label))
print ("Gradient Boosting模型準確度(測試集)：",GradientBoosting_model.score(val_feature, val_label))
GradientBoosting_model_acc = GradientBoosting_model.score(val_feature, val_label)

#9.Extra Trees
from sklearn.ensemble import ExtraTreesClassifier

ExtraTrees_model = ExtraTreesClassifier(n_estimators=10)
ExtraTrees_model.fit(train_feature, train_label)

print ("Extra Trees模型準確度(訓練集)：",ExtraTrees_model.score(train_feature, train_label))
print ("Extra Trees模型準確度(測試集)：",ExtraTrees_model.score(val_feature, val_label))
ExtraTrees_model_acc = ExtraTrees_model.score(val_feature, val_label)   

#10.Bagging
from sklearn.ensemble import BaggingClassifier

Bagging_model = BaggingClassifier(n_estimators=10)
Bagging_model.fit(train_feature, train_label)

print ("Bagging模型準確度(訓練集)：",Bagging_model.score(train_feature, train_label))
print ("Bagging模型準確度(測試集)：",Bagging_model.score(val_feature, val_label))
Bagging_model_acc = Bagging_model.score(val_feature, val_label) 

#11.Extra Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

ExtraGradientBoosting_model = GradientBoostingClassifier(n_estimators=10)
ExtraGradientBoosting_model.fit(train_feature, train_label)

print ("Extra Gradient Boosting模型準確度(訓練集)：",ExtraGradientBoosting_model.score(train_feature, train_label))
print ("Extra Gradient Boosting模型準確度(測試集)：",ExtraGradientBoosting_model.score(val_feature, val_label))
ExtraGradientBoosting_model_acc = ExtraGradientBoosting_model.score(val_feature, val_label)

#12 LightGBM

import lightgbm as lgb

LGBMClassifier_model = lgb.LGBMClassifier(objective='binary', n_estimators=120, learning_rate=0.02)

LGBMClassifier_model.fit(train_feature, train_label)

print ("Extra Gradient Boosting模型準確度(訓練集)：",LGBMClassifier_model.score(train_feature, train_label))
print ("Extra Gradient Boosting模型準確度(測試集)：",LGBMClassifier_model.score(val_feature, val_label))
LGBMClassifier_model_acc = LGBMClassifier_model.score(val_feature, val_label)


#13 XGBoost
from xgboost import XGBClassifier

XGBClassifier_model = XGBClassifier( n_estimators=120, learning_rate=0.001)

XGBClassifier_model.fit(train_feature, train_label)

print ("Extra Gradient Boosting模型準確度(訓練集)：",XGBClassifier_model.score(train_feature, train_label))
print ("Extra Gradient Boosting模型準確度(測試集)：",XGBClassifier_model.score(val_feature, val_label))
XGBClassifier_model_acc = XGBClassifier_model.score(val_feature, val_label)



#model比較
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 
              'Nearest Neighbors', 
              'Decision Trees',
              'Forests of randomized trees', 
              'Neural Network models',
              'GaussianProcess',
              'AdaBoost',
              'Gradient Boosting',
              'Extra Trees',
              'Bagging',
              'Extra Gradient Boosting',
              'LightGBM',
              'XGBoost',
             ],
    'Score': [svm_model_acc,
              KNeighbors_model_acc,
              DecisionTree_model_acc,
              RandomForest_model_model_acc,
              MLP_model_acc,
              GaussianProcess_model_acc,
              AdaBoost_model_acc,
              GradientBoosting_model_acc,
              ExtraTrees_model_acc,
              Bagging_model_acc,
              ExtraGradientBoosting_model_acc,
              LGBMClassifier_model_acc,
              XGBClassifier_model_acc,
              ]
                       })
models.sort_values(by='Score', ascending=False)



#14.Voting
from sklearn.ensemble import VotingClassifier   

VotingClassifier_model = VotingClassifier(estimators=[('ada', AdaBoost_model), ('gb', GradientBoosting_model), ('lgb', LGBMClassifier_model), ('xgb', XGBClassifier_model), ('egb', ExtraGradientBoosting_model)], voting='hard')
VotingClassifier_model.fit(train_feature, train_label)

print ("Voting模型準確度(訓練集)：",VotingClassifier_model.score(train_feature, train_label))
print ("Voting模型準確度(測試集)：",VotingClassifier_model.score(val_feature, val_label))
VotingClassifier_model_acc = VotingClassifier_model.score(val_feature, val_label)



#15.deep learning
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0006), metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=40, min_delta=0.00001, restore_best_weights=True, verbose=1)

history = model.fit(train_feature, train_label, epochs=300, batch_size=32, validation_data=(val_feature, val_label), callbacks=[early_stopping, reduce_lr])

# 評估模型
train_acc = model.evaluate(train_feature, train_label, verbose=0)[1]
deep_learning_model_acc = model.evaluate(val_feature, val_label, verbose=0)[1]

print(f"模型準確度 (訓練集): {train_acc:.4f}")
print(f"模型準確度 (測試集): {deep_learning_model_acc:.4f}")

import matplotlib.pyplot as plt

# 繪製訓練損失和準確率
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.show()




# STep3.模型驗證(Model validation)

test_data.info()

test_data.describe()

abandon_features = ['PassengerId','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Name']

test_data_features = test_data.drop(abandon_features,axis=1)

# 填補包含缺失值的列
test_data_features['Destination'] = test_data_features['Destination'].fillna('TRAPPIST-1e')
test_data_features['HomePlanet'] = test_data_features['HomePlanet'].fillna('Earth')
test_data_features['CryoSleep'] = test_data_features['CryoSleep'].fillna(False).astype(bool)
test_data_features['Cabin'] = test_data_features['Cabin'].fillna('F/82/S')
test_data_features['Facilities_cost'] = test_data_features['Facilities_cost'].fillna(1495)
test_data_features['VIP']=np.where(test_data_features['CryoSleep']==0,1,0).astype(bool)


# Step1-4 特徵挑選及資料正規化與編碼
# HomePlanet 使用 one-hot encoding
onehot_encoder = preprocessing.OneHotEncoder(sparse_output=False)
HomePlanet_onehot = onehot_encoder.fit_transform(test_data_features[['HomePlanet']])
HomePlanet_onehot_df = pd.DataFrame(HomePlanet_onehot, columns=onehot_encoder.get_feature_names_out(['HomePlanet']))

#index重置對齊
test_data_features.reset_index(drop=True, inplace=True)
HomePlanet_onehot_df.reset_index(drop=True, inplace=True)

# 將原本的 HomePlanet 欄位移除,並加入 one-hot encoding 後的欄位
test_data_features = test_data_features.drop('HomePlanet', axis=1)
test_data_features = pd.concat([test_data_features, HomePlanet_onehot_df], axis=1)

# Destination 使用 one-hot encoding
Destination_onehot = onehot_encoder.fit_transform(test_data_features[['Destination']])
Destination_onehot_df = pd.DataFrame(Destination_onehot, columns=onehot_encoder.get_feature_names_out(['Destination']))

#index重置對齊
test_data_features.reset_index(drop=True, inplace=True)
Destination_onehot_df.reset_index(drop=True, inplace=True)

# 將原本的 Destination 欄位移除,並加入 one-hot encoding 後的欄位
test_data_features = test_data_features.drop('Destination', axis=1)
test_data_features = pd.concat([test_data_features, Destination_onehot_df], axis=1)

# CryoSleep 使用 label encoding
label_encoder = preprocessing.LabelEncoder()
test_data_features['CryoSleep'] = label_encoder.fit_transform(test_data_features['CryoSleep'])

# 使用 '/' 將 Cabin 欄位分為三欄
test_data_features[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = test_data_features['Cabin'].str.split('/', expand=True)

# 對 Cabin_Deck 進行 Label Encoding
test_data_features['Cabin_Deck'] = label_encoder.fit_transform(test_data_features['Cabin_Deck'])

# 對 Cabin_Side 進行 Label Encoding
test_data_features['Cabin_Side'] = label_encoder.fit_transform(test_data_features['Cabin_Side'])

test_data_features = test_data_features.drop('Cabin',axis=1)
test_data_features = test_data_features.drop('Cabin_Number',axis=1)

# VIP 缺失值補False
test_data_features['VIP'] = test_data_features['VIP'].fillna(False).astype(bool)

# VIP 使用 label encoding
test_data_features['VIP'] = label_encoder.fit_transform(test_data_features['VIP'])

# Age 缺失值補29歲
test_data_features['Age'] = test_data_features['Age'].fillna(29)



test_data_features

#Predict

final_predictions = VotingClassifier_model.predict(test_data_features)
final_predictions = final_predictions.astype(bool)
output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Transported': final_predictions})
output.to_csv('Submission1.csv', index=False)

final_predictions2 = model.predict(test_data_features)
final_predictions2 = final_predictions2.astype(bool)
output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Transported': final_predictions})
output.to_csv('Submission2.csv', index=False)
