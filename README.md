전기차 가격 예측 해커톤 : 데이터로 EV를 읽다!🚙
===============================================================
---------------------------------------------------------------

### 개발환경
```
Jupyter Notebook
python
```

<hr/>

### 기본 라이브러리 불러오기
```
import numpy as np
import pandas as pd
```

### 데이터 로드 및 전처리
- 데이터 전처리
1. 학습 데이터와 예측 데이터 분리 : train 데이터에 포함된 '가격(백만원)' 데이터 분리
```
train = pd.read_csv('data/final_train.csv')
test = pd.read_csv('data/final_test.csv')

# 학습 및 예측 데이터 분리
x_train = train.drop('가격(백만원)', axis = 1).values
y_train = train['가격(백만원)'].values
y_train = y_train.round().astype(int) # 연속형 값을 반올림하여 정수형으로 변환

x_test = test
```

2. 학습 데이터 분할 : train 데이터를 훈련용 데이터(0.7)와 테스트 데이터(0.3)로 분할
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, train_size = 0.7,
                                                   random_state = 1, stratify = None)
```

3. 표준화
- 학습 데이터 기준 표준화 진행
```
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

### 1. 로지스틱 회귀 초모수 튜닝
- LogisticRegression : 일반화 선형 모델. 독립 변수의 선형 조합을 로지스틱 함수를 사용하여 종속 변수에 대한 확률 점수로 변환.
- GridSearchCV : 하이퍼파라미터 값을 딕셔너리 형태로 제시하여, 그중 가장 좋은 모델 성능을 제공하는 하이퍼파라미터 값 선택.
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

C_list = [0.01, 0.1, 1.0, 10.0, 100.0]
grid = {'C' : C_list}
logistic_gs = GridSearchCV(estimator = LogisticRegression(random_state = 1),
                          param_grid = grid,
                          scoring = 'accuracy', cv = 10, n_jobs = -1)
logistic_gs.fit(X_train_std, Y_train)

print(logistic_gs.best_score_)
print(logistic_gs.best_params_) # 하이퍼파라미터 C(10) 값을 로지스틱 회귀 모델에 적용
```
   
### 모델 훈련
```
logistic = LogisticRegression(C = 10, random_state = 1)
logistic.fit(X_train_std, Y_train)
```

### 2. SVR 교차검증
- SVR : Support Vector Regression. n차원 공간에서 각 클래스 간의 거리를 최대화하는 최적의 선 또는 초평면을 찾아 데이터를 분류(데이터 포인트 사이의 마진이 최대인 초평면 선택).
- GridSearchCV : 하이퍼파라미터 값을 딕셔너리 형태로 제시하여, 그중 가장 좋은 모델 성능을 제공하는 하이퍼파라미터 값 선택.
```
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

grid = {'C' : [0.1, 1, 10, 100], 'degree' : np.arange(1, 11)}
svr_gs = GridSearchCV(estimator = SVR(kernel = 'poly', epsilon = 0.1, gamma = 'scale'),
                     param_grid = grid, scoring = 'r2', cv = 5)
svr_gs.fit(X_train_std, Y_train)

print(svr_gs.best_score_)
print(svr_gs.best_params_) # 하이퍼파라미터 C(100), degree(10) 값을 SVR 모델에 적용
```
### NuSVR 교차검증
```
from sklearn.svm import NuSVR
from sklearn.model_selection import GridSearchCV

grid = {'C' : [0.1, 1, 10, 100], 'degree' : np.arange(1, 11), 'nu' : [0.1, 0.2, 0.3, 0.4, 0.5]}
nusvr_gs = GridSearchCV(estimator = NuSVR(kernel = 'poly', gamma = 'scale'),
                     param_grid = grid, scoring = 'r2', cv = 5)
nusvr_gs.fit(X_train_std, Y_train)

print(nusvr_gs.best_score_)
print(nusvr_gs.best_params_) # 하이퍼파라미터 C(100), degree(10), nu(0.5) 값을 NuSVR 모델에 적용
```

### 모델 훈련(SVR)
```
svr = SVR(kernel = 'poly', degree = 10, C = 100, epsilon = 0.1, gamma = 'scale')
svr.fit(X_train_std, Y_train)
```

### 모델 훈련(NuSVR)
```
nusvr = NuSVR(nu = 0.5, kernel = 'poly', degree = 10, C = 100, gamma = 'scale')
nusvr.fit(X_train_std, Y_train)
```

### 3. RandomForest 교차검증
- RandomForest : 여러 개의 결정 트리를 무작위로 조합(과적합 갑소 + 예측 성능 향상).
- GridSearchCV : 하이퍼파라미터 값을 딕셔너리 형태로 제시하여, 그중 가장 좋은 모델 성능을 제공하는 하이퍼파라미터 값 선택.
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rfr = RandomForestRegressor(0, n_jobs = -1)
params = {'n_estimators' : [10, 50, 100],
          'max_depth' : [6, 8, 10, 12, 14, 16],
          'min_samples_leaf' : [8, 12, 16, 18],
          'min_samples_split' : [8, 16, 20]}

rfr_gs = GridSearchCV(rfr, param_grid = params, cv = 5, n_jobs = -1)
rfr_gs.fit(X_train, Y_train)

print(rfr_gs.best_score_)
print(rfr_gs.best_params_) # {'max_depth': 16, 'min_samples_leaf': 8, 'min_samples_split': 20, 'n_estimators': 100}
```

### 모델 훈련
```
frf_model = RandomForestRegressor(random_state = 0,
                                  max_depth = 16,
                                  min_samples_leaf = 8,
                                  min_samples_split = 20,
                                  n_estimators = 50)
frf_model.fit(X_train, Y_train)
```

### 4. XGBoost 교차검증
- XGBoost : Extreme Gradient Boosting. 모형들의 학습 에러에 가중치를 두고 순차적으로 다음 모델에 반영하여 강한 예측 모델 생성(병렬 학습 가능, 분류 및 회귀 모두 적용 가능).
```
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgbr = xgb.XGBRegressor()
params = {'n_estimators' : [100, 300, 500, 1000],
          'max_depth' : [10, 12, 14, 16],
          'learning_rate' : [0.1, 0.3, 0.5],
          'colsample_bytree' : [0.1, 0.3, 0.5, 0.7],
          'subsample' : [0.3, 0.5, 0.7],
          'min_child_weight' : [1, 3, 5, 7],
          'random_state' : [42],
          'n_jobs' : [-1]}

xgbr_gs = GridSearchCV(xgbr, param_grid = params, cv = 5, n_jobs = -1)
xgbr_gs.fit(X_train, Y_train)

print(xgbr_gs.best_score_)
print(xgbr_gs.best_params_)
# {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 7, 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42, 'subsample': 0.7}
```

### 모델 훈련
```
xgbr_model = xgb.XGBRegressor(n_estimators = 100,
                              max_depth = 10,
                              learning_rate = 0.1,
                              colsample_bytree = 0.7,
                              subsample = 0.7,
                              min_child_weight = 7,
                              random_state = 42,
                              n_jobs = -1)
xgbr_model.fit(X_train, Y_train)
```

### 5. LightGBM 교차검증
- LightGBM : 리프 기준 분할 방식을 사용하여 최대 손실 값을 갖는 리프 노드를 분할(깊고 비대칭적인 트리 생성). 트리 기준 분할 방식보다 예측 오류 손실 최소화 가능.
```
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

lgbr = lgb.LGBMRegressor(num_leaves = 31, objective = 'regression', boosting = 'gbdt')
params = {'num_iterations' : [10, 50, 100],
          'learning_rate' : [0.1, 0.5],
          'max_depth' : [6, 8, 10, 12, 14, 16],
          'min_data_in_leaf' : [12, 16, 18, 20]}

lgbr_gs = GridSearchCV(lgbr, param_grid = params, cv = 5, n_jobs = -1)
lgbr_gs.fit(X_train, Y_train)

print(lgbr_gs.best_score_)
print(lgbr_gs.best_params_)
# {'learning_rate': 0.1, 'max_depth': 14, 'min_data_in_leaf': 20, 'num_iterations': 100}
```
### 모델 훈련
```
lgbr_model = lgb.LGBMRegressor(num_leaves = 31,
                               objective = 'regression',
                               boosting = 'gbdt',
                               max_depth = 14)
lgbr_model.fit(X_train, Y_train)
```

### 6. 딥러닝 모델
1. 라이브러리 불러오기
```
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AlphaDropout, Dropout # Dense : 은닉층, Dropout and AlphaDropout : 은닉층 사이 신경망 랜덤 제거(효율성 증가)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # 모델 성능 변화가 거의 없을 때 훈련 조기 종료
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.preprocessing import StandardScaler # 표준화
from keras.layers import BatchNormalization # 배치 정규화
from keras.initializers import lecun_normal # 가중치 초기화
```
2. 표준화 & 정규화
```
# 표준화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

3. 활성화 함수
- ReLU : 입력 값이 양수일 때 값을 그대로 출력, 음수일 때 0으로 변환하여 출력(기울기 소실 문제 해결). Dying ReLU 현상 발생.
- ELU : Scaled ELU. 조건에 따른 자기 정규화. 조건(입력 특성 표준화, 모든 은닉층의 가중치에 르쿤 정규분포 초기화 적용, 순차 모델)
- Swish : Google 연구원들이 개발한 자체 게이트 활성화 함수. 입력 값이 음수일 때에도 학습 가능(핛브 가중치와 입력 데이터 표현 향상).
4. 최적화 알고리즘(optimizer)를 적용한 model.compile
- Nadam : Adam + NAG. 파라미터 갱신 과정에서 이전 단계의 Momentum(기울기 값이 0인 곳에서도 관성에 의해 업데이트 수행 가능) 대신, 현재 Momentum 사용(미래의 Momentum 사용 효과)
- Adadelta : 학습률을 자동으로 조정하여, 손실 함수의 최적 값을 빠르고 안정적으로 찾는 최적화 알고리즘(기울기 누적 제한, 동적 학습률 조정).
5. 모델
- MLP : Multi-Layer Perceptron. 지도학습에 사용되는 정방향 인공신경망. 최소 하나 이상의 은닉층 포함
- 순차 모델 : Keras에서 제공하는 딥러닝 모델 구성 방식. Sequential()을 사용해 순차적 구조 모델을 쉽게 구성 가능.
```
# 모델 (activation = relu, MinMaxScaler 적용)
model = Sequential()
model.add(Dense(256, activation = 'relu', input_shape = (X_train.shape[1],)))
# model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
# odel.add(Dropout(0.2))
model.add(Dense(1, activation = 'linear'))
model.summary()

# 학습단계에서 loss(손실함수) 계산, 평가단계에서 metrics(평가지표) 계산
model.compile(loss = 'mse', optimizer = 'AdaDelta', metrics = ['mae'])

estopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True, mode = 'min')
mcheckpoint = ModelCheckpoint('/content/drive/MyDrive/Colab Notebooks/공모전/model_best.keras', monitor = 'val_loss', save_best_only = True, mode =  'min')
```

```
# 모델2 (activation = elu, StandardScaler 적용)
model_2 = Sequential()
model_2.add(Dense(256, activation = 'elu', input_shape = (X_train.shape[1],)))
# model_2.add(Dropout(0.2))
model_2.add(Dense(128, activation = 'elu'))
# model_2.add(Dropout(0.2))
model_2.add(Dense(128, activation = 'elu'))
# model_2.add(Dropout(0.2))
model_2.add(Dense(64, activation = 'elu'))
# model_2.add(Dropout(0.2))
model_2.add(Dense(1, activation = 'linear'))
model_2.summary()

# 학습단계에서 loss(손실함수) 계산, 평가단계에서 metrics(평가지표) 계산
model_2.compile(loss = 'mse', optimizer = 'Nadam', metrics = ['mae'])
```

```
# 모델3 (activation = selu(kernel_initializer = 'lecun_normal', AlphaDropout), StandardScaler 적용)
model_3 = Sequential()
model_3.add(Dense(256, activation = 'selu', kernel_initializer = 'lecun_normal', input_shape = (X_train.shape[1],)))
# model_3.add(AlphaDropout(0.2))
model_3.add(Dense(128, activation = 'selu', kernel_initializer = 'lecun_normal'))
# model_3.add(AlphaDropout(0.2))
model_3.add(Dense(128, activation = 'selu', kernel_initializer = 'lecun_normal'))
# model_3.add(AlphaDropout(0.2))
model_3.add(Dense(64, activation = 'selu', kernel_initializer = 'lecun_normal'))
# model_3.add(AlphaDropout(0.2))
model_3.add(Dense(1, activation = 'linear'))
model_3.summary()

# 학습단계에서 loss(손실함수) 계산, 평가단계에서 metrics(평가지표) 계산
model_3.compile(loss = 'mse', optimizer = 'Nadam', metrics = ['mae'])
```

```
# 모델4 (activation = swish, StandardScaler 적용)
model_4 = Sequential()
model_4.add(Dense(256, activation = 'swish', input_shape = (X_train.shape[1],)))
# model_4.add(Dropout(0.2))
model_4.add(Dense(128, activation = 'swish'))
# model_4.add(Dropout(0.2))
model_4.add(Dense(128, activation = 'swish'))
# model_4.add(Dropout(0.2))
model_4.add(Dense(64, activation = 'swish'))
# model_4.add(Dropout(0.2))
model_4.add(Dense(1, activation = 'linear'))
model_4.summary()

# 학습단계에서 loss(손실함수) 계산, 평가단계에서 metrics(평가지표) 계산
model_4.compile(loss = 'mse', optimizer = 'Nadam', metrics = ['mae'])
```

### 모델 훈련
```
history = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = 500, batch_size = 32, callbacks = [estopping, mcheckpoint])
```

```
history = model_2.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = 500, batch_size = 32, callbacks = [estopping, mcheckpoint])
```

```
history = model_3.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = 500, batch_size = 32, callbacks = [estopping, mcheckpoint], verbose = 1)
```

```
history = model_4.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = 500, batch_size = 32, callbacks = [estopping, mcheckpoint], verbose = 1)
```

### 모델 정밀도 시각화
- 성능 평가 지표
1. MAPE : Mean Absolute Percentage Error(평균 절대 비율 오차). MAE를 비율로 표현하여 스케일 의존적 에러 문제점 개선.
2. RMSE : Root Mean Squared Error(평균 제곱근 오차). MSE에 루트를 씌워 왜곡을 줄임.


![image](https://github.com/user-attachments/assets/f243bbdb-404e-4ab7-b829-efa7c4d5c5f1)
