from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

# heart.csv파일을 다운받는다.
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

# 다운받은 파일을 읽어 DataFrame으로 생성
df = pd.read_csv(csv_file)

# df.dtypes : 데이터 타입을 확인해 int나 float로 만들어준다.

# thal의 dtype이 object이기 때문에 Categorical을 통해 1,2,3,4로 만들어준다.
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

#  [ ] 안에 같이있는 label을 분리해 [ ([x], [label]), () ...] 형태로 만들어준다.
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

train_dataset = dataset.shuffle(len(df)).batch(1)

# 모델 생성
def get_compiled_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),tf.keras.layers.Dense(10,activation = 'relu'),tf.keras.layers.Dense(1, activation = 'sigmoid')])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
    return model

model = get_compiled_model()
model.fit(train_dataset, epochs = 15)