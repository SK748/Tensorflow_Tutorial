from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import pandas as pd
import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision = 1, suppress=True)

LABEL_COLUMN = 'survived'
LABELS = [0, 1]

# 저장한 file의 데이터를 아래 조건에 맞게 옮기는 함수
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path,batch_size = 5, label_name = LABEL_COLUMN,na_value = "?",num_epochs = 1, ignore_errors = True, **kwargs)
    return dataset

# dataset의 내용을 보여주는 함수
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s} : {}".format(key, value.numpy()))
            
# Numeric_Features의 col을 age, ~, ~, ~ 에 맞게 만들어주는 함수            
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        print(features)
        print(labels)
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

            return features, labels
# Numeric_Features를 normalization하는 함수
def normalize_numeric_data(data, mean, std):
    return (data-mean)/std


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

# Data normalization (numeric_data)
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

# Categorical_data 정리
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}


categorical_columns = []
for feature, vocab in CATEGORIES.items():
    # one_hot을 통해 데이터 정리
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    
# numeric_data의 normalization, categorical_data의 multi_one_hot을 바탕으로 데이터 전처리
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)    

# 학습 model 생성
model = tf.keras.Sequential([preprocessing_layer,tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dense(1, activation='sigmoid'),])

# 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data
# 학습 시작
model.fit(train_data, epochs=20)


test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]), " | Actual outcome: ", ("SURVIVED" if bool(survived) else "DIED"))