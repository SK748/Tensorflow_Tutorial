from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import tensorflow_datasets as tfds
import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)
  
parent_dir = os.path.dirname(text_dir)

def labeler(example, index):
    return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)
    
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    # list형을 shuffle하기 위해 concatenate로 자료형을 바꿔주고 shuffle
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)  
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  # tokenize : 띄어쓰기를 기준으로 문자열을 토큰으로 분할한다.
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
  # set함수로 update하기 때문에 중복되는것은 삭제된다.
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

# 머신러닝으로 사용하기 위해선 중복을 걸러내고, 각 문자열을 integer로 바꿔야함 이를 아래 함수가 해줌
# TokenTextEncoder 타입으로 반환
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

# 변수와 label을 int로 바꿔줌
def encode(text_tensor, label):
    # 아래 encode는 TokenTextEncoder에 내장된 encode를 사용함
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label
 
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
# 1. all_labeled_data에서 변수를 들고와 encode_map_fn에 적용시킴
# 2. tf.py_function에 의해 inp의 text와 label이 parameter로 encode에 들어가고, 그 반환값이 Tout에 의해 튜플로 반환됨
all_encoded_data = all_labeled_data.map(encode_map_fn)

# train,test data 만들기

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
# 학습 데이터의 크기는 모두 같아야 함 -> 문장형태의 문제에서는 데이터의 길이가 다름 -> padding을 통해 길이를 통일
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

# padding을 위해 zero를 사용하기 때문에 vocabulary size를 1 증가시켜야함
vocab_size +=1

model = tf.keras.Sequential()
# embedding
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
