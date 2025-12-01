import tensorflow as tf
from model import cnns_model        
from data_import import data_import 



train_ds, test_ds, class_names = data_import()

model = cnns_model()

model.summary()

print(class_names)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15
)

loss, accuracy = model.evaluate(test_ds, verbose=1)

model.save('moj_model_migowy.h5')


