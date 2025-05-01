#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import kaggle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# In[ ]:


# download data
data_dir = 'driver-inattention-detection-dataset'
if not os.path.isdir(data_dir):
    kaggle.api.dataset_download_files('zeyad1mashhour/driver-inattention-detection-dataset',
                                    path=data_dir, unzip=True)


# In[ ]:


## loading and preprocessing code to improve model performance for this assignment.

# read the file containing the class labels
with open(os.path.join(data_dir, 'train', '_classes.txt')) as f:
    classes = list(map(str.strip, f))

# read image paths and labels and store them in pandas dataframes
with open(os.path.join(data_dir, 'test', '_annotations.txt'), 'r') as f:
    df_test = pd.DataFrame(dict(img=os.path.join(data_dir, 'test', line.split()[0]),
                                label=classes[int(line.strip()[-1])]) for line in f)

with open(os.path.join(data_dir, 'valid', '_annotations.txt'), 'r') as f:
    df_val = pd.DataFrame(dict(img=os.path.join(data_dir, 'valid', line.split()[0]),
                               label=classes[int(line.strip()[-1])]) for line in f)

with open(os.path.join(data_dir, 'train', '_annotations.txt'), 'r') as f:
    df_train = pd.DataFrame(dict(img=os.path.join(data_dir, 'train', line.split()[0]),
                                 label=classes[int(line.strip()[-1])]) for line in f if len(line.split())>1)

# pre-processing parameters
image_unit_res = 8
img_height, img_width = 9*image_unit_res, 16*image_unit_res # 16:9 preserve aspect ratio
batch_size = 32

def add_noise(img):
    std_coeff = 50*np.random.random()
    noise = np.random.normal(0, std_coeff, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

data_gen = ImageDataGenerator(rescale=1./255, preprocessing_function=add_noise)

train_data = data_gen.flow_from_dataframe(
    df_train, x_col='img', y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode='grayscale')

val_data = data_gen.flow_from_dataframe(
    df_val, x_col='img', y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale')

test_data = data_gen.flow_from_dataframe(
    df_test, x_col='img', y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale')


# In[ ]:


import keras_tuner as kt
from tensorflow.keras import layers, models

def build_model(hp):
    model = models.Sequential()

    # Select number of CNN layers
    num_layers = hp.Int("num_layers", min_value=2, max_value=4, step=1)
    for i in range(num_layers):
        filters = hp.Choice(f"filters_{i}", values=[16, 32, 64])
        model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    # Add Dense layer
    dense_units = hp.Choice("dense_units", values=[64, 128, 256])
    model.add(layers.Dense(dense_units, activation="relu"))

    # Add Dropout layer
    dropout_rate = hp.Float("dropout_rate", min_value=0.3, max_value=0.5, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    # Select output activation function
    output_activation = hp.Choice("output_activation", values=["softmax", "sigmoid"])
    model.add(layers.Dense(6, activation=output_activation))

    # Select learning rate
    learning_rate = hp.Choice("learning_rate", values=[0.0005, 0.001, 0.002])

    # Select optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "rmsprop", "sgd"])
    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# In[ ]:


import os
import json
import kaggle
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ✅ Define EarlyStopping callback to avoid ineffective searching
early_stopping = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001, restore_best_weights=True)


# ✅ Run Keras Tuner to perform hyperparameter search
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="hyper_search",
    project_name="cnn_optimization",
    overwrite=True
)

tuner.search(train_data, validation_data=val_data, epochs=20, batch_size=32, callbacks=[early_stopping])

# ✅ Get the best hyperparameters

best_hps = tuner.get_best_hyperparameters(num_trials=1)
if len(best_hps) > 0:
    best_hp = best_hps[0]
else:
    raise ValueError("No optimal hyperparameters found. Please check the search space!")


# ✅ Print best hyperparameter combination
print(f"🎯 Best hyperparameter combination:")
print(f"🔹 Number of CNN layers: {best_hp.get('num_layers')}")
print(f"🔹 Filters: {[best_hp.get(f'filters_{i}') for i in range(best_hp.get('num_layers'))]}")
print(f"🔹 Dense layer units: {best_hp.get('dense_units')}")
print(f"🔹 Dropout rate: {best_hp.get('dropout_rate')}")
print(f"🔹 Learning rate: {best_hp.get('learning_rate')}")
print(f"🔹 Optimizer: {best_hp.get('optimizer')}")
print(f"🔹 Output activation function: {best_hp.get('output_activation')}")


# ✅ Train the best model
best_model = tuner.hypermodel.build(best_hp)
history = best_model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[early_stopping])



# In[ ]:


# ✅ Log training history
history_log = {
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
}
with open("training_history.json", "w") as f:
    json.dump(history_log, f, indent=4)
print("✅ Training history saved to training_history.json")

# ✅ Save the best model in Keras format
best_model.save("best_cnn_model.keras")
print("✅ Best model saved: best_cnn_model.keras")

# ✅ Reload the model
best_model = tf.keras.models.load_model("best_cnn_model.keras")

# ✅ Evaluate on test set
test_loss, test_acc = best_model.evaluate(test_data, batch_size=32)
print(f"🎯 Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# ✅ Log best hyperparameters & test set results
log_file = "training_log.json"
log_data = {
    "best_hyperparameters": {
        "num_layers": best_hp.get('num_layers'),
        "filters": [best_hp.get(f'filters_{i}') for i in range(best_hp.get('num_layers'))],
        "dense_units": best_hp.get('dense_units'),
        "dropout_rate": best_hp.get('dropout_rate'),
        "learning_rate": best_hp.get('learning_rate'),
        "optimizer": best_hp.get('optimizer'),
        "output_activation": best_hp.get('output_activation'),
        "batch_size": 32  # ✅ Currently, batch size is fixed
    },
    "test_results": {
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }
}

with open(log_file, "w") as f:
    json.dump(log_data, f, indent=4)
print(f"✅ Training log & test results saved to `{log_file}`")

