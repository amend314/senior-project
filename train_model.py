from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from setup_data import DataSet
import time
import os.path
from collections import deque
import tensorflow as tf


class Model():
    def __init__(self, nb_classes, model, seq_length, saved_model=None, features_length=2048):

        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        metrics = ['accuracy']

        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Preparing LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()

        optimizer = Adam(learning_rate=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        print(self.model.summary())

    def lstm(self):
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False, input_shape=self.input_shape, dropout=0.5, recurrent_dropout=0.0001))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model


def train(data_type, seq_length, model, saved_model=None, class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, epoch=100):
    checkpoint = ModelCheckpoint(filepath=os.path.join('data', 'checkpoints', model + '-' + data_type +
                                                       '.{epoch:03d}-{val_loss:.3f}.hdf5'), verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))
    early_stopper = EarlyStopping(patience=5)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + str(timestamp) + '.log'))
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)

    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        X, y = data.get_all_sequences('train', data_type)
        X_test, y_test = data.get_all_sequences('test', data_type)
    else:
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    rm = Model(len(data.classes), model, seq_length, saved_model)

    if load_to_memory:
        rm.model.fit_generator(
            X, y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epoch=epoch
        )
    else:
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpoint],
            validation_data=val_generator,
            validation_steps=40,
            workers=4
        )


def main():

    model = 'lstm'
    saved_model = 'data/checkpoints/lstm-features.005-0.401.hdf5'
    class_limit = None
    seq_length = 40
    load_to_memory = False
    batch_size = 32
    epoch = 300
    data_type = 'features'
    image_shape = None

    train(data_type, seq_length, model, saved_model, class_limit,
          image_shape, load_to_memory, batch_size, epoch)


if __name__ == '__main__':
    main()
