import csv
import numpy as np
import random
import glob
import os.path
import operator
import threading
from keras_preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical


def process_image(image, target_shape):

    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300

        self.data = self.get_data()
        self.classes = self.get_classes()
        self.data = self.clean_data()
        self.image_shape = image_shape

    @staticmethod
    def get_data():

        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data

    def clean_data(self):
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames and item[1] in self.classes:
                data_clean.append(item)
        return data_clean

    def get_classes(self):
        classes = []
        for item in self.data:
            if (item[1]) not in classes:
                classes.append((item[1]))
        classes = sorted(classes)

        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        label_encoded = self.classes.index(class_str)
        label_hot = to_categorical(label_encoded, len(self.classes))
        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_TrainTest(self):

        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences(self, train_test, data_type):

        train, test = self.split_TrainTest()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:
            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                sequence = self.build_image_sequence(frames)
            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("No sequence found, may not be generated")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):

        train, test = self.split_TrainTest()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            for _ in range(batch_size):

                sequence = None
                sample = random.choice(data)
                if data_type == "images":
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                else:
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("No sequence found")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):

        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) +
                            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):

        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":

            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            sequence = self.build_image_sequence(frames)
        else:
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence")
        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*.jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        assert len(input_list) >= size
        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]
        return output[:size]

    def print_class_from_prediction(self, prediction, nb_to_return=5):
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = prediction[i]
        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))

