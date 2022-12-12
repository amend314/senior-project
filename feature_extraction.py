from keras_preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import os.path
from setup_data import DataSet
from tqdm import tqdm


class Extractor():
    def __init__(self, weights=None):
        self.weights = weights

        if weights is None:

            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            self.model = Model(
                inputs= base_model.input,
                outputs= base_model.get_layer('avg_pool').output
            )

        else:
            self.model = load_model(weights)

            self.model.layers.pop()
            self.model.layers.pop()
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_node = []

    def extract(self, image_path):
        print(image_path)
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self.model.predict(x)

        if self.weights is None:
            features = features[0]
        else:
            features = features[0]
        return features


seq_length = 40
class_limit = None

data = DataSet(seq_length=seq_length, class_limit=class_limit)
model = Extractor()

prog_bar = tqdm(total=len(data.data))

for video in data.data:

    path = os.path.join('data', 'sequences', video[2] + '-' + str(seq_length) + '-features')
    if os.path.isfile(path + '.npy'):
        prog_bar.update(1)
        continue

    frames = data.get_frames_for_sample(video)
    frames = data.rescale_list(frames, seq_length)

    sequence = []
    for img in frames:
        features = model.extract(img)
        sequence.append(features)

    np.save(path, sequence)
    prog_bar.update(1)
prog_bar.close()
