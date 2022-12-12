from keras.models import load_model
from setup_data import DataSet
import numpy as np


def predict(data_type,  seq_length, saved_model, image_shape, video_name, class_limit):

    model = load_model(saved_model)

    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape, class_limit=class_limit)

    sample = data.get_frames_by_filename(video_name, data_type)

    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def main():

    saved_model = 'data/checkpoints/lstm-features.005-0.401.hdf5'
    seq_length = 40
    class_limit = None
    video_name = 'cheated_map2_092'
    data_type = 'features'
    image_shape = None

    predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit)


if __name__ == '__main__':
    main()