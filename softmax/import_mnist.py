import numpy as np

def import_training_samples(amount, use_validation_data=True):
    images_result = []
    labels_result = []

    images_file = open("../MNIST-dataset/train-images.idx3-ubyte", 'rb')
    labels_file = open("../MNIST-dataset/train-labels.idx1-ubyte", 'rb')

    images_file.seek(16)
    labels_file.seek(8)

    image_index = 0
    label_index = 0
    for x in range(amount):
        arr = []
        for y in range(784):
            arr.append(float(int.from_bytes(images_file.read(1), "big")) / 255.0)
            image_index += 1
            images_file.seek(16 + image_index)
        np_arr = np.array(arr)
        np_arr = np_arr.reshape(784, 1)
        images_result.append(np_arr)

        labels_result.append(np.zeros((10, 1), int))

        label = int.from_bytes(labels_file.read(1), "big")
        labels_result[-1][label] = 1

        label_index += 1
        labels_file.seek(8 + label_index)

    
    images_file.close()
    labels_file.close()

    if (use_validation_data):
        validation_images = images_result[-10_000:]
        images_result = images_result[:-10_000]

        validation_labels = labels_result[-10_000:]
        labels_result = labels_result[:-10_000]

        return (list(zip(images_result, labels_result)), list(zip(validation_images, validation_labels)))

    return list(zip(images_result, labels_result))

def import_test_samples(amount):
    images_result = []
    labels_result = []

    images_file = open("../MNIST-dataset/t10k-images.idx3-ubyte", 'rb')
    labels_file = open("../MNIST-dataset/t10k-labels.idx1-ubyte", 'rb')

    images_file.seek(16)
    labels_file.seek(8)

    image_index = 0
    label_index = 0
    for x in range(amount):
        arr = []
        for y in range(784):
            arr.append(float(int.from_bytes(images_file.read(1), "big")) / 255.0)
            image_index += 1
            images_file.seek(16 + image_index)
        np_arr = np.array(arr)
        np_arr = np_arr.reshape(784, 1)
        images_result.append(np_arr)

        labels_result.append(np.zeros((10, 1), int))

        label = int.from_bytes(labels_file.read(1), "big")
        labels_result[-1][label] = 1

        label_index += 1
        labels_file.seek(8 + label_index)

    
    images_file.close()
    labels_file.close()
    return list(zip(images_result, labels_result))