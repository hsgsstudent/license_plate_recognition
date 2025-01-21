import numpy as np
import cv2

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'T': '1',
                    'Z': '2',
                    'D': '0',
                    'B': '8',}

dict_int_to_char = {'0': 'D',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '2': 'Z',
                    '8': 'B',
                    '7': 'Z',
                    '9': 'Q',}


def get_digits_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)
    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)

    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train


def get_labels(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, labels, boxes):
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[0] + boxes[2])
    y_max = round(boxes[1] + boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    
    # cv2.imwrite('image.jpg', image)

    image = cv2.putText(image, labels, (x_min - 20, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)


    return image


def get_output_layers(model):
    layers_name = model.getLayerNames()
    output_layers = [layers_name[i - 1] for i in model.getUnconnectedOutLayers()]
    return output_layers


def order_points(coordinates):
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect

def license_format_line1(line):
    line = list(line)
    if len(line) >= 3:
        if line[0] in dict_char_to_int:
            line[0] = dict_char_to_int[line[0]]
        if line[1] in dict_char_to_int:
            line[1] = dict_char_to_int[line[1]]
        if line[2] in dict_int_to_char:
            line[2] = dict_int_to_char[line[2]]
    return ''.join(line)

def license_format_line2(line):
    line = list(line)
    if len(line) >= 5:
        if line[0] in dict_char_to_int:
            line[0] = dict_char_to_int[line[0]]
        if line[1] in dict_char_to_int:
            line[1] = dict_char_to_int[line[1]]
        if line[2] in dict_char_to_int:
            line[2] = dict_char_to_int[line[2]]
        if line[3] in dict_char_to_int:
            line[3] = dict_char_to_int[line[3]]
        if line[4] in dict_char_to_int:
            line[4] = dict_char_to_int[line[4]]
    return ''.join(line)


def convert2Square(image):
    """
    Resize non square image(height != width to square one (height == width)
    :param image: input images
    :return: numpy array
    """

    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image
