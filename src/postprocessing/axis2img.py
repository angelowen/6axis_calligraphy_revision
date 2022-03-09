import numpy as np
import matplotlib.pyplot as plt
import re

def _angle2deg(angle):
    return angle * np.pi / 180


def _get_3d(csv_file, length=185):
    """
    input: 6 axis csv file path
    path: file path
    length: brush length (default: 185)

    output: (x, y, z) visualized data (type: list)
    """
    data = []

    for row in csv_file.index:
        x = csv_file.iloc[row, 0]
        y = csv_file.iloc[row, 1]
        z = csv_file.iloc[row, 2]
        a = csv_file.iloc[row, 3]
        b = csv_file.iloc[row, 4]
        c = csv_file.iloc[row, 5]
        n_stroke = int(re.search(r'\d+$', csv_file.iloc[row, 6]).group())

        a = _angle2deg(a)
        b = _angle2deg(b)
        c = _angle2deg(c)
        # print(f'{n_stroke}: {x}, {y}, {z}, {a}, {b}, {c}')

        R_a = np.array([
            [1, 0, 0],
            [0, np.cos(a), -1 * np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ])
                     
        R_b = np.array([
            [np.cos(b), 0, np.sin(b)],
            [0, 1, 0],
            [-1 * np.sin(b), 0, np.cos(b)]
        ])

        R_c = np.array([
            [np.cos(c), -1 * np.sin(c), 0],
            [np.sin(c), np.cos(c), 0],
            [0, 0, 1]
        ])

        # R = Rc * Rb * Ra
        R = np.dot(
            np.dot(R_c, R_b),
            R_a
        )

        A = np.array([
            [R[0, 0], R[0, 1], R[0, 2], x],
            [R[1, 0], R[1, 1], R[1, 2], y],
            [R[2, 0], R[2, 1], R[2, 2], z],
            [0, 0, 0, 1]
        ])

        B = np.identity(4)
        B[2, 3] = length

        T = np.dot(A, B)

        data.append([T[0, 3], T[1, 3], T[2, 3], n_stroke])

    return data


def _vis_2d_compare(target, inputs, outputs, path_name, idx):
    r"""
    compare only one stroke
    input: xyz data, character name
    data: x, y, z, stroke num (n * 4)
    output: img of each stroke
    """

    inputs = np.array(inputs).reshape(-1, 4)
    outputs = np.array(outputs).reshape(-1, 4)
    target = np.array(target).reshape(-1, 4)

    inputs_x, inputs_y = [], []
    outputs_x, outputs_y = [], []
    target_x, target_y = [], []
    for row in range(target.shape[0]):
        # z > 5 mean the brush is dangling
        if target[row, 2] > 5:
            continue

        inputs_x.append(inputs[row, 0])
        inputs_y.append(inputs[row, 1])

        outputs_x.append(outputs[row, 0])
        outputs_y.append(outputs[row, 1])

        target_x.append(target[row, 0])
        target_y.append(target[row, 1])

    # plot line
    plt.plot(inputs_x, inputs_y, color='blue', label='noised data')
    plt.plot(outputs_x, outputs_y, color='black', label='revised data')
    plt.plot(target_x, target_y, color='red', label='ground truth')

    # save image
    plt.legend(loc='best')
    plt.title(f'{path_name}/{idx}_compare')
    plt.savefig(f'{path_name}/{idx}_compare.png')

    # plt.show()
    plt.close()


def axis2img(target_data, input_data, output_data, file_feature, path):
    """Convert six axis into 2D image

    Args:
        target_data (pandas.Dataframe): target data
        input_data (pandas.Dataframe): input data
        output_data (pandas.Dataframe): output data
        file_feature (string): file feature to identify
        path (string): director path
    """
    # get 3d data from csv
    target  = _get_3d(target_data)
    inputs  = _get_3d(input_data)
    outputs = _get_3d(output_data)

    # get visual 2d img from ndarray
    _vis_2d_compare(
        target=target,
        inputs=inputs,
        outputs=outputs,
        path_name=path,
        idx=f'{file_feature}'
    )