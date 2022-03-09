import pandas as pd

def _insert_test(test_data, input, stroke_idx):
    """insert the input to test_data with right stroke place

    Args:
        test_data (pandas.DataFrame): test data
        input (pandas.DataFrame): input data to insert
        stroke_idx (int): the input stroke index

    Returns:
        pandas.DataFrame: the test data after inserted
    """
    start = -1  # for get the index to insert

    # find the correct index to insert the stroke
    for row in range(test_data.shape[0]):
        if stroke_idx < int(test_data.iloc[row, 6][6:]):
            start = row
            break

    if start == -1:
        start = test_data.shape[0]

    # output = test_data[:start] + input
    output = pd.concat([test_data.iloc[:start], input], ignore_index=True)

    # output = output + test_data[start:]
    output = output.append(test_data.iloc[start:], ignore_index=True)

    return output

def stroke2char(
                target_data, input_data, output_data,
                test_target, test_input, test_output,
                dir_path, stroke_len, stroke_idx
            ):
    """Merge each stroke into a single character

    Args:
        target_data (pandas.DataFrame): original target data
        input_data (pandas.DataFrame): original input data
        output_data (pandas.DataFrame): original output data

        test_target (pandas.DataFrame): all test target data
        test_input (pandas.DataFrame): all test input data
        test_output (pandas.DataFrame): all test output data

        dir_path (string): the path of directory
        stroke_len (int): the length of stroke
        stroke_idx (int): the stroke index

    Returns:
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame:
                test_target, test_input, test_output
    """

    # Update stroke number
    target_data[6] = [f'stroke{stroke_idx}'] * stroke_len
    input_data[6] = [f'stroke{stroke_idx}'] * stroke_len
    output_data[6] = [f'stroke{stroke_idx}'] * stroke_len

    # insert data
    test_target = _insert_test(test_target, target_data, stroke_idx)
    test_input = _insert_test(test_input, input_data, stroke_idx)
    test_output = _insert_test(test_output, output_data, stroke_idx)

    return test_target, test_input, test_output