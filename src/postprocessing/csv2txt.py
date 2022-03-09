def csv2txt(data, txt_path):
    """Convert all CSV files to TXT files.

    Args:
        data (pandas.Dataframe): the data to convert
        txt_path (string): txt file path
    """

    # store in txt file
    with open(txt_path, "w") as txt_file:
        for i in range(data.shape[0]):
            txt_file.write("movl 0 ")
            for j in range(data.shape[1] - 1):
                txt_file.write(f'{float(data.iloc[i, j]):0.4f} ')
            txt_file.write("100.0000 ")
            txt_file.write(f'{data.iloc[i, 6]}\n')