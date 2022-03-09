import matplotlib.pyplot as plt
import numpy as np
from .calligraphy_transform import calligraphy_transform
from .utils import save_file
import os, shutil,re

def draw_pic():
    path = './output/test_char/test_all_input.txt'
    save_path="./output/visual/"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    save_path+=re.split(r"\.|/", path)[4]

    calligraphy_tool = calligraphy_transform()

    z0_point = 1 #3.21083745 [-66.7041, 438.85, 187.479, -177.603, 4.50068, -9.48322] -2.85887236e-03 [-130.099, 459.278,182.715,175.55,-7.84099,70.2961]
    data_6d, data_cmd = calligraphy_tool.read_file(path, is_6dcmd=True)

    # normal
    data_3d, data_angle = calligraphy_tool.six_to_three(data_6d)
    calligraphy_tool.visualize_line_3d(data_3d, data_cmd, z0_point, save_path, with_thickness=True)

