#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/1/9
"""

import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import glob

# 预训练好的网络模型，来自NVIDIA
Model = './models/generator_yellow-stylegan2-config-f.pkl'
_Gs_cache = dict()


# 加载StyleGAN已训练好的网络模型
def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')

        _G, _D, Gs = pickle.load(model_file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        # Print network details.
        # Gs.print_layers()

        _Gs_cache[model] = Gs
    return _Gs_cache[model]


# 使用generator生成图片
def generate_image(generator, latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))


# 将真实人脸图片对应的latent与改变人脸特性/表情的向量相混合，调用generator生成人脸的变化图片
def move_and_show(generator, flag, latent_vector, direction, coeffs):
    fig, ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    # 调用coeffs数组，生成一系列的人脸变化图片
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
        new_latent_vector = new_latent_vector.reshape((1, 18, 512))
        generator.set_dlatents(new_latent_vector)
        new_person_image = generator.generate_images()[0]
        canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
        canvas.paste(PIL.Image.fromarray(new_person_image, 'RGB'), (0, 0))

        if flag == 0:
            filename = 'new_age.{}.png'
        if flag == 1:
            filename = 'new_angle.{}.png'
        if flag == 2:
            filename = 'new_gender.{}.png'
        if flag == 3:
            filename = 'new_eyes.{}.png'
        if flag == 4:
            filename = 'new_glasses.{}.png'
        if flag == 5:
            filename = 'new_smile.{}.png'

        out_filename = filename.format(str(coeff))
        print('[Info] out filename: {}'.format(out_filename))
        # 将生成的图像保存到文件
        canvas.save(os.path.join(config.result_dir, out_filename))

        # 人脸latent与改变人脸特性/表情的向量相混合，只运算前8层（一共18层）
        # new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
        # ax[i].imshow(generate_image(generator, new_latent_vector))
        # ax[i].set_title('Coeff: %0.1f' % coeff)
    # [x.axis('off') for x in ax]
    # 显示
    # plt.show()

    # 根据看到的人脸变化的效果，输入一个你认为合适的浮点数
    # favor_coeff = float(input('Please input your favourate coeff, such as -1.5 or 1.5: '))
    # new_latent_vector = latent_vector.copy()
    # 用输入的浮点数控制生成新的人脸变化
    # new_latent_vector[:8] = (latent_vector + favor_coeff * direction)[:8]
    # 增加一个维度，以符合generator对向量的要求
    # new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    # 将向量赋值给generator
    # generator.set_dlatents(new_latent_vector)
    # 调用generator生成图片
    # new_person_image = generator.generate_images()[0]
    # 画图，1024x1024
    # canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
    # canvas.paste(PIL.Image.fromarray(new_person_image, 'RGB'), ((0, 0)))
    # 根据不同的标志，存入不同的文件名

    # plt.savefig(filename)


def main():
    # 初始化
    tflib.init_tf()
    # 调用预训练模型
    Gs_network = load_Gs(Model)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    # 读取对应真实人脸的latent，用于图像变化，qing_01.npy可以替换为你自己的文件名
    os.makedirs(config.dlatents_dir, exist_ok=True)
    person = np.load(os.path.join(config.dlatents_dir, 'aoa-mina_01.npy'))

    # 读取已训练好的用于改变人脸特性/表情的向量
    # 包括：改变年龄、改变水平角度、改变性别、改变眼睛大小、是否佩戴眼镜、改变笑容等
    age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
    angle_direction = np.load('ffhq_dataset/latent_directions/angle_horizontal.npy')
    gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
    eyes_direction = np.load('ffhq_dataset/latent_directions/eyes_open.npy')
    glasses_direction = np.load('ffhq_dataset/latent_directions/glasses.npy')
    smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')

    # 混合人脸和变化向量，生成变化后的图片
    move_and_show(generator, 0, person, age_direction, [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    move_and_show(generator, 1, person, angle_direction, [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    move_and_show(generator, 2, person, gender_direction, [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    move_and_show(generator, 3, person, eyes_direction, [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
    move_and_show(generator, 4, person, glasses_direction, [-6, -4, -3, -2, 0, 2, 3, 4, 6])
    move_and_show(generator, 5, person, smile_direction, [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])


if __name__ == "__main__":
    main()
