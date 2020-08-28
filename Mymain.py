# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.
import os
import bz2
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import numpy as np


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
direction_name = ['beauty.npy', 'gender.npy', 'height.npy', 'width.npy', 'age.npy', 'angle_vertical.npy', 'angle_horizontal.npy']


def read_feature(file_name):
    file = open(file_name, mode='r')
    # 使用readlines() 读取所有行的数据，会返回一个列表，列表中存放的数据就是每一行的内容
    contents = file.readlines()
    # 准备一个列表，用来存放取出来的数据
    code = np.zeros((512, ))
    # for循环遍历列表，去除每一行读取到的内容
    for i in range(512):
        name = contents[i]
        name = name.strip('\n')
        code[i] = name
    code = np.float32(code)
    file.close()
    return code


def getLocalFile():
    root = tk.Tk()
    root.withdraw()

    filePath = filedialog.askopenfilename()

    print('文件路径：', filePath)
    return filePath


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def move_latent_and_save(latent_vector, coeffs, file_path, Gs_network, Gs_syn_kwargs):
    new_latent_vector = latent_vector.copy()
    direction = np.load('latent_directions/' + direction_name[1])
    for i in range(45):
        new_latent_vector[0][:8] = (latent_vector[0] + float(coeffs) * direction)[:8]
        images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
        result = Image.fromarray(images[0], 'RGB')
        result.save(file_path + direction_name[6] + str(i) + '.png')
        coeffs = coeffs + 0.4





if __name__ == '__main__':
    # img_path = getLocalFile()
    # file_path = os.path.splitext(img_path)[0]
    # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
    #                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))
    # landmarks_detector = LandmarksDetector(landmarks_model_path)
    # face_img_path = file_path + '_face.png'
    # for i, face_landmarks in (enumerate(landmarks_detector.get_landmarks(img_path), start=1)):
    #     image_align(img_path, face_img_path, face_landmarks)
    # face_img = Image.open(face_img_path)
    # face_img.show()
    #
    # # Initialize generator and perceptual model
    tflib.init_tf()
    # flag = True
    # while flag:
    #     print("choose network!")
    #     print("[1]:网红")
    #     print("[2]:黄种人")
    #     print("[3]:明星")
    #     print("[4]:模特")
    #     network_num = int(input("your choose:"))
    #     if network_num == 1:
    #         network_path = 'networks/generator_wanghong-stylegan2-config-f.pkl'
    #         flag = False
    #     elif network_num == 2:
    #         network_path = 'networks/generator_yellow-stylegan2-config-f.pkl'
    #         flag = False
    #     elif network_num == 3:
    #         network_path = 'networks/generator_star-stylegan2-config-f.pkl'
    #         flag = False
    #     elif network_num == 4:
    #         network_path = 'networks/generator_model-stylegan2-config-f.pkl'
    #         flag = False
    network_path = 'networks/generator_yellow-stylegan2-config-f.pkl'
    generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks(network_path)
    # batch_size = 1
    # image_size = 256
    # lr = 1.0
    # iterations = 1000
    # randomize_noise = False
    # generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    # perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    # perceptual_model.build_perceptual_model(generator.generated_image)
    # face_img_list = [face_img_path]
    # # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    # for images_batch in tqdm(split_to_batches(face_img_list, batch_size), total=len(face_img_list)//batch_size):
    #     names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
    #
    #     perceptual_model.set_reference_images(images_batch)
    #     op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
    #     pbar = tqdm(op, leave=False, total=iterations)
    #     for loss in pbar:
    #         pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
    #     print(' '.join(names), ' loss:', loss)
    #
    #     # Generate images from found dlatents and save them
    #     generated_images = generator.generate_images()
    #     generated_dlatents = generator.get_dlatents()
    #     for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
    #         img = Image.fromarray(img_array, 'RGB')
    #         img.save(file_path + '_generated.png')
    #         np.save(file_path + '_dlatent.npy', dlatent)
    #
    #     generator.reset_dlatents()

    # dlatent = np.load('G:/SR/stylegan-srtp/html/static/40102_face_dlatent.npy')

    # w = np.array([dlatent])
    noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items() if name.startswith('noise')]

    face_latent = read_feature('results/generate_codes/0000.txt')
    z = np.stack(face_latent for _ in range(1))
    tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    w = Gs_network.components.mapping.run(z, None)


    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    # 在这儿选择调整的方向，共有21种调整方式，它们的名称与分别对应的功能如下所示。
    '''
        age.npy - 调整年龄
        angle_horizontal.npy - 在左右方向上调整人脸角度
        angle_vertical.npy - 在上下方向上调整人脸角度
        beauty.npy - 调整颜值
        emotion_angry.npy - 调整此项可增添/减弱一些生气的情绪（调整步幅建议缩小）
        emotion_disgust.npy - 调整此项可增添/减弱一些厌恶的情绪（调整步幅建议缩小）
        emotion_easy.npy - 调整此项可增添/减弱一些平静的情绪（调整步幅建议缩小）
        emotion_fear.npy - 调整此项可增添/减弱一些害怕的情绪（调整步幅建议缩小）
        emotion_happy.npy - 调整此项可增添/减弱一些开心的情绪（调整步幅建议缩小）
        emotion_sad.npy - 调整此项可增添/减弱一些伤心的情绪（调整步幅建议缩小）
        emotion_surprise.npy - 调整此项可增添/减弱一些惊讶的情绪（调整步幅建议缩小）
        eyes_open.npy - 调整眼睛的闭合程度
        face_shape.npy - 调整脸型
        gender.npy - 调整性别
        glasses.npy - 调整是否戴眼镜
        height.npy - 调整脸的高度
        race_black.npy - 调整此项可接近/远离向黑种人变化
        race_white.npy - 调整此项可接近/远离向白种人变化
        race_yellow.npy - 调整此项可接近/远离向黄种人变化
        smile.npy - 调整笑容
        width.npy - 调整脸的宽度
    '''
    # direction_file = 'gender.npy'  # 从上面的编辑向量中选择一个
    # 在这儿选择调整的大小，向量里面的值表示调整幅度，可以自行编辑，对于每个值都会生成一张图片并保存。
    coeffs = -10
    file_path = 'G:/SR/stylegan-srtp/results/0_beauty/'
    # 开始调整并保存图片
    move_latent_and_save(w, coeffs, file_path, Gs_network, Gs_syn_kwargs)