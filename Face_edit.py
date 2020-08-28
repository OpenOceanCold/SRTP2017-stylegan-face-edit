# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.
import os
import sys
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import numpy as np


direction_name = ['beauty.npy', 'gender.npy', 'height.npy', 'width.npy', 'age.npy', 'angle_vertical.npy', 'angle_horizontal.npy']


def move_latent_and_save(latent_vector, coeffs, file_path, Gs_network, Gs_syn_kwargs):
    new_latent_vector = latent_vector.copy()
    for i in range(len(coeffs)):
        direction = np.load('latent_directions/' + direction_name[i])
        new_latent_vector[0][:8] = (new_latent_vector[0] + float(coeffs[i]) * direction)[:8]
    images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
    result = Image.fromarray(images[0], 'RGB')
    result.save(file_path + '_Edited.png')



if __name__ == '__main__':
    # Initialize generator and perceptual model
    tflib.init_tf()

    network_path = 'networks/generator_yellow-stylegan2-config-f.pkl'
    network_num = sys.argv[1]
    if network_num == 1:
        network_path = 'networks/stylegan2-ffhq-config-f.pkl'
    elif network_num == 2:
        network_path = 'networks/generator_yellow-stylegan2-config-f.pkl'
    elif network_num == 3:
        network_path = 'networks/generator_star-stylegan2-config-f.pkl'
    elif network_num == 4:
        network_path = 'networks/generator_model-stylegan2-config-f.pkl'
    elif network_num == 5:
        network_path = 'networks/generator_wanghong-stylegan2-config-f.pkl'

    generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks(network_path)

    dlatent_path = sys.argv[2]
    dlatent = np.load(dlatent_path)
    w = np.array([dlatent])

    noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1

    coeffs = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]]
    file_path = os.path.splitext(dlatent_path)[0]

    move_latent_and_save(w, coeffs, file_path, Gs_network, Gs_syn_kwargs)