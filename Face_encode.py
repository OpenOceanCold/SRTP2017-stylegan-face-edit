# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.
import os
import sys
from tqdm import tqdm
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import numpy as np


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
    batch_size = 1
    image_size = 256
    lr = 1.0
    iterations = 1000
    randomize_noise = False
    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    face_img_path = sys.argv[2]
    face_img_list = [face_img_path]
    file_path = os.path.splitext(face_img_path)[0]
    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(face_img_list, batch_size), total=len(face_img_list)//batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
    
        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
        pbar = tqdm(op, leave=False, total=iterations)
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
        print(' '.join(names), ' loss:', loss)
    
        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = Image.fromarray(img_array, 'RGB')
            img.save(file_path + '_generated.png')
            np.save(file_path + '_dlatent.npy', dlatent)
    
        generator.reset_dlatents()

