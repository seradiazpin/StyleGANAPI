import uuid
import warnings

import PIL
import os
from core.projector.projector import Projector
from core.training import misc
from core.util.save import save_image
from data.firebase.firebase import FireBase

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
from threading import Lock
from dnnlib import tflib
import dnnlib
from core.util.save import save_PIL_image

model_path = "network/network-snapshot-008964.pkl"
model_path_vgg = "network/vgg16_zhang_perceptual.pkl"

g_Gs = None
g_Synthesis = None
g_Lpips = None
g_Projector = None
g_Session = None
g_LoadingMutex = Lock()


def load_generator():
    with g_LoadingMutex:
        global g_Gs, g_Synthesis
        if g_Gs:
            return g_Gs, g_Synthesis

        if model_path is None:
            print('invalid model name:', model_path)
            return

        global g_Session
        if g_Session is None:
            print('Initializing dnnlib...')
            dnnlib.tflib.init_tf()
            g_Session = tf.get_default_session()

        print('Loading model %s ...' % model_path)

        with open(model_path, 'rb') as f:
            with g_Session.as_default():
                Gi, Di, Gs = pickle.load(f)
                g_Gs = Gs
                global g_dLatentsIn
                g_dLatentsIn = tf.placeholder(tf.float32, [Gs.components.synthesis.input_shape[1] * Gs.input_shape[1]])
                dlatents_expr = tf.reshape(g_dLatentsIn, [1, Gs.components.synthesis.input_shape[1], Gs.input_shape[1]])
                g_Synthesis = Gs.components.synthesis.get_output_for(dlatents_expr, randomize_noise=False)

    return g_Gs, g_Synthesis


def loadLpips():
    with g_LoadingMutex:
        global g_Lpips
        if g_Lpips:
            return g_Lpips

        if model_path_vgg is None:
            print('invalid model name:', model_path_vgg)
            return

        global g_Session
        if g_Session is None:
            print('Initializing dnnlib...')
            dnnlib.tflib.init_tf()
            g_Session = tf.get_default_session()

        print('Loading model lpips ...')

        with open(model_path_vgg, 'rb') as f:
            with g_Session.as_default():
                lpips = pickle.load(f)
                g_Lpips = lpips

    return g_Lpips


def loadProjector():
    global g_Projector
    if g_Projector:
        return g_Projector

    gs, _ = load_generator()
    lpips = loadLpips()

    g_Projector = Projector()
    g_Projector.regularize_noise_weight = float(os.environ.get('REGULARIZE_NOISE_WEIGHT', 1e5))
    g_Projector.initial_noise_factor = float(os.environ.get('INITIAL_NOISE_FACTOR', 0.05))
    g_Projector.uniform_latents = int(os.environ.get('UNIFORM_LATENTS', 0)) > 0
    g_Projector.euclidean_dist_weight = float(os.environ.get('EUCLIDEAN_DIST_WEIGHT', 1))
    g_Projector.regularize_magnitude_weight = float(os.environ.get('REGULARIZE_MAGNITUDE_WEIGHT', 0))
    g_Projector.set_network(gs, lpips)

    return g_Projector


def generate_projection_mix(src_seeds, style_ranges, image):
    image = PIL.Image.open(image).convert('RGB')
    image = image.resize((1024, 1024), PIL.Image.ANTIALIAS)
    print(np.array(image).shape)
    image_array = np.array(image).swapaxes(0, 2).swapaxes(1, 2)
    image_array = misc.adjust_dynamic_range(image_array, [0, 255], [-1, 1])
    global g_Session
    proj = loadProjector()

    proj.start([image_array])
    projection_images = []
    steps = 200
    snap = 10
    with g_Session.as_default():
        for step in proj.runSteps(steps):
            print('\rstep: %d' % step, end='', flush=True)
            if step % snap == 0 and step != steps:
                results = proj.get_images()
                projection_images.append(save_PIL_image(misc.convert_to_pil_image(
                    misc.create_image_grid(results), drange=[-1, 1]), './static/projected/project-%d.png' % step))
            if step == steps:
                results = proj.get_images()
                projection_images.append(save_PIL_image(misc.convert_to_pil_image(
                    misc.create_image_grid(results), drange=[-1, 1]), './static/projected/project-last.png'))
        dlatents = proj.get_dlatents()
        noises = proj.get_noises()
        with open("latent.txt", "w") as txt_file:
            for line in dlatents[0][17]:
                txt_file.write(" ".join('%.2f' % line) + "\n")
        print('dlatents:', dlatents.shape)
        print('noises:', len(noises), noises[0].shape, noises[-1].shape)
    Gs, synthesis = load_generator()
    fmt = dict(output_transform=dict(
        func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
    a = {}

    with g_Session.as_default():
        src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        dst_dlatents = proj.get_dlatents()
        src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
        ##dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **fmt)
        dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **fmt)
        row_dlatents = np.stack([dst_dlatents[0]] * len(src_seeds))
        row_dlatents[:, style_ranges[0]] = src_dlatents[:, style_ranges[0]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
        a = save_mix(src_images[0], src_latents, dst_images[0], dst_dlatents, row_images[0], row_dlatents)
        """
        for i in range(11):
            row_dlatents[:, style_ranges[0]] = 0.1 * i * src_dlatents[:, style_ranges[0]]
            row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
            a["mix"] = save_image(row_images[0], "./static/mix/mixpos0{0}.png".format(i))
            row_dlatents[:, style_ranges[0]] = 0.1 * i * dst_dlatents[:, style_ranges[0]]
            row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
            a["mix"] = save_image(row_images[0], "./static/mix/mixneg0{0}.png".format(i))
        """
    return a


def mix_images(src_seeds, dst_seeds, style_ranges):
    global g_Session
    Gs, synthesis = load_generator()
    fmt = dict(output_transform=dict(
        func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
    a = {}
    with g_Session.as_default():
        src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
        src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
        dst_dlatents = Gs.components.mapping.run(dst_latents, None)  # [seed, layer, component]
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **fmt)
        dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **fmt)
        row_dlatents = np.stack([dst_dlatents[0]] * len(src_seeds))
        row_dlatents[:, style_ranges[0]] = src_dlatents[:, style_ranges[0]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
        a = save_mix(src_images[0], src_latents, dst_images[0], dst_latents, row_images[0], row_dlatents)
    return a


def save_mix(src_images, src_latents, dst_images, dst_latents, mix_image, mix_latents):
    id_mix = uuid.uuid4()
    id_src = uuid.uuid4()
    id_dst = uuid.uuid4()

    result = {"src": save_image(src_images,
                                {"type_description": "mix_src", "src_id": str(id_src), "type": "2",
                                 "latent": dict(enumerate(src_latents.tolist()))},
                                "./static/mix/"),
              "dst": save_image(dst_images,
                                {"type_description": "mix_dst",
                                 "dst_id": str(id_dst), "type": "2",
                                 "latent": dict(enumerate(dst_latents.tolist()))},
                                "./static/mix/"),
              "mix": save_image(mix_image,
                                {"type_description": "mix_rst", "mix_id": str(id_mix), "type": "2",
                                 "latent": dict(enumerate(mix_latents.tolist()))},
                                "./static/mix/")}
    FireBase().create(u'Mixed', {"src_id": str(id_src), "dst_id": str(id_dst), "mix_id": str(id_mix),
                                 "type_description": "mix_images", "type": "2"})
    return result
