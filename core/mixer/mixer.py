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
from data.firebase.firebase import FireBase
from config import Settings
import json

settings = Settings()
model_path = settings.stylegan_network
model_path_vgg = settings.vgg_network

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


def latent_vector_by_id(id):
    fb = FireBase()
    data = fb.read_query(u'Generated', u'seed', u'==', id)
    if len(data) != 0:
        res = json.loads(data[0].to_dict()["parameters"])["latent"]['0']
        return res
    return None


def generate_projection_mix(src_seeds, style_ranges, id_image, style_tag=0):
    latent_vector = latent_vector_by_id(id_image)
    if latent_vector is None:
        return None

    Gs, synthesis = load_generator()
    fmt = dict(output_transform=dict(
        func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
    with g_Session.as_default():
        src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        dst_latents = np.array(latent_vector)
        dst_dlatents = Gs.components.mapping.run(dst_latents, None)
        src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **fmt)
        dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **fmt)
        row_dlatents = np.stack([dst_dlatents[0]] * len(src_seeds))
        row_dlatents[:, style_ranges[0]] = src_dlatents[:, style_ranges[0]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
        a = save_mix(src_images[0], src_latents, src_seeds[0], dst_images[0], latent_vector, id_image, row_images[0],
                     row_dlatents, "{0}-{1}-{2}".format(src_seeds[0], id_image, style_tag))
    return a

def generate_projection_mix_2(src_seeds, style_ranges, id_image, style_tag=0):
    latent_vector = latent_vector_by_id(id_image)
    if latent_vector is None:
        return None

    Gs, synthesis = load_generator()
    fmt = dict(output_transform=dict(
        func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
    with g_Session.as_default():
        dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        src_latents = np.array(latent_vector)
        dst_dlatents = Gs.components.mapping.run(dst_latents, None)
        src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **fmt)
        dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **fmt)
        row_dlatents = np.stack([src_dlatents[0]] * len(src_seeds))
        row_dlatents[:, style_ranges[0]] = dst_dlatents[:, style_ranges[0]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
        a = save_mix(src_images[0], src_latents, src_seeds[0], dst_images[0], latent_vector, id_image, row_images[0],
                     row_dlatents, "{0}-{1}-{2}".format(src_seeds[0], id_image, style_tag))
    return a

def check_if_exist_seed(seed):
    fb = FireBase()
    data = fb.read_query(u'Generated', u'seed', u'==', seed)
    url = {}
    if len(data) != 0:
        for doc in data:
            url = doc.to_dict()
            url["link_small"] = fb.get_file_url(file=url["link_small"])
            url["link"] = fb.get_file_url(file=url["link"])
            url["id"] = doc.id
        return url
    return None


def mix_images(src_seeds, dst_seeds, style_ranges, style_tag=0):
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
        a = save_mix(src_images[0], src_latents, src_seeds[0], dst_images[0], dst_latents, dst_seeds[0], row_images[0],
                     row_dlatents,
                     "{0}-{1}-{2}".format(src_seeds[0], dst_seeds[0], style_tag))
    return a


def save_mix(src_images, src_latents, src_seed, dst_images, dst_latents, dst_seed, mix_image, mix_latents, mix_seeds):
    id_mix = uuid.uuid4()
    id_src = uuid.uuid4()
    id_dst = uuid.uuid4()
    src_data = check_if_exist_seed(src_seed)
    dst_data = check_if_exist_seed(dst_seed)
    mix_data = check_if_exist_seed(mix_seeds)
    result = {}
    if src_data is None:
        save_image(src_images,
                   {"type_description": "mix_src", "src_id": str(id_src), "type": "2",
                    "latent": dict(enumerate(src_latents.tolist())), "seed": src_seed},
                   "./static/mix/")
    if dst_data is None:
        save_image(dst_images,
                   {"type_description": "mix_dst",
                    "dst_id": str(id_dst), "type": "2",
                    "latent": dict(enumerate(dst_latents.tolist())), "seed": dst_seed},
                   "./static/mix/")
    if mix_data is None:
        save_image(mix_image,
                   {"type_description": "mix_rst", "mix_id": str(id_mix), "type": "2",
                    "latent": dict(enumerate(mix_latents.tolist())), "seed": mix_seeds},
                   "./static/mix/")
    FireBase().create(u'Mixed', {"src_id": str(id_src), "dst_id": str(id_dst), "mix_id": str(id_mix),
                                 "type_description": "mix_images", "type": "2"})
    result["src"] = check_if_exist_seed(src_seed)
    result["dst"] = check_if_exist_seed(dst_seed)
    result["mix"] = check_if_exist_seed(mix_seeds)
    return result
