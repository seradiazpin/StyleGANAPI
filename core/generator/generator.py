import warnings

from core.util.save import save_image
from data.firebase.firebase import FireBase

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
from threading import Lock
import dnnlib as dnnlib
from dnnlib import tflib

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


def generate(seed=4444, latents=None):
    exist = check_if_exist_seed(seed)
    if exist is not None:
        return exist
    else:
        generate_network(seed, latents)
        exist = check_if_exist_seed(seed)
        return exist


def generate_network(seed=4444, latents=None):
    global g_Session
    Gs, synthesis = load_generator()

    if seed != 4444:
        rnd = np.random.RandomState(seed)
    else:
        rnd = np.random.RandomState()
    if latents is None or len(latents) == 0:
        latents = rnd.randn(1, Gs.input_shape[1])
    else:
        latents = np.array([np.array(latents)])

    images = None
    # Generate image.
    fmt = dict(output_transform=dict(
        func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
    with g_Session.as_default():
        images = Gs.run(latents, None, truncation_psi=1,
                        randomize_noise=True, **fmt)  # 6.95s
    return save_image(images[0], {"type_description": "generated", "type": "0", "seed": seed,
                                  "latent": dict(enumerate(latents.tolist()))})
