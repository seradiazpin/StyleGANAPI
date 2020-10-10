import warnings

from data.firebase.firebase import FireBase

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf
from threading import Lock
from core.util import encoder
import dnnlib as dnnlib
from dnnlib import tflib
import uuid

model_path = "network/network-snapshot-008484.pkl"
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


def generate(seed=4444, latents=None):
    global g_Session
    Gs, synthesis = load_generator()

    if seed is not None:
        rnd = np.random.RandomState(seed)
    else:
        rnd = np.random.RandomState(4444)
    if latents is None or len(latents) == 0:
        latents = rnd.randn(1, Gs.input_shape[1])
    else:
        latents = np.array([np.array(latents)])

    images = None
    # Generate image.
    fmt = dict(output_transform=dict(
        func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
    with g_Session.as_default():
        images = Gs.run(latents, None, truncation_psi=0.5,
                        randomize_noise=True, **fmt)  # 6.95s
    return save_image(images[0])


def save_image(images, file="./static/generated/example.png"):
    # Save image.
    new_img = PIL.Image.fromarray(images, 'RGB').resize(
        (1920, 1080), PIL.Image.ANTIALIAS)
    if file != "":
        PIL.Image.fromarray(images, 'RGB').resize(
            (1920, 1080), PIL.Image.ANTIALIAS).save(file)
    FireBase().create(u'Generated', {"link": "generated/{0}.png".format(uuid.uuid4()), "tags": ["test"], "time": "test", "path": file})
    return encoder.img_to_base64(new_img)
