import time
import uuid
import warnings

from core.util.save import save_PIL_image
from data.firebase.firebase import FireBase

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib
import tensorflow.compat.v1 as tf
import os
from threading import Lock
from core.projector.projector import Projector
from core.training import misc
from config import Settings

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


def generate_projection(image):
    image = PIL.Image.open(image).convert('RGB')
    image = image.resize((1024, 1024), PIL.Image.ANTIALIAS)
    global g_Session
    proj = loadProjector()
    image_array = np.array(image).swapaxes(0, 2).swapaxes(1, 2)
    image_array = misc.adjust_dynamic_range(image_array, [0, 255], [-1, 1])

    proj.start([image_array])
    id_projection = uuid.uuid4()
    steps = 100
    with g_Session.as_default():
        for step in proj.runSteps(steps):
            print('\rstep: %d' % step, end='', flush=True)
        dlatents = proj.get_dlatents()
        results = proj.get_images()
        save_PIL_image(misc.convert_to_pil_image(misc.create_image_grid(results), drange=[-1, 1]),
                       {"type_description": "project_image", "src_id": str(id_projection),
                        "latent": dict(enumerate(dlatents.tolist())), "seed":  str(id_projection)})

    exist = check_if_exist_seed(str(id_projection))
    return exist
