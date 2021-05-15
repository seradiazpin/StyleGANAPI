import os
import warnings

from core.network_operations.INetworkWrapper import INetworkWrapper
from core.projector.projector import Projector

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import dnnlib as dnnlib
from dnnlib import tflib


class StyleGANWrapper(INetworkWrapper):

    def __init__(self, model, vgg):
        self.model = model
        self.vgg = vgg
        self.Gs = None
        self.synthesis = None
        self.lpips = None
        self.projector = None
        self.session = None

    def LoadNetwork(self):
        if self.Gs:
            return self.Gs, self.synthesis
        if self.model is None:
            print('invalid model name:', self.model)
            return
        if self.session is None:
            print('Initializing dnnlib...')
            dnnlib.tflib.init_tf()
            self.session = tf.get_default_session()
        print('Loading model %s ...' % self.model)
        with open(self.model, 'rb') as f:
            with self.session.as_default():
                Gi, Di, Gs = pickle.load(f)
                self.Gs = Gs
                dLatentsIn = tf.placeholder(tf.float32, [Gs.components.synthesis.input_shape[1] * Gs.input_shape[1]])
                dlatents_expr = tf.reshape(dLatentsIn, [1, Gs.components.synthesis.input_shape[1], Gs.input_shape[1]])
                self.synthesis = Gs.components.synthesis.get_output_for(dlatents_expr, randomize_noise=False)

    def LoadVGG(self):
        if self.lpips:
            return self.lpips

        if self.vgg is None:
            print('invalid model name:', self.vgg)
            return

        if self.session is None:
            print('Initializing dnnlib...')
            dnnlib.tflib.init_tf()
            self.session = tf.get_default_session()

        print('Loading model lpips ...')

        with open(self.vgg, 'rb') as f:
            with self.session.as_default():
                lpips = pickle.load(f)
                self.lpips = lpips

    def LoadProjector(self):
        if self.projector:
            return self.projector

        self.LoadNetwork()
        self.LoadVGG()

        self.projector = Projector()
        self.projector.regularize_noise_weight = float(os.environ.get('REGULARIZE_NOISE_WEIGHT', 1e5))
        self.projector.initial_noise_factor = float(os.environ.get('INITIAL_NOISE_FACTOR', 0.05))
        self.projector.uniform_latents = int(os.environ.get('UNIFORM_LATENTS', 0)) > 0
        self.projector.euclidean_dist_weight = float(os.environ.get('EUCLIDEAN_DIST_WEIGHT', 1))
        self.projector.regularize_magnitude_weight = float(os.environ.get('REGULARIZE_MAGNITUDE_WEIGHT', 0))
        self.projector.set_network(self.Gs, self.lpips)

    def GenerateImage(self, seed=None, latents=None):
        self.LoadNetwork()
        if seed is None:
            rnd = np.random.RandomState(seed)
        else:
            rnd = np.random.RandomState()
        if latents is None or len(latents) == 0:
            latents = rnd.randn(1, self.Gs.input_shape[1])
        else:
            latents = np.array([np.array(latents)])
        # Generate image.
        fmt = dict(output_transform=dict(
            func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
        with self.session.as_default():
            images = self.Gs.run(latents, None, truncation_psi=1,
                                 randomize_noise=True, **fmt)  # 6.95s
        return images[0], latents

    def ChoseStyleLayers(self, style_tag):
        if style_tag == 0:
            return [range(0, 4)]
        elif style_tag == 1:
            return [range(4, 8)]
        elif style_tag == 2:
            return [range(8, 16)]
        else:
            return [range(8, 18)]

    def MixImages(self, src_seeds=None, dst_seeds=None, style_tag=0, project_latent=None):
        self.LoadNetwork()
        fmt = dict(output_transform=dict(
            func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=4)
        style_ranges = self.ChoseStyleLayers(style_tag)
        with self.session.as_default():

            src_latents = np.stack(np.random.RandomState(seed).randn(self.Gs.input_shape[1]) for seed in src_seeds)
            src_dlatents = self.Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
            dst_latents = None
            dst_dlatents = None
            if project_latent is None:
                dst_latents = np.stack(np.random.RandomState(seed).randn(self.Gs.input_shape[1]) for seed in dst_seeds)
                dst_dlatents = self.Gs.components.mapping.run(dst_latents, None)  # [seed, layer, component]
            else:
                dst_latents = project_latent
                dst_dlatents = np.array([project_latent])
            src_images = self.Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **fmt)
            dst_images = self.Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **fmt)
            row_dlatents = np.stack([dst_dlatents[0]] * len(src_seeds))
            row_dlatents[:, style_ranges[0]] = src_dlatents[:, style_ranges[0]]
            row_images = self.Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **fmt)
            return src_images[0], src_latents, dst_images[0], dst_latents, row_images[0], row_dlatents

    def ProjectImage(self, image_array=None):
        self.LoadProjector()
        self.projector.start([image_array])
        steps = 1000
        with self.session.as_default():
            for step in self.projector.runSteps(steps):
                print('\rstep: %d' % step, end='', flush=True)
            dlatents = self.projector.get_dlatents()
            results = self.projector.get_images()
            return results, dlatents
