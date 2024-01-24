# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NeRF and its MLPs, with helper functions for construction and rendering."""

import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Tuple

import ipdb
from flax import linen as nn
import gin
from flax.training import checkpoints

from internal import configs
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import render
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import torch
gin.config.external_configurable(math.safe_exp, module='math')
gin.config.external_configurable(coord.contract, module='coord')


def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = random.split(rng)
  return key, rng


@gin.configurable
class Model(nn.Module):
  """A mip-Nerf360 model containing all MLPs."""
  config: Any = None  # A Config class, must be set upon construction.
  num_prop_samples: int = 64  # The number of samples for each proposal level.
  num_nerf_samples: int = 32  # The number of samples the final nerf level.
  num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
  bg_intensity_range: Tuple[float] = (1., 1.)  # The range of background colors.
  anneal_slope: float = 10  # Higher = more rapid annealing.
  stop_level_grad: bool = True  # If True, don't backprop across levels.
  use_viewdirs: bool = True  # If True, use view directions as input.
  raydist_fn: Callable[..., Any] = None  # The curve used for ray dists.
  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  disable_integration: bool = False  # If True, use PE instead of IPE.
  single_jitter: bool = True  # If True, jitter whole rays instead of samples.
  dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
  dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
  num_glo_features: int = 0  # GLO vector length, disabled if 0.
  num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
  learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
  near_anneal_rate: Optional[float] = None  # How fast to anneal in near bound.
  near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
  single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
  resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
  use_gpu_resampling: bool = False  # Use gather ops for faster GPU resampling.
  opaque_background: bool = False  # If true, make the background opaque.
  name: str = 'not_setted'
  learn_cam_exposure: bool = False

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac,
      compute_extras,
      zero_glo=True,
      is_train=True,
  ):
    """The mip-NeRF Model.

    Args:
      rng: random number generator (or None for deterministic output).
      rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.

    Returns:
      ret: list, [*(rgb, distance, acc)]
      rendering: the rendering results of each pixels
      ray_history: results of each intervals on the ray
    """

    # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
    # being regularized.
    nerf_mlp = NerfMLP()
    prop_mlp = nerf_mlp if self.single_mlp else PropMLP()

    if self.num_glo_features > 0:
      if not zero_glo:
        # Construct/grab GLO vectors for the cameras of each input ray.
        glo_vecs = nn.Embed(self.num_glo_embeddings, self.num_glo_features)
        cam_idx = rays.cam_idx[..., 0]
        glo_vec = glo_vecs(cam_idx)
      else:
        glo_vec = jnp.zeros(rays.origins.shape[:-1] + (self.num_glo_features,))
    else:
      glo_vec = None

    if self.learned_exposure_scaling:
      # rawnerf
      # Setup learned scaling factors for output colors.
      max_num_exposures = self.num_glo_embeddings
      # Initialize the learned scaling offsets at 0.
      init_fn = jax.nn.initializers.zeros
      exposure_scaling_offsets = nn.Embed(
        max_num_exposures,
        features=3,
        embedding_init=init_fn,
        name='exposure_scaling_offsets')

    cam_exposures = None
    if self.learn_cam_exposure:
      exposure_scaling_embedding = nn.Embed(
        1000,
        features=1,
        embedding_init=jax.nn.initializers.ones,
        name='cam_exposure_factor')

    # Define the mapping from normalized to metric ray distance.
    _, s_to_t = coord.construct_ray_warps(self.raydist_fn, rays.near, rays.far)

    # Initialize the range of (normalized) distances for each ray to [0, 1],
    # and assign that single interval a weight of 1. These distances and weights
    # will be repeatedly updated as we proceed through sampling levels.
    # `near_anneal_rate` can be used to anneal in the near bound at the start
    # of training, eg. 0.1 anneals in the bound over the first 10% of training.
    if self.near_anneal_rate is None:
      init_s_near = 0.
    else:
      init_s_near = jnp.clip(1 - train_frac / self.near_anneal_rate, 0,
                             self.near_anneal_init)
    init_s_far = 1.
    sdist = jnp.concatenate([
      jnp.full_like(rays.near, init_s_near),
      jnp.full_like(rays.far, init_s_far)
    ],
      axis=-1)
    weights = jnp.ones_like(rays.near)
    prod_num_samples = 1

    ray_history = []
    renderings = []
    # import ipdb; ipdb.set_trace()
    for i_level in range(self.num_levels):
      is_prop = i_level < (self.num_levels - 1)
      num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

      # Dilate by some multiple of the expected span of each current interval,
      # with some bias added in.
      dilation = self.dilation_bias + self.dilation_multiplier * (
          init_s_far - init_s_near) / prod_num_samples

      # Record the product of the number of samples seen so far.
      prod_num_samples *= num_samples

      # After the first level (where dilation would be a no-op) optionally
      # dilate the interval weights along each ray slightly so that they're
      # overestimates, which can reduce aliasing.
      use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
      if i_level > 0 and use_dilation:
        sdist, weights = stepfun.max_dilate_weights(
          sdist,
          weights,
          dilation,
          domain=(init_s_near, init_s_far),
          renormalize=True)
        sdist = sdist[..., 1:-1]
        weights = weights[..., 1:-1]

      # Optionally anneal the weights as a function of training iteration.
      if self.anneal_slope > 0:
        # Schlick's bias function, see https://arxiv.org/abs/2010.09714
        bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, self.anneal_slope)
      else:
        anneal = 1.

      # A slightly more stable way to compute weights**anneal. If the distance
      # between adjacent intervals is zero then its weight is fixed to 0.
      logits_resample = jnp.where(
        sdist[..., 1:] > sdist[..., :-1],
        anneal * jnp.log(weights + self.resample_padding), -jnp.inf)

      # Draw sampled intervals from each ray's current weights.
      # jax.debug.breakpoint()
      key, rng = random_split(rng)
      sdist, centers = stepfun.sample_intervals(
        key,
        sdist,
        logits_resample,
        num_samples,
        single_jitter=self.single_jitter,
        domain=(init_s_near, init_s_far),
        use_gpu_resampling=self.use_gpu_resampling,
        return_centers=True
      )

      # Optimization will usually go nonlinear if you propagate gradients
      # through sampling.
      if self.stop_level_grad:
        sdist = jax.lax.stop_gradient(sdist)

      # Convert normalized distances to metric distances.
      tdist = s_to_t(sdist)

      # Cast our rays, by turning our distance intervals into Gaussians.
      gaussians = render.cast_rays(
        tdist,
        rays.origins,
        rays.directions,
        rays.radii,
        self.ray_shape,
        diag=False)

      # if rays.origins.shape[0] == 2048:
      #   jax.debug.breakpoint()

      if self.disable_integration:
        # Setting the covariance of our Gaussian samples to 0 disables the
        # "integrated" part of integrated positional encoding.
        gaussians = (gaussians[0], jnp.zeros_like(gaussians[1]))

      # Push our Gaussians through one of our two MLPs.
      mlp = prop_mlp if is_prop else nerf_mlp
      key, rng = random_split(rng)

      # ---------------------------------------------------------
      if self.learn_cam_exposure:
        if is_train:
          exp_emb_input = rays.cam_idx
        else:
          exp_emb_input = jnp.zeros_like(rays.cam_idx)

        cam_exposures = exposure_scaling_embedding(exp_emb_input)

      ray_results = mlp(
        key,
        gaussians,
        viewdirs=rays.viewdirs if self.use_viewdirs else None,
        imageplane=rays.imageplane,
        glo_vec=None if is_prop else glo_vec,
        # exposure=rays.exposure_values,
        exposure=cam_exposures,
        # is_train=is_train
      )
      # ---------------------------------------------------------

      # Get the weights used by volumetric rendering (and our other losses).
      weights = render.compute_alpha_weights(
        ray_results['density'],
        tdist,
        rays.directions,
        opaque_background=self.opaque_background,
      )[0]

      # Define or sample the background color for each ray.
      if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
        # If the min and max of the range are equal, just take it.
        bg_rgbs = self.bg_intensity_range[0]
      elif rng is None:
        # If rendering is deterministic, use the midpoint of the range.
        bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
      else:
        # Sample RGB values from the range for each ray.
        key, rng = random_split(rng)
        bg_rgbs = random.uniform(
          key,
          shape=weights.shape[:-1] + (3,),
          minval=self.bg_intensity_range[0],
          maxval=self.bg_intensity_range[1])

      # RawNeRF exposure logic.
      if rays.exposure_idx is not None:
        # Scale output colors by the exposure.
        ray_results['rgb'] *= rays.exposure_values[..., None, :]
        if self.learned_exposure_scaling:
          exposure_idx = rays.exposure_idx[..., 0]
          # Force scaling offset to always be zero when exposure_idx is 0.
          # This constraint fixes a reference point for the scene's brightness.
          mask = exposure_idx > 0
          # Scaling is parameterized as an offset from 1.
          scaling = 1 + mask[..., None] * exposure_scaling_offsets(exposure_idx)
          ray_results['rgb'] *= scaling[..., None, :]

      # Render each ray.
      # Add new visualization :: Step I:
      # Add it to the ray_results and pass it to `extras`
      rendering = render.volumetric_rendering(
        ray_results['rgb'],
        weights,
        tdist,
        bg_rgbs,
        rays.far,
        compute_extras,
        extras={
          k: v
          for k, v in ray_results.items()
          if k.startswith('normals') or k in ['roughness', 'L', 'R', 'rgb_enhanced', 'L_enhanced', 'coeff_gamma',
                                              'coeff_alpha','coeff_beta']
        })

      if compute_extras:
        # Collect some rays to visualize directly. By naming these quantities
        # with `ray_` they get treated differently downstream --- they're
        # treated as bags of rays, rather than image chunks.
        n = self.config.vis_num_rays
        rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
        rendering['ray_weights'] = (
          weights.reshape([-1, weights.shape[-1]])[:n, :])
        rgb = ray_results['rgb']
        rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]

      renderings.append(rendering)
      ray_results['sdist'] = jnp.copy(sdist)
      ray_results['weights'] = jnp.copy(weights)
      ray_history.append(ray_results)

    if compute_extras:
      # Because the proposal network doesn't produce meaningful colors, for
      # easier visualization we replace their colors with the final average
      # color.
      weights = [r['ray_weights'] for r in renderings]
      rgbs = [r['ray_rgbs'] for r in renderings]
      final_rgb = jnp.sum(rgbs[-1] * weights[-1][..., None], axis=-2)
      avg_rgbs = [
        jnp.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
      ]
      for i in range(len(avg_rgbs)):
        renderings[i]['ray_rgbs'] = avg_rgbs[i]

    return renderings, ray_history


def construct_model(rng, rays, config):
  """Construct a mip-NeRF 360 model.

  Args:
    rng: jnp.ndarray. Random number generator.
    rays: an example of input Rays.
    config: A Config class.

  Returns:
    model: initialized nn.Module, a NeRF model with parameters.
    init_variables: flax.Module.state, initialized NeRF model parameters.
  """
  # Grab just 10 rays, to minimize memory overhead during construction.
  ray = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:10],
                               rays)
  model = Model(config=config)
  init_variables = model.init(
    rng,  # The RNG used by flax to initialize random weights.
    rng=None,  # The RNG used by sampling within the model.
    rays=ray,
    train_frac=1.,
    compute_extras=False,
    zero_glo=model.num_glo_features == 0)
  print(f'Running model name: {model.name}')

  # optional: load pretrained data branch
  if config.data_mlp_ckpt is not None:
    from flax.core import freeze, unfreeze
    state = checkpoints.restore_checkpoint(config.data_mlp_ckpt, None, step=config.data_mlp_ckpt_step)
    init_variables = unfreeze(init_variables)
    for k, v in init_variables['params']['NerfMLP_0'].items():
      if k.startswith('Dense_'):
        print(f'Load pretrained layer {k}.')
        init_variables['params']['NerfMLP_0'][k] = state['params']['params']['NerfMLP_0'][k]
    init_variables = freeze(init_variables)
  return model, init_variables

class CSDN_Tem(nn.Module):
  in_ch: int
  out_ch: int

  def setup(self):
    # 卷积层按通道分组
    self.depth_conv = nn.Conv(name="depth_conv", kernel_size=(3, 3), padding="SAME", features=self.in_ch,
                                feature_group_count=self.in_ch)

    # 输出通道数调整
    self.point_conv = nn.Conv(name="point_conv", kernel_size=(1, 1), padding="VALID", features=self.out_ch)

  @nn.compact
  def __call__(self, inputs):
    out = self.depth_conv(inputs)
    out = self.point_conv(out)
    return out


class EnhanceNetNoPool(nn.Module):
  number_f: int = 32

  def setup(self):
    self.relu = nn.relu
    self.sigmoid = nn.sigmoid
    self.e_conv1 = CSDN_Tem(1, self.number_f)
    self.e_conv2 = CSDN_Tem(self.number_f, self.number_f)
    self.e_conv3 = CSDN_Tem(self.number_f, self.number_f)
    self.e_conv4 = CSDN_Tem(self.number_f, self.number_f)
    self.e_conv5 = CSDN_Tem(self.number_f * 2, self.number_f)
    self.e_conv6 = CSDN_Tem(self.number_f * 2, self.number_f)
    self.e_conv7 = CSDN_Tem(self.number_f * 2, 1)

  def enhance(self, x, x_r):
    for _ in range(3):
      x = x + x_r * (jnp.power(x, 2) - x)
    enhance_image_1 = x + x_r * (jnp.power(x, 2) - x)
    x = enhance_image_1 + x_r * (jnp.power(enhance_image_1, 2) - enhance_image_1)
    x = x + x_r * (jnp.power(x, 2) - x)
    x = x + x_r * (jnp.power(x, 2) - x)
    enhance_image = x + x_r * (jnp.power(x, 2) - x)
    return enhance_image

  @nn.compact
  def __call__(self, x):

    x1 = self.relu(self.e_conv1(x))
    x2 = self.relu(self.e_conv2(x1))
    x3 = self.relu(self.e_conv3(x2))
    x4 = self.relu(self.e_conv4(x3))

    x5 = self.relu(self.e_conv5(jnp.concatenate([x3, x4], axis=-1)))
    x6 = self.relu(self.e_conv6(jnp.concatenate([x2, x5], axis=-1)))
    x_r = jnp.tanh(self.e_conv7(jnp.concatenate([x1, x6], axis=-1)))

    enhance_image = self.enhance(x, x_r)
    return enhance_image, x_r

class MLP(nn.Module):
  """A PosEnc MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  bottleneck_width: int = 256  # The width of the bottleneck vector.
  net_depth_viewdirs: int = 1  # The depth of the second part of ML.
  net_width_viewdirs: int = 128  # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  # norm_layer: str = None    # Use batchnorm / layernorm or not.
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
  use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
  use_directional_enc: bool = False  # If True, use IDE to encode directions.
  # If False and if use_directional_enc is True, use zero roughness in IDE.
  enable_pred_roughness: bool = False
  # Roughness activation function.
  roughness_activation: Callable[..., Any] = nn.softplus
  roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
  use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
  use_specular_tint: bool = False  # If True, predict tint.
  use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
  bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
  density_activation: Callable[..., Any] = nn.softplus  # Density activation.
  density_bias: float = -1.  # Shift added to raw densities pre-activation.
  density_noise: float = 0.  # Standard deviation of noise added to raw density.
  rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
  rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
  rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  enable_pred_normals: bool = False  # If True compute predicted normals.
  disable_density_normals: bool = False  # If True don't compute normals.
  disable_rgb: bool = False  # If True don't output RGB.
  warp_fn: Callable[..., Any] = None
  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).

  net_depth_R: int = 1  # reflectance MLP
  net_width_R: int = 128  # reflectance MLP
  decompose_LR: bool = False  # RGB = L * R
  learn_gamma: bool = False
  fixed_gamma_value: float = 2.2
  learn_alpha: bool = False  
  learned_gamma_nc: int = 3  
  learned_alpha_nc: int = 3
  learn_zerodce: bool = False
  learn_residual: bool = False
  learned_residual_nc: int = 3
  skip_layer_R: int = 4
  learn_low_freq_coeff: int = 0  # 0 for false, 1 for learning low freq, 2 for using the input of sigma_MLP.
  low_freq_viewdir_dim: int = 3
  low_freq_pos_deg: int = 1
  stop_gradient_phi: bool = True
  change_coeff_act: bool = False
  sigma:float = 20.0
  omega:float = 20.0

  # learn_cam_exposure: bool = False

  def setup(self):
    # Make sure that normals are computed if reflection direction is used.
    if self.use_reflections and not (self.enable_pred_normals or
                                     not self.disable_density_normals):
      raise ValueError('Normals must be computed for reflection directions.')

    # Precompute and store (the transpose of) the basis being used.
    self.pos_basis_t = jnp.array(
      geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)).T

    # Precompute and define viewdir or refdir encoding function.
    if self.use_directional_enc:
      self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
    else:

      def dir_enc_fn(direction, _):
        # encoded channel: 3 x max_deg x 2 + 3
        return coord.pos_enc(
          direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

      self.dir_enc_fn = dir_enc_fn

    if self.learn_zerodce:
      self.zerodce_model = EnhanceNetNoPool()
      rng = jax.random.PRNGKey(0)
      inputs = jnp.ones((128, 1, 1, 1))
      self.params = self.zerodce_model.init(rng, inputs)
      # output, _ = self.zerodce_model.apply(params, inputs)

  @nn.compact
  def __call__(
      self,
      rng,
      gaussians,
      viewdirs=None,
      imageplane=None,
      glo_vec=None,
      exposure=None,
      # camidx=None
      # is_train=True,
  ):
    """Evaluate the MLP.

    Args:
      rng: jnp.ndarray. Random number generator.
      gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
      viewdirs: jnp.ndarray(float32), [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane: jnp.ndarray(float32), [batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: jnp.ndarray(float32), with a shape of [..., num_rgb_channels].
      density: jnp.ndarray(float32), with a shape of [...].
      normals: jnp.ndarray(float32), with a shape of [..., 3], or None.
      normals_pred: jnp.ndarray(float32), with a shape of [..., 3], or None.
      roughness: jnp.ndarray(float32), with a shape of [..., 1], or None.
    """

    dense_layer = functools.partial(
      nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)())
    # if self.norm_layer == 'bn':
    #   norm_layer = functools.partial(nn.BatchNorm, use_running_average=not is_train)
    # else:
    #   assert self.norm_layer is None
    #   norm_layer = None

    density_key, rng = random_split(rng)

    # -----------------------------------------------------------------
    def predict_density(means, covs):
      """Helper function to output density."""
      # Encode input positions

      if self.warp_fn is not None:
        means, covs = coord.track_linearize(self.warp_fn, means, covs)

      lifted_means, lifted_vars = (
        coord.lift_and_diagonalize(means, covs, self.pos_basis_t))

      # channel num: self.max_deg_point x 3 x 2, range: -1 to 1
      x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                   self.min_deg_point, self.max_deg_point)
      # x = lifted_means
      inputs = x
      # Evaluate network to produce the output density.
      for i in range(self.net_depth):
        x = dense_layer(self.net_width)(x)
        # omega = self.omega*x
        # scale = self.sigma*x
        # x = jnp.cos(omega) * jnp.exp(-(scale ** 2))
        x = self.net_activation(x)
        if i % self.skip_layer == 0 and i > 0:
          x = jnp.concatenate([x, inputs], axis=-1)
      raw_density = dense_layer(1)(x)[..., 0]  # Hardcoded to a single channel.
      # Add noise to regularize the density predictions if needed.
      if (density_key is not None) and (self.density_noise > 0):
        raw_density += self.density_noise * random.normal(
          density_key, raw_density.shape)
      return raw_density, x

    # -----------------------------------------------------------------

    means, covs = gaussians
    if self.disable_density_normals:
      raw_density, x = predict_density(means, covs)
      raw_grad_density = None
      normals = None
    else:
      # Flatten the input so value_and_grad can be vmap'ed.
      means_flat = means.reshape((-1, means.shape[-1]))
      covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1:])

      # Evaluate the network and its gradient on the flattened input.
      predict_density_and_grad_fn = jax.vmap(
        jax.value_and_grad(predict_density, has_aux=True), in_axes=(0, 0))
      (raw_density_flat, x_flat), raw_grad_density_flat = (
        predict_density_and_grad_fn(means_flat, covs_flat))

      # Unflatten the output.
      raw_density = raw_density_flat.reshape(means.shape[:-1])
      x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
      raw_grad_density = raw_grad_density_flat.reshape(means.shape)

      # Compute normal vectors as negative normalized density gradient.
      # We normalize the gradient of raw (pre-activation) density because
      # it's the same as post-activation density, but is more numerically stable
      # when the activation function has a steep or flat gradient.
      normals = -ref_utils.l2_normalize(raw_grad_density)

    if self.enable_pred_normals:
      grad_pred = dense_layer(3)(x)

      # Normalize negative predicted gradients to get predicted normal vectors.
      normals_pred = -ref_utils.l2_normalize(grad_pred)
      normals_to_use = normals_pred
    else:
      grad_pred = None
      normals_pred = None
      normals_to_use = normals

    # Apply bias and activation to raw density
    density = self.density_activation(raw_density + self.density_bias)

    roughness = None
    if self.disable_rgb:
      rgb = jnp.zeros_like(means)
    else:
      R = None
      if viewdirs is not None:
        # Predict diffuse color.
        if self.use_diffuse_color:
          raw_rgb_diffuse = dense_layer(self.num_rgb_channels)(x)

        if self.use_specular_tint:
          tint = nn.sigmoid(dense_layer(3)(x))

        if self.enable_pred_roughness:
          raw_roughness = dense_layer(1)(x)
          roughness = (
            self.roughness_activation(raw_roughness + self.roughness_bias))

        # Output of the first part of MLP.
        if self.bottleneck_width > 0:
          bottleneck = dense_layer(self.bottleneck_width)(x)

          # Add bottleneck noise.
          if (rng is not None) and (self.bottleneck_noise > 0):
            key, rng = random_split(rng)
            bottleneck += self.bottleneck_noise * random.normal(
              key, bottleneck.shape)

          x = [bottleneck]
        else:
          x = []

        # Encode view (or reflection) directions.
        if self.use_reflections:
          # Compute reflection directions. Note that we flip viewdirs before
          # reflecting, because they point from the camera to the point,
          # whereas ref_utils.reflect() assumes they point toward the camera.
          # Returned refdirs then point from the point to the environment.
          refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
          # Encode reflection directions.
          dir_enc = self.dir_enc_fn(refdirs, roughness)
        else:
          # Encode view directions.
          dir_enc = self.dir_enc_fn(viewdirs, roughness)

          # if viewdirs.shape[0] == 2048:
          #   # ipdb.set_trace()
          #   jax.debug.breakpoint()

          dir_enc = jnp.broadcast_to(
            dir_enc[..., None, :],
            bottleneck.shape[:-1] + (dir_enc.shape[-1],))

        # input bottleneck, output reflectance
        # ---------------------------------------------------------------
        inputs = bottleneck
        x1 = bottleneck
        R = None
        if self.decompose_LR:
          for i in range(self.net_depth_R):
            x1 = dense_layer(self.net_width_R)(x1)
            x1 = self.net_activation(x1)
            if i % self.skip_layer_R == 0 and i > 0:
              x1 = jnp.concatenate([x1, inputs], axis=-1)
          R = self.rgb_activation(dense_layer(3)(x1))
          # jax.debug.print("R_max: {x}, R_min: {y}, R_mean: {z}", x=R.max(), y=R.min(), z=R.mean())

        # Append view (or reflection) direction encoding to bottleneck vector.
        x.append(dir_enc)

        # Append dot product between normal vectors and view directions.
        if self.use_n_dot_v:
          dotprod = jnp.sum(
            normals_to_use * viewdirs[..., None, :], axis=-1, keepdims=True)
          x.append(dotprod)

        # Append GLO vector if used.
        if glo_vec is not None:
          glo_vec = jnp.broadcast_to(glo_vec[..., None, :],
                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
          x.append(glo_vec)

        # Concatenate bottleneck, directional encoding, and GLO.
        x = jnp.concatenate(x, axis=-1)

        # Output of the second part of MLP.
        inputs = x
        for i in range(self.net_depth_viewdirs):
          x = dense_layer(self.net_width_viewdirs)(x)
          x = self.net_activation(x)
          if i % self.skip_layer_dir == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

      # If using diffuse/specular colors, then `rgb` is treated as linear
      # specular color. Otherwise it's treated as the color itself.
      # If decompose_LR is true, then here 'rgb' is actually the illumination map.
      rgb = self.rgb_activation(self.rgb_premultiplier *
                                dense_layer(self.num_rgb_channels)(x) +
                                self.rgb_bias)

      # multiply a learnable exposure factor
      # when decomposing LR, here we only multiply exposure on L
      # else, we multiply exposure and the final RGB.
      if exposure is not None:
        # print('learn exposure!')
        rgb = rgb * exposure

      sg_fn = jax.lax.stop_gradient if self.stop_gradient_phi else lambda xx: xx
      coeff_beta = None
      if self.learn_zerodce:
        x1 = rgb

        x1 = sg_fn(x1)
        temp = x1
        x1 = jnp.reshape(x1, (-1, x1.shape[-1]))
        x1 = jnp.expand_dims(jnp.expand_dims(x1, axis=1), axis=1)  # BHWC

        enhanced_x,coeff_beta = self.zerodce_model.apply(self.params,x1)
        enhanced_x = jnp.reshape(enhanced_x,temp.shape)
        coeff_beta = jnp.reshape(coeff_beta, temp.shape)


      # Learn alpha and gamma if needed
      coeff_gamma = None
      coeff_alpha = None
      final_gamma = None
      residual = 0
      sg_fn = jax.lax.stop_gradient if self.stop_gradient_phi else lambda xx: xx
      if self.learn_gamma or self.learn_alpha or self.learn_residual:
        # get enhance_mlp input
        if self.learn_low_freq_coeff == 0:
          L_feature = sg_fn(jnp.concatenate([x, rgb], axis=-1))

        elif self.learn_low_freq_coeff == 1:
          lifted_means, lifted_vars = (
            coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
          # channel num: max_deg_point x 3 x 2, range: -1 to 1
          pos = coord.integrated_pos_enc(lifted_means, lifted_vars, 0, self.low_freq_pos_deg)
          L_feature = jnp.concatenate([dir_enc[..., :self.low_freq_viewdir_dim], pos], axis=-1)

        elif self.learn_low_freq_coeff == 2:
          lifted_means, lifted_vars = (
            coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
          pos = coord.integrated_pos_enc(lifted_means, lifted_vars, self.min_deg_point, self.max_deg_point)
          L_feature = jnp.concatenate([dir_enc, pos], axis=-1)

        else:
          raise NotImplementedError("")

        # build enhance_mlp
        # use a small net
        x = dense_layer(self.net_width_viewdirs, name='enhance_mlp')(L_feature)
        # if norm_layer is not None:
        #   x = norm_layer(name='enhance_mlp_norm')(x)

        x = self.net_activation(x)

        # out_nc = 0
        if self.learn_gamma:
          # Dense 15:
          # coeff_gamma = dense_layer(self.learned_gamma_nc)(x)
          # coeff_gamma = self.net_activation(coeff_gamma)
          mid = dense_layer(128, name='gamma_mlp_1')(x)
          # if norm_layer is not None:
          #   mid = norm_layer(name='gamma_norm_layer_1')(mid)
          mid = self.net_activation(mid)
          mid = jnp.concatenate([mid, x], axis=-1)
          mid = dense_layer(self.learned_gamma_nc, name='gamma_mlp_2')(mid)
          if self.change_coeff_act:
            coeff_gamma = self.net_activation(mid)
          else:
            coeff_gamma = nn.sigmoid(mid)

        if self.learn_alpha:
          # coeff_alpha = dense_layer(self.learn_alpha_nc)(x)
          # coeff_alpha = self.net_activation(coeff_alpha)
          mid = dense_layer(128, name='alpha_mlp_1')(x)
          # if norm_layer is not None:
          #   mid = norm_layer(name='alpha_norm_layer_1')(mid)
          mid = self.net_activation(mid)
          mid = jnp.concatenate([mid, x], axis=-1)
          mid = dense_layer(self.learned_alpha_nc, name='alpha_mlp_2')(mid)
          if self.change_coeff_act:
            coeff_alpha = self.net_activation(mid)
          else:
            coeff_alpha = nn.sigmoid(mid)

      # get real RGB if needed
      L = rgb_enhanced = L_enhanced = gamma_base = None
      if R is not None:
        # L ([bs, n_rays, 1], from 0 ~ 1) times R ([bs, n_rays, 3], from 0 ~ 1)
        L = rgb

        L = jnp.broadcast_to(L, R.shape)
        rgb = L * R
        if self.learn_zerodce:
          L_enhanced = sg_fn(enhanced_x)
          R_sg = sg_fn(R)
          L_enhanced = jnp.broadcast_to(L_enhanced,R.shape)
          rgb_enhanced = L_enhanced * R_sg
          coeff_beta = jnp.broadcast_to(coeff_beta,R.shape)

        if self.learn_gamma or self.learn_alpha or self.learn_residual:
          L_sg = sg_fn(L)
          R_sg = sg_fn(R)

          # get final gamma
          gamma_base = None
          if self.learn_gamma:
            # use a fixed global value
            gamma_base = self.fixed_gamma_value

            final_gamma = 1 / (coeff_gamma + gamma_base)

          assert self.learn_gamma and self.learn_alpha
          L_enhanced = (L_sg / (coeff_alpha + 0.0001)) ** final_gamma + residual

          rgb_enhanced = L_enhanced * R_sg

      if self.use_diffuse_color:
        # Initialize linear diffuse color around 0.25, so that the combined
        # linear color is initialized around 0.5.
        diffuse_linear = nn.sigmoid(raw_rgb_diffuse - jnp.log(3.0))
        if self.use_specular_tint:
          specular_linear = tint * rgb
        else:
          specular_linear = 0.5 * rgb

        # Combine specular and diffuse components and tone map to sRGB.
        rgb = jnp.clip(
          image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

      # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

    return dict(
      density=density,
      rgb=rgb,
      rgb_enhanced=rgb_enhanced,
      L_enhanced=L_enhanced,
      gamma_base=gamma_base,
      coeff_gamma=coeff_gamma,
      coeff_alpha=coeff_alpha,
      coeff_beta=coeff_beta,
      residual=residual,
      L=L,
      R=R,
      raw_grad_density=raw_grad_density,
      grad_pred=grad_pred,
      normals=normals,
      normals_pred=normals_pred,
      roughness=roughness,
    )


@gin.configurable
class NerfMLP(MLP):
  pass


@gin.configurable
class PropMLP(MLP):
  pass


def render_image(
    render_fn: Callable[
      [jnp.array, utils.Rays],
      Tuple[List[Mapping[Text, jnp.ndarray]],
            List[Tuple[jnp.ndarray, ...]]]
    ],
    rays: utils.Rays,
    rng: jnp.array,
    config: configs.Config,
    verbose: bool = True) -> MutableMapping[Text, Any]:
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rng, rays) -> pytree.
    rays: a `Rays` pytree, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    config: A Config class.
    verbose: print progress indicators.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays.origins.shape[:2]
  num_rays = height * width
  rays = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.process_index()
  chunks = []
  idx0s = range(0, num_rays, config.render_chunk_size)
  for i_chunk, idx0 in enumerate(idx0s):
    # pylint: disable=cell-var-from-loop
    if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
      print(f'Rendering chunk {i_chunk}/{len(idx0s) - 1}')
    chunk_rays = (
      jax.tree_util.tree_map(
        lambda r: r[idx0:idx0 + config.render_chunk_size], rays))
    actual_chunk_size = chunk_rays.origins.shape[0]
    rays_remaining = actual_chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = jax.tree_util.tree_map(
        lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by host_count.
    rays_per_host = chunk_rays.origins.shape[0] // jax.process_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = jax.tree_util.tree_map(lambda r: utils.shard(r[start:stop]),
                                        chunk_rays)
    chunk_renderings, _ = render_fn(rng, chunk_rays)

    # Unshard the renderings.
    chunk_renderings = jax.tree_util.tree_map(
      lambda v: utils.unshard(v[0], padding), chunk_renderings)

    # Gather the final pass for 2D buffers and all passes for ray bundles.
    chunk_rendering = chunk_renderings[-1]
    for k in chunk_renderings[0]:
      if k.startswith('ray_'):
        chunk_rendering[k] = [r[k] for r in chunk_renderings]

    chunks.append(chunk_rendering)

  # Concatenate all chunks within each leaf of a single pytree.
  rendering = (
    jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *chunks))
  for k, z in rendering.items():
    if not k.startswith('ray_'):
      # Reshape 2D buffers into original image shape.
      rendering[k] = z.reshape((height, width) + z.shape[1:])

  # After all of the ray bundles have been concatenated together, extract a
  # new random bundle (deterministically) from the concatenation that is the
  # same size as one of the individual bundles.
  keys = [k for k in rendering if k.startswith('ray_')]
  if keys:
    num_rays = rendering[keys[0]][0].shape[0]
    ray_idx = random.permutation(random.PRNGKey(0), num_rays)
    ray_idx = ray_idx[:config.vis_num_rays]
    for k in keys:
      rendering[k] = [r[ray_idx] for r in rendering[k]]

  return rendering
