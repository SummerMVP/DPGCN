from __future__ import division
import abc
import collections
import math
import sys

import numpy
import tensorflow as tf

from DPGCN.utils import *
# from utils import *
# account.py
EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])


# TODO(liqzhang) To ensure the same API for AmortizedAccountant and
# MomentsAccountant, we pass the union of arguments to both, so we
# have unused_sigma for AmortizedAccountant and unused_eps_delta for
# MomentsAccountant. Consider to revise the API to avoid the unused
# arguments.  It would be good to use @abc.abstractmethod, etc, to
# define the common interface as a base class.

class MomentsAccountant(object):
  """Privacy accountant which keeps track of moments of privacy loss.f

  Note: The constructor of this class creates tf.Variables that must
  be initialized with tf.global_variables_initializer() or similar calls.

  MomentsAccountant accumulates the high moments of the privacy loss. It
  requires a method for computing differenital moments of the noise (See
  below for the definition). So every specific accountant should subclass
  this class by implementing _differential_moments method.

  Denote by X_i the random variable of privacy loss at the i-th step.
  Consider two databases D, D' which differ by one item. X_i takes value
  log Pr[M(D')==x]/Pr[M(D)==x] with probability Pr[M(D)==x].
  In MomentsAccountant, we keep track of y_i(L) = log E[exp(L X_i)] for some
  large enough L. To compute the final privacy spending,  we apply Chernoff
  bound (assuming the random noise added at each step is independent) to
  bound the total privacy loss Z = sum X_i as follows:
    Pr[Z > e] = Pr[exp(L Z) > exp(L e)]
              < E[exp(L Z)] / exp(L e)
              = Prod_i E[exp(L X_i)] / exp(L e)
              = exp(sum_i log E[exp(L X_i)]) / exp(L e)
              = exp(sum_i y_i(L) - L e)
  Hence the mechanism is (e, d)-differentially private for
    d =  exp(sum_i y_i(L) - L e).
  We require d < 1, i.e. e > sum_i y_i(L) / L. We maintain y_i(L) for several
  L to compute the best d for any give e (normally should be the lowest L
  such that 2 * sum_i y_i(L) / L < e.

  We further assume that at each step, the mechanism operates on a random
  sample with sampling probability q = batch_size / total_examples. Then
    E[exp(L X)] = E[(Pr[M(D)==x / Pr[M(D')==x])^L]
  By distinguishing two cases of whether D < D' or D' < D, we have
  that
    E[exp(L X)] <= max (I1, I2)
  where
    I1 = (1-q) E ((1-q) + q P(X+1) / P(X))^L + q E ((1-q) + q P(X) / P(X-1))^L
    I2 = E (P(X) / ((1-q) + q P(X+1)))^L

  In order to compute I1 and I2, one can consider to
    1. use an asymptotic bound, which recovers the advance composition theorem;
    2. use the closed formula (like GaussianMomentsAccountant);
    3. use numerical integration or random sample estimation.

  Dependent on the distribution, we can often obtain a tigher estimation on
  the moments and hence a more accurate estimation of the privacy loss than
  obtained using generic composition theorems.

  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, total_examples, moment_orders=32):
    """Initialize a MomentsAccountant.

    Args:
      total_examples: total number of examples.
      moment_orders: the order of moments to keep.
    """

    assert total_examples > 0
    self._total_examples = total_examples
    self._moment_orders = (moment_orders
                           if isinstance(moment_orders, (list, tuple))
                           else range(1, moment_orders + 1))
    self._max_moment_order = max(self._moment_orders)
    assert self._max_moment_order < 100, "The moment order is too large."
    self._log_moments = [tf.Variable(numpy.float64(0.0),
                                     trainable=False,
                                     name=("log_moments-%d" % moment_order))
                         for moment_order in self._moment_orders]

  @abc.abstractmethod
  def _compute_log_moment(self, sigma, q, moment_order):
    """Compute high moment of privacy loss.

    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      q: the sampling ratio.
      moment_order: the order of moment.
    Returns:
      log E[exp(moment_order * X)]
    """
    pass

  def accumulate_privacy_spending(self, unused_eps_delta,
                                  sigma, num_examples):
    """Accumulate privacy spending.

    In particular, accounts for privacy spending when we assume there
    are num_examples, and we are releasing the vector
    (sum_{i=1}^{num_examples} x_i) + Normal(0, stddev=l2norm_bound*sigma)
    where l2norm_bound is the maximum l2_norm of each example x_i, and
    the num_examples have been randomly selected out of a pool of
    self.total_examples.

    Args:
      unused_eps_delta: EpsDelta pair which can be tensors. Unused
        in this accountant.
      sigma: the noise sigma, in the multiples of the sensitivity (that is,
        if the l2norm sensitivity is k, then the caller must have added
        Gaussian noise with stddev=k*sigma to the result of the query).
      num_examples: the number of examples involved.
    Returns:
      a TensorFlow operation for updating the privacy spending.
    """
    q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples

    moments_accum_ops = []
    for i in range(len(self._log_moments)):
      moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
      moments_accum_ops.append(tf.assign_add(self._log_moments[i], moment))
    return tf.group(*moments_accum_ops)

  def _compute_delta(self, log_moments, eps):
    """Compute delta for given log_moments and eps.

    Args:
      log_moments: the log moments of privacy loss, in the form of pairs
        of (moment_order, log_moment)
      eps: the target epsilon.
    Returns:
      delta
    """
    min_delta = 1.0
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        #sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        continue
      if log_moment < moment_order * eps:
        min_delta = min(min_delta,
                        math.exp(log_moment - moment_order * eps))
    return min_delta

  def _compute_eps(self, log_moments, delta):
    min_eps = float("inf")
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        #sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        continue
      min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
    return min_eps

  def get_privacy_spent(self, sess, target_eps=None, target_deltas=None):
    """Compute privacy spending in (e, d)-DP form for a single or list of eps.

    Args:
      sess: the session to run the tensor.
      target_eps: a list of target epsilon's for which we would like to
        compute corresponding delta value.
      target_deltas: a list of target deltas for which we would like to
        compute the corresponding eps value. Caller must specify
        either target_eps or target_delta.
    Returns:
      A list of EpsDelta pairs.
    """
    assert (target_eps is None) ^ (target_deltas is None)
    eps_deltas = []
    log_moments = sess.run(self._log_moments)
    log_moments_with_order = zip(self._moment_orders, log_moments)
    if target_eps is not None:
      for eps in target_eps:
        eps_deltas.append(
            EpsDelta(eps, self._compute_delta(log_moments_with_order, eps)))
    else:
      assert target_deltas
      for delta in target_deltas:
        eps_deltas.append(
            EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
    return eps_deltas


class GaussianMomentsAccountant(MomentsAccountant):
  """MomentsAccountant which assumes Gaussian noise.

  GaussianMomentsAccountant assumes the noise added is centered Gaussian
  noise N(0, sigma^2 I). In this case, we can compute the differential moments
  accurately using a formula.

  For asymptotic bound, for Gaussian noise with variance sigma^2, we can show
  for L < sigma^2,  q L < sigma,
    log E[exp(L X)] = O(q^2 L^2 / sigma^2).
  Using this we derive that for training T epoches, with batch ratio q,
  the Gaussian mechanism with variance sigma^2 (with q < 1/sigma) is (e, d)
  private for d = exp(T/q q^2 L^2 / sigma^2 - L e). Setting L = sigma^2,
  Tq = e/2, the mechanism is (e, exp(-e sigma^2/2))-DP. Equivalently, the
  mechanism is (e, d)-DP if sigma = sqrt{2 log(1/d)}/e, q < 1/sigma,
  and T < e/(2q). This bound is better than the bound obtained using general
  composition theorems, by an Omega(sqrt{log k}) factor on epsilon, if we run
  k steps. Since we use direct estimate, the obtained privacy bound has tight
  constant.

  For GaussianMomentAccountant, it suffices to compute I1, as I1 >= I2,
  which reduce to computing E(P(x+s)/P(x+s-1) - 1)^i for s = 0 and 1. In the
  companion gaussian_moments.py file, we supply procedure for computing both
  I1 and I2 (the computation of I2 is through multi-precision integration
  package). It can be verified that indeed I1 >= I2 for wide range of parameters
  we have tried, though at the moment we are unable to prove this claim.

  We recommend that when using this accountant, users independently verify
  using gaussian_moments.py that for their parameters, I1 is indeed larger
  than I2. This can be done by following the instructions in
  gaussian_moments.py.
  """

  def __init__(self, total_examples, moment_orders=64):
    """Initialization.

    Args:
      total_examples: total number of examples.
      moment_orders: the order of moments to keep.
    """
    super(self.__class__, self).__init__(total_examples, moment_orders)
    self._binomial_table = GenerateBinomialTable(self._max_moment_order)

  def _differential_moments(self, sigma, s, t):
    """Compute 0 to t-th differential moments for Gaussian variable.

        E[(P(x+s)/P(x+s-1)-1)^t]
      = sum_{i=0}^t (t choose i) (-1)^{t-i} E[(P(x+s)/P(x+s-1))^i]
      = sum_{i=0}^t (t choose i) (-1)^{t-i} E[exp(-i*(2*x+2*s-1)/(2*sigma^2))]
      = sum_{i=0}^t (t choose i) (-1)^{t-i} exp(i(i+1-2*s)/(2 sigma^2))
    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      s: the shift.
      t: 0 to t-th moment.
    Returns:
      0 to t-th moment as a tensor of shape [t+1].
    """
    assert t <= self._max_moment_order, ("The order of %d is out "
                                         "of the upper bound %d."
                                         % (t, self._max_moment_order))
    binomial = tf.slice(self._binomial_table, [0, 0],
                        [t + 1, t + 1])
    signs = numpy.zeros((t + 1, t + 1), dtype=numpy.float64)
    for i in range(t + 1):
      for j in range(t + 1):
        signs[i, j] = 1.0 - 2 * ((i - j) % 2)
    exponents = tf.constant([j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                             for j in range(t + 1)], dtype=tf.float64)
    # x[i, j] = binomial[i, j] * signs[i, j] = (i choose j) * (-1)^{i-j}
    x = tf.multiply(binomial, signs)
    # y[i, j] = x[i, j] * exp(exponents[j])
    #         = (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
    # Note: this computation is done by broadcasting pointwise multiplication
    # between [t+1, t+1] tensor and [t+1] tensor.
    y = tf.multiply(x, tf.exp(exponents))
    # z[i] = sum_j y[i, j]
    #      = sum_j (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
    z = tf.reduce_sum(y, 1)
    return z

  def _compute_log_moment(self, sigma, q, moment_order):
    """Compute high moment of privacy loss.

    Args:
      sigma: the noise sigma, in the multiples of the sensitivity.
      q: the sampling ratio.
      moment_order: the order of moment.
    Returns:
      log E[exp(moment_order * X)]
    """
    assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                    "of the upper bound %d."
                                                    % (moment_order,
                                                       self._max_moment_order))
    binomial_table = tf.slice(self._binomial_table, [moment_order, 0],
                              [1, moment_order + 1])
    # qs = [1 q q^2 ... q^L] = exp([0 1 2 ... L] * log(q))
    qs = tf.exp(tf.constant([i * 1.0 for i in range(moment_order + 1)],
                            dtype=tf.float64) * tf.cast(
                                tf.log(q), dtype=tf.float64))
    moments0 = self._differential_moments(sigma, 0.0, moment_order)
    term0 = tf.reduce_sum(binomial_table * qs * moments0)
    moments1 = self._differential_moments(sigma, 1.0, moment_order)
    term1 = tf.reduce_sum(binomial_table * qs * moments1)
    return tf.squeeze(tf.log(tf.cast(q * term0 + (1.0 - q) * term1,
                                     tf.float64)))


class DummyAccountant(object):
  """An accountant that does no accounting."""

  def accumulate_privacy_spending(self, *unused_args):
    return tf.no_op()

  def get_privacy_spent(self, unused_sess, **unused_kwargs):
    return [EpsDelta(numpy.inf, 1.0)]

import math
from typing import List, Tuple, Union

import numpy as np
from scipy import special


########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf
        for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in the paper mentioned above.
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""Computes RDP of the Sampled Gaussian Mechanism at order ``alpha``.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.0:
        return alpha / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
    q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.

    Returns:
      The RDP guarantees at all orders; can be np.inf.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def get_privacy_spent(
    orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.

    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.

    Returns:
        Pair of epsilon and optimal order alpha.

    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )
    eps = rdp_vec - math.log(delta[0]) / (orders_vec - 1)
    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    return eps[idx_opt], orders_vec[idx_opt]
