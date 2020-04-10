import tensorflow as tf

def dice_coe(
  output,
  target,
  loss_type='jaccard', 
  axis=[1, 2, 3], 
  smooth=1e-5, 
  compute='quotient',
  alpha=0.5,
  beta=0.5
):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``dice``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.
    alpha : float
        relative weight of false negatives in Tversky index, default 0.5
    beta : float
        relative weight of false positives in Tversky index, default 0.5

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    inse = tf.cast(tf.reduce_sum(tf.multiply(output,target), axis=axis), dtype=tf.float32)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(tf.multiply(output,output), axis=axis)
        r = tf.reduce_sum(tf.multiply(target,target), axis=axis)
    elif loss_type == 'dice':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    l = tf.cast(l,dtype=tf.float32)
    r = tf.cast(r,dtype=tf.float32)
    
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    tf_smooth = tf.constant(smooth)
    tf_smooth = tf.constant(smooth)
    numerator = tf.constant(2.0) * tf.cast(inse,dtype=tf.float32)
    if alpha==0.5 and beta==0.5:
      denominator = tf.cast(l + r, dtype=tf.float32)
    else:
      tversky_alpha = tf.constant(alpha,dtype=tf.float32)
      tversky_beta  = tf.constant(beta,dtype=tf.float32)
      denominator = tf.cast(2.0*(1.0 - tversky_alpha - tversky_beta*inse + tversky_alpha*l + tversky_beta*r) , dtype=tf.float32)
    if compute == 'quotient':
      dice = (numerator + tf_smooth) / (denominator + tf_smooth)
      #dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
      dice = tf.reduce_mean(dice)
      return dice
    elif compute == 'numerator':
      numerator = tf.reduce_sum(numerator)
      return numerator
    elif compute == 'denominator':
      denominator = tf.reduce_sum(denominator)
      return denominator
    else:
      raise Exception("Unknown compute")

