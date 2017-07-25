import tensorflow as tf

def get_pixel_value(img, x, y):
  """
  Utility function to get pixel value for coordinate
  x, y from a 4D tensor.
  """
  N = img.get_shape().as_list()[0]
  C = img.get_shape().as_list()[3]
  return tf.slice(img, [0,x,y,0], [N, 1, 1, C])


def interpolate(img, new_h, new_w):
  """
  perform bilinear interpolation
  """
  N = img.get_shape().as_list()[0]
  H = img.get_shape().as_list()[1]
  W = img.get_shape().as_list()[2]
  C = img.get_shape().as_list()[3]

  max_x = tf.cast(H-1, 'int32')
  max_y = tf.cast(W-1, 'int32')
  zero = tf.zeros([], dtype='int32')

  x = tf.cast(x, 'float32')
  x = 0.5 * ((x+1.0)*tf.cast(W, 'float32'))
  y = tf.cast(y, 'float32')
  y = 0.5 * ((y+1.0)*tf.cast(H, 'float32'))

  x0 = tf.cast(tf.floor(x), 'int32')
  x0 = tf.clip_by_value(x0, zero, max_x)
  x1 = x0 + 1
  x1 = tf.clip_by_value(x1, zero, max_x)
  y0 = tf.cast(tf.floor(y), 'int32')
  y0 = tf.clip_by_value(y0, zero, max_y)
  y1 = y0 + 1
  y1 = tf.clip_by_value(y1, zero, max_y)

  a = get_pixel_value(img, x0, y0)
  b = get_pixel_value(img, x0, y1)
  c = get_pixel_value(img, x1, y0)
  d = get_pixel_value(img, x1, y1)

  x0 = tf.cast(x0, 'float32')
  x1 = tf.cast(x1, 'float32')
  y0 = tf.cast(y0, 'float32')
  y1 = tf.cast(y1, 'float32')

  wa = (x1-x) * (y1-y)
  wb = (x1-x) * (y-y0)
  wc = (x-x0) * (y1-y)
  wd = (x-x0) * (y-y0)

  wa = tf.expand_dims(wa, axis=3)
  wb = tf.expand_dims(wb, axis=3)
  wc = tf.expand_dims(wc, axis=3)
  wd = tf.expand_dims(wd, axis=3)

  out = tf.add_n([wa*a, wb*b, wc*c, wd*d])

  return out

if __name__ == '__main__':
  img = tf.ones([1,3,3,1], 'int32')
  theta = tf.ones([1,2,3])
  grids = affine_grid_generator(10, 10, theta)
  xs = tf.squeeze(grids[:, 0:1, :, :])
  ys = tf.squeeze(grids[:, 1:2, :, :])
  out = interpolate(img, xs, ys)
  with tf.Session() as sess:
    sess.run(img)
    sess.run(out)



