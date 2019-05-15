import tensorflow as tf

def IoU_calculator(x, y, w, h, l_x, l_y, l_w, l_h):
    """calaulate IoU
    Args:
      x: net predicted x
      y: net predicted y
      w: net predicted width
      h: net predicted height
      l_x: label x
      l_y: label y
      l_w: label width
      l_h: label height
    
    Returns:
      IoU
    """
    
    # convert to coner
    x_max = x + w/2
    y_max = y + h/2
    x_min = x - w/2
    y_min = y - h/2
 
    l_x_max = l_x + l_w/2
    l_y_max = l_y + l_h/2
    l_x_min = l_x - l_w/2
    l_y_min = l_y - l_h/2
    # calculate the inter
    inter_x_max = tf.minimum(x_max, l_x_max)
    inter_x_min = tf.maximum(x_min, l_x_min)
 
    inter_y_max = tf.minimum(y_max, l_y_max)
    inter_y_min = tf.maximum(y_min, l_y_min)
 
    inter_w = inter_x_max - inter_x_min
    inter_h = inter_y_max - inter_y_min
    
    inter = tf.cond(tf.logical_or(tf.less_equal(inter_w,0), tf.less_equal(inter_h,0)), 
                    lambda:tf.cast(0,tf.float32), 
                    lambda:tf.multiply(inter_w,inter_h))
    # calculate the union
    union = w*h + l_w*l_h - inter
    
    IoU = inter / union
    return IoU
