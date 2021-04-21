import tensorflow.compat.v1 as tf
import config as cfg

def findImages(input_path, output_path):
    image_contents = tf.read_file(input_path)
    image = tf.cast(tf.image.decode_png(image_contents, channels=cfg.FLAGS.input_image_channels), dtype=tf.float32)
    image = tf.image.resize_images(image, [cfg.FLAGS.inference_image_height, cfg.FLAGS.inference_image_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

    return image, output_path

def readDirectories(filenames_path):
    f = open(filenames_path, 'r')
    l_in = []
    l_out = []
    for line in f:
        print(line[:])
        i_name, o_name = line[:].split(',')
        if not tf.gfile.Exists(i_name):
            #raise ValueError('Cannot find file')
            continue

        l_in.append(i_name)
        l_out.append(o_name)

    return l_in, l_out

def generate_iterator(filenames_path):
    image, outputPath = readDirectories(filenames_path)
    image = tf.constant(image)
    outputPath = tf.constant(outputPath)
    data = tf.data.Dataset.from_tensor_slices((image, outputPath))
    data = data.map(findImages)
    data = data.batch(1)
    iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(data), tf.data.get_output_shapes(data))
    iterator_init_op = iterator.make_initializer(data)

    return iterator, iterator_init_op
