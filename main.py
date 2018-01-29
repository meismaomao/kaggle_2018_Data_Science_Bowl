from pixel2pixel import pixel2pixel
import tensorflow as tf

def main(argv=None):

    with tf.Session() as sess:
        model = pixel2pixel(sess=sess)
        model.train()

if __name__ == '__main__':
    main()