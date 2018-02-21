import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

class Detection:
    def __init__(self,sess, logits, keep_prob, saver, image, image_shape):
        self.sess = sess
        self.logits = logits
        self.keep_prob = keep_prob
        self.saver = saver
        self.image = image
        self.image_shape = image_shape

    def load_vgg(sess, vgg_path):
        """
        Load Pretrained VGG Model into TensorFlow.
        :param sess: TensorFlow Session
        :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
        :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
        """
        # TODO: Implement function
        #   Use tf.saved_model.loader.load to load the model and weights
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)

        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        return w1, keep, w3, w4, w7
#tests.test_load_vgg(Detection.load_vgg, tf)

    def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
        """
        Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
        :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
        :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
        :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
        :param num_classes: Number of classes to classify
        :return: The Tensor for the last layer of output
        """
        # TODO: Implement function
        
        # 1x1 CONVOLUTIONS
        conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,strides=(1,1),padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out,num_classes,1,strides=(1,1),padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out,num_classes,1,strides=(1,1),padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        # UPSAMPLING TO ORIGINAL IMAGE SIZE
        output7 = tf.layers.conv2d_transpose(conv_1x1_7,num_classes,4,strides=(2,2),padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        ## SKIP CONNECTIONS
        skipped7 = tf.add(output7,conv_1x1_4)

        # UPSAMPLING TO ORIGINAL IMAGE SIZE
        output4 = tf.layers.conv2d_transpose(skipped7,num_classes,4,strides=(2,2),padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        ## SKIP CONNECTIONS
        skipped4 = tf.add(output4,conv_1x1_3)

        # UPSAMPLING TO ORIGINAL IMAGE SIZE
        output3 = tf.layers.conv2d_transpose(skipped4,num_classes,16,strides=(8,8),padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        ### SCALING LAYERS
    #    pool3_out_scaled = tf.multiply(pool3_out, 0.0001, name=‘pool3_out_scaled’)
    #    pool4_out_scaled = tf.multiply(pool4_out, 0.01, name=‘pool4_out_scaled’)

        return output3

#tests.test_layers(Detection.layers)

    def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the correct label image
        :param learning_rate: TF Placeholder for the learning rate
        :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """
        # TODO: Implement function
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        correct_label = tf.reshape(correct_label, (-1,num_classes))
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        train_op = optimizer.minimize(cross_entropy_loss)

        return logits, train_op, cross_entropy_loss
#tests.test_optimize(Detection.optimize)

    def train_nn(saver,sess,epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                correct_label, keep_prob, learning_rate):
        """
        Train neural network and print out the loss during training.
        :param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
        :param train_op: TF Operation to train the neural network
        :param cross_entropy_loss: TF Tensor for the amount of loss
        :param input_image: TF Placeholder for input images
        :param correct_label: TF Placeholder for label images
        :param keep_prob: TF Placeholder for dropout keep probability
        :param learning_rate: TF Placeholder for learning rate
        """
        # TODO: Implement function
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print("EPOCH {} ...".format(epoch+1))
            for image,label in get_batches_fn(batch_size):
                ## Training
                _,loss = sess.run([train_op, cross_entropy_loss], 
                                feed_dict={input_image: image, correct_label: label,keep_prob: 0.5, learning_rate: 0.0009})
                print("Loss: = {:.3f}".format(loss))
            print()
        saver.save(sess,'./semseg')

#tests.test_train_nn(Detection.train_nn)

    def try_on_video(video_file):
        clip1 = VideoFileClip(video_file)
        white_clip = clip1.fl_image(pipeline)
        white_clip.write_videofile("project_output.mp4", audio=False)

    def pipeline(image_file):
        with tf.Session() as sess:
            self.saver.restore(self.sess, './semseg')
            image_outputs = helper.gen_video_output(sess, self.logits, self.keep_prob, self.image, self.image_file, self.image_shape)
    
def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    #object = Detection()
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

#   with tf.Graph().as_default():
    with tf.Session() as sess:
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        epochs = 3
        batch_size = 5

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer_3_out, layer_4_out, layer_7_out = Detection.load_vgg(sess,vgg_path)
        last_layer = Detection.layers(layer_3_out, layer_4_out, layer_7_out, num_classes)
        logits, train_op, cross_entropy_loss = Detection.optimize(last_layer, correct_label, learning_rate, num_classes)
        
        #saver = tf.train.Saver({'input_image':input_image,'correct_label':correct_label, 'keep_prob':keep_prob,'learning_rate':learning_rate})
        saver = tf.train.Saver()
        object = Detection(sess, logits, keep_prob,saver, input_image, image_shape)

        # TODO: Train NN using the train_nn function
       # Detection.train_nn(saver,sess,epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
        #    correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # OPTIONAL: Apply the trained model to a video
        Detection.try_on_video("project_video.mp4")

if __name__ == '__main__':
    run()
