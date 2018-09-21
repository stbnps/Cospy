
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from time import localtime, strftime
import argparse
import tensorflow as tf
import numpy as np
import deeplab as model
import common

np.set_printoptions(threshold=np.inf)

flags = tf.app.flags

FLAGS = flags.FLAGS

scale = None


# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('batch_size', 50,
                     'Batch size.')

flags.DEFINE_integer('window_size', 1024,
                     'Window size.')
            
flags.DEFINE_integer('window_margin', 150,
                     'Window margin.')


flags.DEFINE_float('base_learning_rate', 0.001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.9,
                   'The rate to decay the base learning rate.')



flags.DEFINE_float('weight_decay', 0.0001,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [FLAGS.window_size, 1],
                           'Image crop size [height, width] during training.')


# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')


# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', [12, 24, 36],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')




def _build_deeplab(inputs, outputs_to_num_classes, train_crop_size, resize_size, is_training):


    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=train_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    outputs_to_scales_to_logits = model.multi_scale_logits(
        inputs,
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

    outputs_to_scales_to_logits[common.OUTPUT_TYPE][model._MERGED_LOGITS_SCOPE] = tf.identity( 
    outputs_to_scales_to_logits[common.OUTPUT_TYPE][model._MERGED_LOGITS_SCOPE],
    name = common.OUTPUT_TYPE
    )

    logits = outputs_to_scales_to_logits[common.OUTPUT_TYPE][model._MERGED_LOGITS_SCOPE]

    logits = tf.image.resize_bilinear(logits, (resize_size, 1), align_corners=True)

    return logits



def preprocess_data_mask(wave, pick, augment):

    if augment:
        resize_scale = tf.random_uniform((), 0.7, 1.3)
        new_size = tf.convert_to_tensor([tf.cast(resize_scale * tf.cast(tf.shape(wave)[0], tf.float32), tf.int32), 1])
        wave = tf.image.resize_images(wave, new_size)
        pick = tf.cast(tf.round(tf.cast(pick, tf.float32) * resize_scale), tf.int64)

    size = FLAGS.window_size
    start = tf.random_uniform([1,], minval=pick-size+FLAGS.window_margin, maxval=pick-FLAGS.window_margin, dtype=tf.int64)

    beging_slice = tf.concat([start, [0,0]], 0)
    
    wave = tf.slice(wave, beging_slice, [size,1,3]) #[start,0, 0], [size, 1, 3])

    p = pick - start

    mask = tf.zeros((size,1), tf.float32)

    p = tf.cast(p, tf.float32)



    r = tf.convert_to_tensor(np.asarray(range(size), dtype=np.float32))
    
    m1 = tf.cast(tf.abs(r - p) < 10, tf.float32)
    m1 = tf.expand_dims(m1, 1)

    m2 = tf.ones_like(m1)
    m2 = m2 - m1
    
    
    mask = tf.stack([m1, m2],axis=2)


    distance_map = p - tf.range(0, size, 1, dtype=tf.float32)
    distance_map /= 64.0
    distance_map = tf.reshape(distance_map, (size, 1, 1))

    return wave, mask, distance_map, tf.cast(p, tf.int64)




def parse_fn(example, augment):

    example_fmt = {
    "wave_raw": tf.FixedLenFeature((), tf.string, ""),
    "pick": tf.FixedLenFeature((), tf.int64, -1)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    wave = tf.decode_raw(parsed["wave_raw"], tf.float32)
    wave = tf.reshape(wave, (12000, 1, 3))
    pick = parsed['pick']
    wave, mask, distance_map, p = preprocess_data_mask(wave, pick, augment)

    return wave, mask, distance_map, p


def input_fn(file_pattern, augment):

    dataset = tf.data.TFRecordDataset(file_pattern)

    dataset = dataset.map(lambda x : parse_fn(x, augment))
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_size)
    dataset = dataset.batch(FLAGS.batch_size)

    return dataset

def get_summaries(prefix, cross_entropy_loss_placeholder, l1_loss_placeholder, distance_first_mean_placeholder, distance_first_std_placeholder, distances_first_placeholder, distance_mean_placeholder, distance_std_placeholder, distances_placeholder, precision_placeholder, recall_placeholder, f1_score_placeholder, f1_score_first_placeholder):

    cross_entropy_loss_summary = tf.summary.scalar(prefix + '_cross_entropy_loss', cross_entropy_loss_placeholder)
    l1_loss_summary = tf.summary.scalar(prefix + '_l1_loss', l1_loss_placeholder)
    pick_distance_first_mean_summary = tf.summary.scalar(prefix + '_pick_distance_first_mean', distance_first_mean_placeholder)
    pick_distance_first_std_summary = tf.summary.scalar(prefix + '_pick_distance_first_std', distance_first_std_placeholder)
    pick_distance_first_histogram_summary = tf.summary.histogram(prefix + '_distances_first', distances_first_placeholder)
    pick_distance_mean_summary = tf.summary.scalar(prefix + '_pick_distance_mean', distance_mean_placeholder)
    pick_distance_std_summary = tf.summary.scalar(prefix + '_pick_distance_std', distance_std_placeholder)
    pick_distance_histogram_summary = tf.summary.histogram(prefix + '_distances', distances_placeholder)
    pick_precision_summary = tf.summary.scalar(prefix + '_pick_precision', precision_placeholder)
    pick_recall_summary = tf.summary.scalar(prefix + '_pick_recall', recall_placeholder)
    pick_f1_score_summary = tf.summary.scalar(prefix + '_pick_f1_score', f1_score_placeholder)
    pick_f1_score_first_summary = tf.summary.scalar(prefix + '_pick_f1_score_first', f1_score_first_placeholder)
    
    summaries = tf.summary.merge([cross_entropy_loss_summary, l1_loss_summary, 
        pick_distance_first_mean_summary, pick_distance_first_std_summary, pick_distance_first_histogram_summary, 
        pick_distance_mean_summary, pick_distance_std_summary, pick_distance_histogram_summary, 
        pick_precision_summary, pick_recall_summary, pick_f1_score_summary, pick_f1_score_first_summary])

    return summaries


def hough_voting(predicted_mask, predicted_distance_map):

    predicted_distance_map *= 64.0
    predicted_distance_map += range(predicted_mask.shape[1])

    picks = np.zeros(predicted_mask.shape[0])
    for i in range(predicted_mask.shape[0]):
        weights = predicted_mask[i]
        p = np.nan_to_num(predicted_distance_map[i])
        hough_votes, _ = np.histogram(p, range(predicted_mask.shape[1] + 1), weights = weights)
        pick = np.argmax(hough_votes)
        picks[i] = pick

    return picks

def compute_metrics(tp, fp, fn):
    eps = 0.00001
    if (tp + fp) < eps:
        precision = tp / (tp + fp + eps)
    else:
        precision = tp / (tp + fp)
    
    if (tp + fn) < eps:
        recall = tp / (tp + fn + eps)
    else:
        recall = tp / (tp + fn)
    
    if (precision + recall) < eps:
        f1 = 2.0 * precision * recall / (precision + recall + eps)
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    
    return (precision, recall, f1)

def run_epoch(sess, init_op, next_wave, optimizers, cross_entropies, predicted_pick, next_pick, distance_l1, roi_logits, roi_wave_placeholder, roi_distance_map_placeholder, next_distance_map):
    
    losses_cross_entropy = []
    losses_l1 = []
    distances = []
    distances_first = []
    loss = 0.0

    tp_first = 0.0
    fp_first = 0.0
    fn_first = 0.0

    tp = 0.0
    fp = 0.0
    fn = 0.0
    sess.run(init_op)
    i = 0
    while True:          

        try:  

            
            if optimizers is not None:
                optimizer = optimizers[0]
                feeds = optimizer[1]
                feeds[roi_wave_placeholder] = np.zeros((1,64,1,3))
                feeds[roi_distance_map_placeholder] = np.zeros((1,64,1,1))
                _, next_wave_, cross_entropies_, predicted_pick_, next_pick_, next_distance_map_  = sess.run(
                    [optimizer[0], next_wave, cross_entropies, predicted_pick, next_pick, next_distance_map], feed_dict=feeds)  
            else:

                next_wave_, cross_entropies_, predicted_pick_, next_pick_, next_distance_map_  = sess.run(
                    [next_wave, cross_entropies, predicted_pick, next_pick, next_distance_map])

            
            cross_entropies_ = np.mean(cross_entropies_, (1,2))
            losses_cross_entropy += cross_entropies_.tolist()
            
            predicted_pick_pos = np.argmax(predicted_pick_, 1)
            predicted_pick_pos = np.reshape(predicted_pick_pos, (-1,))

            predicted_pick_val = np.max(predicted_pick_, 1)
            predicted_pick_val = np.reshape(predicted_pick_val, (-1,))
            next_pick_ = np.reshape(next_pick_, (-1,))





            pick_distance = np.abs(predicted_pick_pos - next_pick_)

     
            tp_first += np.sum((predicted_pick_val > 0.5) & (pick_distance < 10))
            fp_first += np.sum((predicted_pick_val > 0.5) & (pick_distance >= 10))
            fn_first += np.sum(predicted_pick_val <= 0.5)

            pick_distance = pick_distance[predicted_pick_val > 0.5]

            distances_first += pick_distance[pick_distance < 50].tolist()


            roi_start = predicted_pick_pos - 32

            roi_start[roi_start > (FLAGS.window_size - 66)] = (FLAGS.window_size - 66) 

            indices_y = np.tile(np.reshape(np.asarray(range(predicted_pick_.shape[0])), (predicted_pick_.shape[0],1)), 64)
            indices_x = np.repeat(np.reshape(np.asarray(range(64)), (1,64)), predicted_pick_.shape[0], 0)
            indices_x += np.reshape(roi_start, (predicted_pick_.shape[0], 1))

            



            rois = next_wave_[indices_y, indices_x, :, :]
            roi_distance_maps = next_distance_map_[indices_y, indices_x, :, :]
            roi_mask_prediction = predicted_pick_[indices_y, indices_x, :, :]


            if optimizers is not None:
                optimizer = optimizers[1]
                feeds = optimizer[1]
                feeds[roi_wave_placeholder] = rois
                feeds[roi_distance_map_placeholder] = roi_distance_maps
                _, distance_l1_, roi_logits_  = sess.run([optimizer[0], distance_l1, roi_logits], feed_dict=feeds) 
            else:
                feeds = {}
                feeds[roi_wave_placeholder] = rois
                feeds[roi_distance_map_placeholder] = roi_distance_maps
                distance_l1_, roi_logits_  = sess.run([distance_l1, roi_logits], feed_dict=feeds) 
            
            
            
             

            l1s = np.mean(distance_l1_, (1,2))
            losses_l1 += l1s.tolist()

            roi_mask_prediction = np.reshape(roi_mask_prediction, (-1, 64))
            roi_logits_ = np.reshape(roi_logits_, (-1, 64))
            predicted_pick_pos = hough_voting(roi_mask_prediction, roi_logits_)

            predicted_pick_pos += roi_start

            pick_distance = np.abs(predicted_pick_pos - next_pick_)

     
            tp += np.sum((predicted_pick_val > 0.5) & (pick_distance < 10))
            fp += np.sum((predicted_pick_val > 0.5) & (pick_distance >= 10))
            fn += np.sum(predicted_pick_val <= 0.5)

            pick_distance = pick_distance[predicted_pick_val > 0.5]

            distances += pick_distance[pick_distance < 50].tolist()


        except tf.errors.OutOfRangeError:
            break
    
    distances = np.asarray(distances)
    distances_first = np.asarray(distances_first)

    loss_cross_entropy = np.mean(losses_cross_entropy)
    loss_l1 = np.mean(losses_l1)
    distance_mean = np.mean(distances)
    distance_std = np.std(distances)

    distance_first_mean = np.mean(distances_first)
    distance_first_std = np.std(distances_first)


    precision, recall, f1 = compute_metrics(tp, fp, fn)

    precision_first, recall_first, f1_first = compute_metrics(tp_first, fp_first, fn_first)
    

    return (loss_cross_entropy, loss_l1, distance_first_mean, distance_first_std, distances_first, distance_mean, distance_std, distances, precision, recall, f1, precision_first, recall_first, f1_first)

    

    

def write_summary(sess, summaries, summary_writer, it, 
        cross_entropy_loss_placeholder, cross_entropy_loss, 
        l1_loss_placeholder, l1_loss,
        distance_first_mean_placeholder, distance_first_mean, 
        distance_first_std_placeholder, distance_first_std, 
        distances_first_placeholder, distances_first,
        distance_mean_placeholder, distance_mean, 
        distance_std_placeholder, distance_std, 
        distances_placeholder, distances, 
        precision_placeholder, precision,
        recall_placeholder, recall,
        f1_score_placeholder, f1_score,
        f1_score_first_placeholder, f1_score_first):

    summary = sess.run(summaries, feed_dict={
        cross_entropy_loss_placeholder: cross_entropy_loss,
        l1_loss_placeholder: l1_loss,
        distance_first_mean_placeholder: distance_first_mean,
        distance_first_std_placeholder: distance_first_std,
        distances_first_placeholder: distances_first,
        distance_mean_placeholder : distance_mean,
        distance_std_placeholder: distance_std,
        distances_placeholder: distances, 
        precision_placeholder: precision,
        recall_placeholder: recall,
        f1_score_placeholder: f1_score,
        f1_score_first_placeholder: f1_score_first,
    })
    summary_writer.add_summary(summary, it)


def main(unused_argv):

    global scale
    scale = tf.get_variable('normal_scale', dtype=tf.float32, initializer=tf.constant(10.0), trainable=False)


    # Update with your own tfrecords
    training_dataset = input_fn('train.tfrecords', True)
    validation_dataset = input_fn('validation.tfrecords', False)
    test_dataset = input_fn('test.tfrecords', False)

    

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                            training_dataset.output_shapes)

    next_wave, next_pick_mask, next_distance_map, next_pick = iterator.get_next()

    roi_wave_placeholder = tf.placeholder(tf.float32, (None, 64, 1, 3))
    roi_distance_map_placeholder = tf.placeholder(tf.float32, (None, 64, 1, 1))
    

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    

    with tf.variable_scope('first'):
        logits = _build_deeplab(next_wave, {common.OUTPUT_TYPE: 2}, FLAGS.train_crop_size, FLAGS.window_size, is_training = True)
        
    with tf.variable_scope('first', reuse=True):
        logits_test = _build_deeplab(next_wave, {common.OUTPUT_TYPE: 2}, FLAGS.train_crop_size, FLAGS.window_size, is_training = False)


    
    sm_logits = tf.nn.softmax(logits)
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(labels = next_pick_mask, logits = logits)
    loss_cross_entropy = tf.reduce_mean(cross_entropies)
    predicted_pick = tf.slice(sm_logits, [0, 0, 0, 0], [-1, -1, -1, 1])


    sm_logits_test = tf.nn.softmax(logits_test)
    cross_entropies_test = tf.nn.softmax_cross_entropy_with_logits_v2(labels = next_pick_mask, logits = logits_test)
    loss_cross_entropy_test = tf.reduce_mean(cross_entropies_test)
    predicted_pick_test = tf.slice(sm_logits_test, [0, 0, 0, 0], [-1, -1, -1, 1])




    with tf.variable_scope('second'):
        roi_logits = _build_deeplab(roi_wave_placeholder, {common.OUTPUT_TYPE: 1}, [64, 1], 64, is_training = True)
        
    with tf.variable_scope('second', reuse = True):
        roi_logits_test = _build_deeplab(roi_wave_placeholder, {common.OUTPUT_TYPE: 1}, [64, 1], 64, is_training = False)

    distance_l1 = tf.abs(roi_logits - roi_distance_map_placeholder)
    loss_l1 = tf.reduce_mean(distance_l1)

    distance_l1_test = tf.abs(roi_logits_test - roi_distance_map_placeholder)
    loss_l1_test = tf.reduce_mean(distance_l1_test)



        
    cross_entropy_loss_placeholder = tf.placeholder(tf.float32, ())
    l1_loss_placeholder = tf.placeholder(tf.float32, ())
    distance_first_mean_placeholder = tf.placeholder(tf.float32, ())
    distance_first_std_placeholder = tf.placeholder(tf.float32, ())
    distances_first_placeholder = tf.placeholder(tf.float32)
    distance_mean_placeholder = tf.placeholder(tf.float32, ())
    distance_std_placeholder = tf.placeholder(tf.float32, ())
    distances_placeholder = tf.placeholder(tf.float32)
    precision_placeholder = tf.placeholder(tf.float32, ())
    recall_placeholder = tf.placeholder(tf.float32, ())
    f1_score_placeholder = tf.placeholder(tf.float32, ())

    f1_score_first_placeholder = tf.placeholder(tf.float32, ())
    

    summaries_train = get_summaries('train', cross_entropy_loss_placeholder, l1_loss_placeholder, 
        distance_first_mean_placeholder, distance_first_std_placeholder, distances_first_placeholder, 
        distance_mean_placeholder, distance_std_placeholder, distances_placeholder, 
        precision_placeholder, recall_placeholder, f1_score_placeholder, f1_score_first_placeholder)

    summaries_validation = get_summaries('validation', cross_entropy_loss_placeholder, l1_loss_placeholder, 
        distance_first_mean_placeholder, distance_first_std_placeholder, distances_first_placeholder, 
        distance_mean_placeholder, distance_std_placeholder, distances_placeholder, 
        precision_placeholder, recall_placeholder, f1_score_placeholder, f1_score_first_placeholder)
    
    summaries_test = get_summaries('test', cross_entropy_loss_placeholder, l1_loss_placeholder, 
        distance_first_mean_placeholder, distance_first_std_placeholder, distances_first_placeholder, 
        distance_mean_placeholder, distance_std_placeholder, distances_placeholder, 
        precision_placeholder, recall_placeholder, f1_score_placeholder, f1_score_first_placeholder)

    learning_rate_placeholder = tf.placeholder(tf.float32, ())

    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'first')
    second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'second')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer_cross_entropy = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss_cross_entropy, var_list=first_train_vars)

    with tf.control_dependencies(update_ops):
        optimizer_l1 = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss_l1, var_list=second_train_vars)
    



    init = tf.global_variables_initializer()



    saver = tf.train.Saver()



    with tf.Session() as sess:

        sess.run(init)



        run_name = strftime("%Y-%m-%d %H:%M:%S", localtime())
        summary_writer = tf.summary.FileWriter('tensorboard/' + run_name, sess.graph, max_queue=0, flush_secs=0)
        
        best_validation_loss = 9999.0
        best_distance = 9999.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0

        base_learning_rate = FLAGS.base_learning_rate
        minimum_learning_rate = base_learning_rate * 0.01
        decay_factor = FLAGS.learning_rate_decay_factor


        for epoch in range(1000):
            
            learning_rate = base_learning_rate * (decay_factor ** (epoch / 2))

            if learning_rate < minimum_learning_rate:
                learning_rate = minimum_learning_rate


            train_metrics = run_epoch(sess, training_init_op, next_wave,
                [(optimizer_cross_entropy, {learning_rate_placeholder : learning_rate}), (optimizer_l1, {learning_rate_placeholder : learning_rate})], 
                cross_entropies, predicted_pick, next_pick, distance_l1, roi_logits, roi_wave_placeholder, roi_distance_map_placeholder, next_distance_map)

            validation_metrics = run_epoch(sess, validation_init_op, next_wave, None, cross_entropies_test, predicted_pick_test, next_pick, distance_l1_test, roi_logits_test,
                roi_wave_placeholder, roi_distance_map_placeholder, next_distance_map)

            test_metrics = run_epoch(sess, test_init_op, next_wave, None, cross_entropies_test, predicted_pick_test, next_pick, distance_l1_test, roi_logits_test,
                roi_wave_placeholder, roi_distance_map_placeholder, next_distance_map)



            write_summary(sess, summaries_train, summary_writer, epoch, 
                cross_entropy_loss_placeholder, train_metrics[0], 
                l1_loss_placeholder, train_metrics[1],
                distance_first_mean_placeholder, train_metrics[2], 
                distance_first_std_placeholder, train_metrics[3], 
                distances_first_placeholder, train_metrics[4],
                distance_mean_placeholder, train_metrics[5], 
                distance_std_placeholder, train_metrics[6], 
                distances_placeholder, train_metrics[7],
                precision_placeholder, train_metrics[8],
                recall_placeholder, train_metrics[9],
                f1_score_placeholder, train_metrics[10],
                f1_score_first_placeholder, train_metrics[13])

            write_summary(sess, summaries_validation, summary_writer, epoch, 
                cross_entropy_loss_placeholder, validation_metrics[0], 
                l1_loss_placeholder, validation_metrics[1],
                distance_first_mean_placeholder, validation_metrics[2], 
                distance_first_std_placeholder, validation_metrics[3], 
                distances_first_placeholder, validation_metrics[4],
                distance_mean_placeholder, validation_metrics[5], 
                distance_std_placeholder, validation_metrics[6], 
                distances_placeholder, validation_metrics[7],
                precision_placeholder, validation_metrics[8],
                recall_placeholder, validation_metrics[9],
                f1_score_placeholder, validation_metrics[10],
                f1_score_first_placeholder, validation_metrics[13])
            
            write_summary(sess, summaries_test, summary_writer, epoch, 
                cross_entropy_loss_placeholder, test_metrics[0], 
                l1_loss_placeholder, test_metrics[1],
                distance_first_mean_placeholder, test_metrics[2], 
                distance_first_std_placeholder, test_metrics[3], 
                distances_first_placeholder, test_metrics[4],
                distance_mean_placeholder, test_metrics[5], 
                distance_std_placeholder, test_metrics[6], 
                distances_placeholder, test_metrics[7],
                precision_placeholder, test_metrics[8],
                recall_placeholder, test_metrics[9],
                f1_score_placeholder, test_metrics[10],
                f1_score_first_placeholder, test_metrics[13])

            
            if validation_metrics[0] < best_validation_loss:
                best_validation_loss = validation_metrics[0]
                saver.save(sess, "best_validation_loss_model/model.ckpt")

            if validation_metrics[5] < best_distance:
                best_distance = validation_metrics[5]
                saver.save(sess, "best_validation_distance_model/model.ckpt")

            if validation_metrics[8] > best_precision:
                best_precision = validation_metrics[8]
                saver.save(sess, "best_validation_precision_model/model.ckpt")

            if validation_metrics[9] > best_recall:
                best_recall = validation_metrics[9]
                saver.save(sess, "best_validation_recall_model/model.ckpt")

            if validation_metrics[10] > best_f1:
                best_f1 = validation_metrics[10]
                saver.save(sess, "best_validation_f1_model/model.ckpt")
            

            

                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    tf.app.run(main=main, argv=[sys.argv[0]])
