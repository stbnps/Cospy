
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import glob
from random import shuffle
import gc


FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(waves, times, writer):

    for i in range(len(waves)):


        if np.isnan(np.sum(waves[i])):
            continue
            
        wave_raw = waves[i].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'pick': _int64_feature(int(times[i])),
                    'wave_raw': _bytes_feature(wave_raw)
                }))
        writer.write(example.SerializeToString())


def main(unused_argv):
    data_folder = FLAGS.datadir
    waves_filenames = glob.glob(data_folder + 'waves*.npy')
    times_filenames = glob.glob(data_folder + 'times*.npy')

    waves_filenames.sort()
    times_filenames.sort()


    writer_train = tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfdir, 'train.tfrecords'))
    writer_validation = tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfdir, 'validation.tfrecords'))
    writer_test = tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfdir, 'test.tfrecords'))

    n = 0

    for i in range(len(waves_filenames)):
        waves = np.load(waves_filenames[i])
        times = np.load(times_filenames[i])

        waves = waves[np.abs(times - 6000) < 3]
        times = times[np.abs(times - 6000) < 3]

        n += waves.shape[0]

        waves = waves.astype(np.float32)
        times = times.astype(np.int64)



        waves = np.rollaxis(waves, 2, 1)
        waves = np.expand_dims(waves, 2)


        waves_mean = np.mean(waves, (1,2))
        waves_mean = np.expand_dims(np.expand_dims(waves_mean,1),1)

        
        waves -= waves_mean


        waves_std = np.std(waves, (1,2))
        waves_std = np.expand_dims(np.expand_dims(waves_std,1),1)

        waves /= (waves_std + 0.000001)


        indices = range(len(waves))

        shuffle(indices)
        waves = waves[indices]
        times = times[indices]

        waves_train = waves[ : int(0.8*(len(waves) - 1))]
        waves_validation = waves[int(0.8*(len(waves) - 1)) : int(0.9*(len(waves) - 1))]
        waves_test = waves[int(0.9*(len(waves) - 1)) :]

        times_train = times[ : int(0.8*(len(waves) - 1))]
        times_validation = times[int(0.8*(len(waves) - 1)) : int(0.9*(len(waves) - 1))]
        times_test = times[int(0.9*(len(waves) - 1)) :]


        convert_to(waves_train, times_train, writer_train)
        convert_to(waves_validation, times_validation, writer_validation)
        convert_to(waves_test, times_test, writer_test)
    
    print(n)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tfdir',
        type=str,
        default='tfrecords/',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='data/',
        help='Directory to download data files and write the converted result'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
