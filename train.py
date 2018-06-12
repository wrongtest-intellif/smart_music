import random
import argparse
import logging
import librosa
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, path):
        self.data = []
        self.tag_map = {}
        self.max_length = 0
        with open(path) as data_file:
            for line in data_file:
                segs = line.strip().split("\t")
                filename, tags = segs[0], segs[1]

                wave, sr = librosa.load(filename, mono=True)
                mfcc = librosa.feature.mfcc(wave, sr)
         
                tags = tags.split(",")
                for tag in tags:
                    if tag not in self.tag_map:
                        self.tag_map[tag] = len(self.tag_map)
                self.data.append((filename, mfcc.shape[1], tags))
                self.max_length = max(self.max_length, mfcc.shape[1])
        random.shuffle(self.data)
        self.index = 0        

    def get_size(self):
        return len(self.data)

    def get_num_classes(self):
        return len(self.tag_map)

    def get_max_length(self):
        return self.max_length

    def get_batch(self, batch_size):
        label = np.zeros([batch_size, len(self.tag_map)], dtype="int64")
        feature_length = np.zeros([batch_size], dtype="int64")
        feature = np.zeros([batch_size, self.max_length, 20])
        for i in range(batch_size):
            filename, length, tags = self.data[self.index]

            self.index += 1
            if self.index >= len(self.data):
                random.shuffle(self.data)
                self.index = 0

            wave, sr = librosa.load(filename, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            feature[i][:length] = mfcc.transpose()
            feature_length[i] = length

            for tag in tags:
                label[i][self.tag_map[tag]] = 1
        return feature, feature_length, label


class ClassifyModel:

    def __init__(self, num_classes, input_feature_size):
        self.num_classes = num_classes
        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, None, input_feature_size])
        self.feature_length = tf.placeholder(dtype=tf.int64, shape=[None])
        self.label = tf.placeholder(dtype=tf.int64)
        self.logit = self.get_logit(self.feature, self.feature_length)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.label))
        self.optimizer = tf.train.AdamOptimizer(0.01)
        self.train_op = self.optimizer.minimize(self.loss)
        self.metrics = self.get_metric()

    def get_logit(self, feature, feature_length):
        feature = tf.transpose(feature, [1, 0, 2])
        (rnn1, rnn2), _ = tf.nn.bidirectional_dynamic_rnn(tf.nn.rnn_cell.GRUCell(128),
                                              tf.nn.rnn_cell.GRUCell(128),
                                              feature, feature_length,
                                              dtype=tf.float32,
                                              time_major=True)
        rnn = tf.concat([rnn1, rnn2], axis=2)
        pool = tf.reduce_max(tf.transpose(rnn, [1, 0, 2]), axis=1)
        with tf.variable_scope("classify"):
            w = tf.get_variable("classify_w", [128 * 2, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("classify_b", [self.num_classes],
                                initializer=tf.zeros_initializer())
            logits = tf.matmul(pool, w) + b
            logits = tf.nn.relu(logits)
        return logits

    def get_loss(self, logits, labels):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return cross_entropy

    def get_metric(self):
        batch_size = tf.cast(tf.shape(self.logit)[0], dtype=tf.int64)
        pred = tf.nn.softmax(self.logit)
        pred = tf.argmax(pred, axis=1)
        pred_range = tf.range(0, batch_size)
        indices = tf.stack([pred_range, pred], axis=1)
        accuracy_count = tf.reduce_sum(tf.gather_nd(self.label, indices))
        return {
            "accuracy_count": accuracy_count,
            "loss": self.loss * tf.cast(batch_size, tf.float32)
        }

    def run_train(self, sess, feature, feature_length, label):
        result, _ = sess.run([self.metrics, self.train_op],
                          {self.feature: feature,
                           self.feature_length: feature_length,
                           self.label: label})
        return result
    

def parse_args():
    parser = argparse.ArgumentParser(description='music')
    parser.add_argument('--train_path', help='training set path', required=True, type=str)
    parser.add_argument('--checkpoint_dir', help='checkpoint directory path', default=None, type=str)
    parser.add_argument('--epoch_number', help='end epoch of training', default=10, type=int)
    parser.add_argument('--batch_size', help='batch size', default=8, type=float)
    parser.add_argument('--learning_rate', help='base learning rate', default=0.01, type=float)
    parser.add_argument('--saved_model_dir', help='directory of saved model', default=None, type=str)
    parser.add_argument('--train_report_steps', help='steps to print train metric', default=1, type=int)
    parser.add_argument('--train_checkpoint_steps', help='steps to save model', default=1, type=int)
    args = parser.parse_args()
    return args

    
def main(args):
    data_loader = DataLoader(args.train_path)

    logging.info("Start building network...")
    net = ClassifyModel(data_loader.get_num_classes(), 20)

    step_size = args.epoch_number * data_loader.get_size() / args.batch_size
    logging.info("Start training with %d steps.." % step_size)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        cur_loss = 0.
        cur_accuracy_count = 0.
        cur_samples = 0
        for step_id in range(1, step_size + 1):
            data, length, label = data_loader.get_batch(args.batch_size)
            metrics = net.run_train(sess, data, length, label)

            cur_loss += metrics["loss"]
            cur_accuracy_count += metrics["accuracy_count"]
            cur_samples += data.shape[0]

            if step_id % args.train_report_steps == 0:
                logging.info("[Step %d] loss=%.3f accuracy=%.3f" % (
                   step_id, cur_loss / cur_samples, cur_accuracy_count / cur_samples))
                cur_accuracy_count = 0.
                cur_loss = 0.
                cur_samples = 0

            if args.checkpoint_dir is not None and step_id % args.train_checkpoint_steps == 0:
                logging.info("[Step %d] Dump checkpoint..." % step_id)
                saver.save(sess, args.checkpoint_dir + "/model-" + str(step_id))



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main(parse_args())
