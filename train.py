import random
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

    def get_num_classes(self):
        return len(self.tag_map)

    def get_max_length(self):
        return self.max_length

    def get_batch(self, batch_size):
        assert batch_size < len(self.data)
        if self.index + batch_size >= len(self.data):
            random.shuffle(self.data)
            self.index = 0

        label = np.zeros([batch_size, len(self.tag_map)], dtype="int64")
        feature_length = np.zeros([batch_size], dtype="int64")
        feature = np.zeros([batch_size, self.max_length, 20])
        for i in range(batch_size):
            filename, length, tags = self.data[self.index]
            self.index += 1

            wave, sr = librosa.load(filename, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            feature[i][:length] = mfcc.transpose()
            feature_length[i] = length

            for tag in tags:
                label[i][self.tag_map[tag]] = 1
        return feature, feature_length, label


class ClassifyModel():
    def __init__(self, num_classes, input_feature_size):
        self.num_classes = num_classes
        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, None, input_feature_size])
        self.feature_length = tf.placeholder(dtype=tf.int64, shape=[None])
        self.label = tf.placeholder(dtype=tf.int64)
        self.logit = self.get_logit(self.feature, self.feature_length)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit, labels=self.label))
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)

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

    def run_train(self, sess, feature, feature_length, label):
        return sess.run(self.train_op, {self.feature: feature,
                                        self.feature_length: feature_length,
                                        self.label: label})
    

def main():
    data_loader = DataLoader("sample.csv")
    
    net = ClassifyModel(data_loader.get_num_classes(), 20)

    data, length, label = data_loader.get_batch(2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.run_train(sess, data, length, label)


if __name__ == "__main__":
    main()
