import tensorflow as tf
from Parameters import Parameters as pm
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from Data_process import batch_iter, process_seq


class LSTM_CRF(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.Model()

    def Model(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding_ = tf.Variable(tf.truncated_normal([pm.vocab_size, pm.embedding_size], -0.25, 0.25), name='w')
            embedding = tf.nn.embedding_lookup(embedding_, self.input_x)
            self.embedding = tf.nn.dropout(embedding, pm.keep_pro)

        with tf.name_scope('biLSTM'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim)
            outputs, outstates = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,inputs=self.embedding,
                                                                 sequence_length=self.seq_length, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)#将双向RNN的结果进行拼接
            #outputs三维张量，[batchsize,seq_length,2*hidden_dim],

        with tf.name_scope('output'):
            s = tf.shape(outputs)
            output = tf.reshape(outputs, [-1, 2*pm.hidden_dim])
            output = tf.layers.dense(output, pm.num_tags)
            output = tf.contrib.layers.dropout(output, pm.keep_pro)
            self.logits = tf.reshape(output, [-1, s[1], pm.num_tags])

        with tf.name_scope('crf'):
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.input_y, sequence_lengths=self.seq_length)
            # log_likelihood是对数似然函数，transition_params是转移概率矩阵
            #crf_log_likelihood{inputs:[batch_size,max_seq_length,num_tags],
                                #tag_indices:[batchsize,max_seq_length],
                                #sequence_lengths:[real_seq_length]
            #transition_params: A [num_tags, num_tags] transition matrix
            #log_likelihood: A scalar containing the log-likelihood of the given sequence of tag indices.

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-log_likelihood) #最大似然取负，使用梯度下降

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variable = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variable), global_step=self.global_step)
            #self.optimizer = optimizer.minimize(optimizer, global_step= self.global_step)


    def feed_data(self, x_batch, y_batch, seq_length, keep_pro):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_length,
                     self.keep_pro: keep_pro
                     }
        return feed_dict

    def test(self, sess, x, y):
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            x_pad, seq_length_x = process_seq(x_batch)
            y_pad, seq_length_y = process_seq(y_batch)
            feed_dict = self.feed_data(x_pad, y_pad, seq_length_x, 1.0)
            loss = sess.run(self.loss, feed_dict=feed_dict)

            return loss

    def predict(self, sess, seqs):
        seq_pad, seq_length = process_seq(seqs)
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict={self.input_x: seq_pad,
                                                                                               self.seq_length: seq_length,
                                                                                               self.keep_pro: 1.0})
        label_ = []
        for logit, length in zip(logits, seq_length):
            #logit 每个子句的输出值，length子句的真实长度，logit[:length]的真实输出值
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = viterbi_decode(logit[:length], transition_params)
            label_.append(viterbi_seq)
        return label_




