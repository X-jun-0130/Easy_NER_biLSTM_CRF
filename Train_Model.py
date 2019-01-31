import os
from Parameters import Parameters as pm
from biLstm_Crf import LSTM_CRF
import tensorflow as tf
from Data_process import sequence2id, process_seq, batch_iter

def train():
    tensorboard_dir = './tensorboard/biLstm_crf'
    save_dir = './checkpoints/biLstm_crf'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    content_train, label_train = sequence2id(pm.train_data)
    content_test, label_test = sequence2id(pm.test_data)

    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch+1)
        num_batchs = int((len(content_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(content_train, label_train)
        for x_batch, y_batch in batch_train:
            x_batch, seq_leng_x = process_seq(x_batch)
            y_batch, seq_leng_y = process_seq(y_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_leng_x, pm.keep_pro)
            _, global_step, loss, tain_summary = session.run([model.optimizer, model.global_step, model.loss, merged_summary],
                                                             feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss = model.test(session, content_test, label_test)
                print('global_step:', global_step, 'train_loss:', loss, 'test_loss:', test_loss)

            if global_step % (4*num_batchs) == 0:
                print('Saving Model...')
                saver.save(session, save_path=save_path, global_step=global_step)
        pm.learning_rate *= pm.lr


if __name__ == '__main__':
    pm = pm
    model = LSTM_CRF()
    train()