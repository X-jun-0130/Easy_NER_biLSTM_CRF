import pickle
from Parameters import Parameters as pm
from biLstm_Crf import LSTM_CRF
import tensorflow as tf

#将句子转换为字序列
def get_word(sentence):
    word_list = []
    sentence = ''.join(sentence.split(' '))
    for i in sentence:
        word_list.append(i)
    return word_list

def read_file(filename):
    content = []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        content.append(word_list)
    return content

def sequence2id(filename):
    '''
    :param filename:
    :return: 将文字，转换为数字
    '''
    content2id = []
    content = read_file(filename)
    with open('./data/word2id.pkl', 'rb') as fr:
        word = pickle.load(fr)
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key.isdigit():
                key = '<NUM>'
            elif key not in word:
                key = '<UNK>'
            w.append(word[key])
        content2id.append(w)
    return content2id

def convert(sentence, label_line):
    word_dict = {}
    wordlist = get_word(sentence)
    label = [k.astype(str) for k in label_line]
    label.append('0') #防止最后一位不为0
    for i in range(len(label)):
        if label[i] == '1':
            n = label.index('0', i)
            word = ''.join(wordlist[i:n])
            word_dict[word] = 'PER'
        elif label[i] == '3':
            n = label.index('0', i)
            word = ''.join(wordlist[i:n])
            word_dict[word] = 'LOC'
        elif label[i] == '5':
            n = label.index('0', i)
            word = ''.join(wordlist[i:n])
            word_dict[word] = 'ORG'

    return word_dict


def val():
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/biLstm_crf')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    content = sequence2id(pm.eva)
    pre_label = model.predict(session, content)
    label.extend(pre_label)

    return label


if __name__ == '__main__':
    pm = pm
    model = LSTM_CRF()

    label = val()
    with open(pm.eva, 'r', encoding='utf-8') as f:
        sentences = [line.strip('\n') for line in f]

    for i in range(len(sentences)):
        word_dict = convert(sentences[i], label[i])
        print(sentences[i])
        print(word_dict)



