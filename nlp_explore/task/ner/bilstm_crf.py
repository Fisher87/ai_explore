import math
import helper
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
np.set_printoptions(threshold=np.nan)
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.orthogonal_initializer(dtype=tf.float32)(shape = shape)


class BILSTM_CRF(object):

    def __init__(self, num_chars, num_slot_class, num_intent_classes, num_steps=30,
                 num_epochs=100, embedding_matrix=None, is_training=True, crf_flag=3, weight=False,
                 inputs=None, input_tag=None):
        # Parameter
        self.max_f1 = 0
        self.max_acc = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 128
        self.num_layers = 1
        self.emb_dim = 200
        self.hidden_dim = 100
        self.input_tag_size = 9
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_slot_class = num_slot_class
        self.num_intent_class = num_intent_classes
        self.crf_flag = crf_flag

        # placeholder of x, y and weight
        if is_training==False:
            self.inputs = tf.reshape(inputs, shape=[-1, self.num_steps])
        else:
            self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])

        # self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.slot_targets = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_steps])
        self.targets_transition = tf.placeholder(tf.int32, [None])
        #self.sequence_len = tf.placeholder(tf.int32, [None])
        self.intent_target = tf.placeholder(tf.int32, [None])

        # char embedding
        if embedding_matrix is None:
            print("embedding_matrix is None")
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        else:
            print("embedding_matrix is not None")
            self.embedding = tf.Variable(embedding_matrix, trainable=True, name="emb", dtype=tf.float32)

        if is_training==False:
            self.input_tag = tf.reshape(input_tag, shape=[-1, self.num_steps, self.input_tag_size])
        else:
            self.input_tag = tf.placeholder(tf.float32, [None, self.num_steps, self.input_tag_size])

        # self.input_tag = tf.placeholder(tf.float32, [None, self.num_steps, self.input_tag_size])
        self.input_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.input_emb = tf.concat(axis=2, values=[self.input_emb, self.input_tag])

        # self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        # self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim+self.tag_size])
        # self.inputs_emb = tf.split(axis=0, num_or_size_splits=self.num_steps, value=self.inputs_emb)

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim,initializer=_initializer)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim,initializer=_initializer)

        # dropout
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.sequence_len = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.sequence_len = tf.cast(self.sequence_len, tf.int32)

        # forward and backward
        self.outputs, _ = rnn.bidirectional_dynamic_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.input_emb,
            dtype=tf.float32,
            sequence_length=self.sequence_len
        )

        rnn_output1 = self.outputs[1][:,0,:]
        #rnn_output0 = helper.collect_final_step_of_lstm(self.outputs[0],self.sequence_len-1)
        self.intent_input = rnn_output1 # tf.concat(axis=1, values=[rnn_output0,rnn_output1])
        #self.intent_input=tf.reduce_max(self.outputs[0],1)
        dense_layer = tf.layers.Dense(units=self.num_intent_class, name="intent")
        self.intent_logits = dense_layer(self.intent_input)

        self.intent_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.intent_logits, labels=self.intent_target))
        self.intent_prediction = tf.argmax(self.intent_logits, 1, name="predictions")

        self.outputs = tf.concat(axis=-1, values=self.outputs)
        # self.outputs = tf.reshape(tf.concat(axis=1, values=self.outputs), [-1, self.hidden_dim * 2])
        dense_layer2 = tf.layers.Dense(units=self.num_slot_class, name="tag")
        self.slot_logits = dense_layer2(self.outputs)

        if crf_flag==1:
            self.slot_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.slot_logits, self.slot_targets))
        elif crf_flag==2:
            self.tags_scores = tf.reshape(self.slot_logits, [self.batch_size, self.num_steps, self.num_slot_class])
            self.transitions = tf.get_variable("transitions", [self.num_slot_class+1, self.num_slot_class+1])

            dummy_val = -1000
            class_pad = dummy_val * tf.ones([self.batch_size, self.num_steps, 1], tf.float32)

            self.observations = tf.concat(axis=2, values=[self.tags_scores, class_pad])

            W = tf.ones([self.batch_size, 1], tf.float32)
            begin_vec = tf.constant([dummy_val] * self.num_slot_class + [0], dtype=tf.float32)
            begin_vec = tf.multiply(W, begin_vec)

            end_vec = tf.constant([0] + [dummy_val] * self.num_slot_class, dtype=tf.float32)
            end_vec = tf.multiply(W, end_vec)

            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_slot_class+1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_slot_class+1])

            self.observations = tf.concat(axis=1, values=[begin_vec, self.observations, end_vec])

            self.mask = tf.cast(tf.reshape(tf.sign(self.slot_targets), [self.batch_size * self.num_steps]), tf.float32)

            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_slot_class + tf.reshape(self.slot_targets, [self.batch_size * self.num_steps]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)

            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre  = self.forward(self.observations, self.transitions, self.sequence_len)

            # loss
            self.slot_loss = - (self.target_path_score - self.total_path_score)
        elif crf_flag==3:
            slot_loss, self.transition_params = crf_log_likelihood(inputs=self.slot_logits, tag_indices=self.slot_targets, sequence_lengths=self.sequence_len)
            self.slot_loss = -tf.reduce_mean(slot_loss)
        self.sum_loss= self.slot_loss + 200 * self.intent_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.sum_loss)

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keepdims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat(axis=0, values=[transitions] * self.batch_size), [self.batch_size, self.num_slot_class+1, self.num_slot_class+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_slot_class+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_slot_class+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_slot_class+1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_slot_class+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t

        alphas = tf.reshape(tf.concat(axis=0, values=alphas), [self.num_steps + 2, self.batch_size, self.num_slot_class+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_slot_class+1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_slot_class+1, 1])

        max_scores = tf.reshape(tf.concat(axis=0, values=max_scores), (self.num_steps + 1, self.batch_size, self.num_slot_class+1))
        max_scores_pre = tf.reshape(tf.concat(axis=0, values=max_scores_pre), (self.num_steps + 1, self.batch_size, self.num_slot_class+1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre

    def train(self, sess, saver, save_file, X_train, y_train, X_valid, y_valid, X_train_tag,
              X_valid_tag, y_intent_train, y_intent_valid, model_dev, seq_len_train, seq_len_valid):

        char2id, id2char = helper.loadMap("meta_data/char2id")
        label2id, id2label = helper.loadMap("meta_data/label2id")

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        max_f1 = 0.0
        max_intent_acc = 0.0

        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            shuffle_index = np.arange(len(X_train))
            np.random.shuffle(shuffle_index)
            X_train = X_train[shuffle_index]
            seq_len_train = seq_len_train[shuffle_index]
            y_train = y_train[shuffle_index]
            X_train_tag = X_train_tag[shuffle_index]
            y_intent_train = y_intent_train[shuffle_index]
            print("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                # train
                X_train_batch, y_train_batch, X_train_tag_batch, y_intent_train_batch, seq_len_batch_train = \
                    helper.nextBatch(X_train, y_train, X_train_tag, y_intent_train, seq_len_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)

                y_train_weight_batch = 1 + np.array((y_train_batch == label2id['B']) | (y_train_batch == label2id['E']) | (y_train_batch == label2id['X']) | (y_train_batch == label2id['Z']) | (y_train_batch == label2id['U']) | (y_train_batch == label2id['W']), float)
                transition_batch = helper.get_transition(y_train_batch)

                if self.crf_flag == 2:
                    _, loss_train, max_scores, max_scores_pre, predicts_train_intent, len_train  = \
                    sess.run([
                        self.optimizer,
                        self.sum_loss,
                        self.max_scores,
                        self.max_scores_pre,
                        self.intent_prediction,
                        self.sequence_len,
                    ],
                        feed_dict={
                            self.targets_transition:transition_batch,
                            self.inputs:X_train_batch,
                            self.slot_targets:y_train_batch,
                            self.targets_weight:y_train_weight_batch,
                            self.input_tag:X_train_tag_batch,
                            self.intent_target:y_intent_train_batch,
                         #   self.sequence_len:seq_len_batch_train
                        })

                    if iteration % 100 == 0:
                        predicts_train = self.viterbi(max_scores, max_scores_pre, len_train, predict_size=self.batch_size)
                        precision_train, recall_train, f1_train, acc_train = self.evaluate(X_train_batch, y_train_batch, y_intent_train_batch, predicts_train, predicts_train_intent, id2char, id2label)
                        print("iteration, train loss, train precision, train recall, train f1, train acc",iteration, loss_train, precision_train, recall_train, f1_train, acc_train)
                elif self.crf_flag==3:
                    _, transition_params_train, slot_train_logits, loss_train, predicts_train_intent, train_seq_length = \
                        sess.run([
                            self.optimizer,
                            self.transition_params,
                            self.slot_logits,
                            self.sum_loss,
                            self.intent_prediction,
                            self.sequence_len,
                        ],
                            feed_dict={
                                # self.targets_transition: transition_batch,
                                self.inputs: X_train_batch,
                                self.slot_targets: y_train_batch,
                                # self.targets_weight: y_train_weight_batch,
                                self.input_tag: X_train_tag_batch,
                                self.intent_target: y_intent_train_batch,
                         #       self.sequence_len: seq_len_batch_train
                            })
                    if iteration % 100 == 0:
                        label_list = []
                        for logit, seq_len in zip(slot_train_logits, train_seq_length):
                            if seq_len == 0:  # padding 0 at last of the data
                                break
                            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params_train)
                            label_list.append(viterbi_seq)
                        predicts_train = label_list
                        precision_train, recall_train, f1_train, acc_train = self.evaluate(X_train_batch, y_train_batch,
                                                                                   y_intent_train_batch, predicts_train,
                                                                                   predicts_train_intent, id2char,
                                                                                   id2label)
                        print("iteration, train loss, train precision, train recall, train f1, train acc", iteration,
                        loss_train, precision_train, recall_train, f1_train, acc_train)

                # validation
                if iteration % 200 == 0:
                    f1_valid_sum = 0.0
                    acc_valid_sum = 0.0
                    loss_valid_sum = 0.0
                    precision_valid_sum = 0.0
                    recall_valid_sum = 0.0
                    num_iterations_valid = int(math.ceil(1.0 * len(X_valid) / model_dev.batch_size))
                    for ttt in range(num_iterations_valid):
                        X_valid_batch, y_valid_batch, X_valid_input_tag_batch, y_intent_valid_batch, seq_len_valid_batch = \
                            helper.nextBatch(X_valid, y_valid, X_valid_tag, y_intent_valid, seq_len_valid, start_index=ttt * model_dev.batch_size, batch_size=model_dev.batch_size)

                        y_val_weight_batch = 1 + np.array((y_valid_batch == label2id['B']) | (y_valid_batch == label2id['E']) | (y_valid_batch == label2id['X']) | (y_valid_batch == label2id['Z']) | (y_valid_batch == label2id['U']) | (y_valid_batch == label2id['W']), float)
                        transition_batch = helper.get_transition(y_valid_batch)
                        if self.crf_flag==2:
                            loss_valid, max_scores, max_scores_pre, predicts_valid_intent, length_dev = \
                            sess.run([
                                model_dev.sum_loss,
                                model_dev.max_scores,
                                model_dev.max_scores_pre,
                                model_dev.intent_prediction,
                                model_dev.sequence_len,
                            ],
                                feed_dict={
                                    model_dev.targets_transition:transition_batch,
                                    model_dev.inputs:X_valid_batch,
                                    model_dev.slot_targets:y_valid_batch,
                                    model_dev.targets_weight:y_val_weight_batch,
                                    model_dev.input_tag:X_valid_input_tag_batch,
                                    model_dev.intent_target:y_intent_valid_batch,
                                    # model_dev.sequence_len:seq_len_valid_batch
                                })
                            predicts_valid = model_dev.viterbi(max_scores, max_scores_pre, length_dev,
                                                           predict_size=model_dev.batch_size)
                        elif self.crf_flag==3:
                            slot_train_logits, transition_params_train, length_dev, intent_prediction, loss_valid = \
                                sess.run([model_dev.slot_logits,
                                          model_dev.transition_params,
                                          model_dev.sequence_len,
                                          model_dev.intent_prediction,
                                          model_dev.sum_loss],
                                                                 feed_dict={
                                    # model_dev.targets_transition:transition_batch,
                                    model_dev.inputs:X_valid_batch,
                                    model_dev.slot_targets:y_valid_batch,
                                    # model_dev.targets_weight:y_val_weight_batch,
                                    model_dev.input_tag:X_valid_input_tag_batch,
                                    model_dev.intent_target:y_intent_valid_batch,
                                    # model_dev.sequence_len:seq_len_valid_batch
                                })
                            label_list = []
                            for logit, seq_len in zip(slot_train_logits, length_dev):
                                if seq_len == 0:  # padding 0 at last of the data
                                    break
                                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params_train)
                                label_list.append(viterbi_seq)
                            predicts_valid = label_list
                            predicts_valid_intent = intent_prediction

                        precision_valid, recall_valid, f1_valid, acc_valid = \
                            model_dev.evaluate(X_valid_batch, y_valid_batch, y_intent_valid_batch, predicts_valid, predicts_valid_intent, id2char, id2label)

                        f1_valid_sum += f1_valid
                        acc_valid_sum += acc_valid
                        loss_valid_sum += loss_valid
                        precision_valid_sum += precision_valid
                        recall_valid_sum += recall_valid
                    if f1_valid_sum>max_f1:
                        max_f1=f1_valid_sum
                        saver.save(sess, "predict_output/model")
                    if acc_valid_sum>max_intent_acc:
                        max_intent_acc=acc_valid_sum
                        # saver.save(sess, "predict_output/model")
                    print("iteration, valid loss, valid precision, valid recall, valid f1, valid acc", iteration,
                          loss_valid_sum/num_iterations_valid, precision_valid_sum/num_iterations_valid,
                          recall_valid_sum/num_iterations_valid, f1_valid_sum/num_iterations_valid, acc_valid_sum/num_iterations_valid)

        print("max slot f1:", max_f1/num_iterations_valid)
        print("max intent acc", max_intent_acc/num_iterations_valid)

    def test(self, sess, X_test, X_test_str, X_test_tag, y_intent_test, y_test, output_path):
        char2id, id2char = helper.loadMap("meta_data/char2id")
        label2id, id2label = helper.loadMap("meta_data/label2id")
        intentlabel2id, intentid2label = helper.loadMap("meta_data/intentlabel2id")
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print("number of iteration: " + str(num_iterations))
        correct=0
        count=0
        print(len(y_test))
        print(len(X_test))
        out_intent_file = open("test_output/test_y_intent_out","w",encoding="utf-8")
        with open(output_path, mode = "w", encoding="utf-8") as outfile:
            total_f1 = 0
            total_p = 0
            total_r = 0
            total_acc = 0
            for i in range(num_iterations-1,-1,-1):
                #correct=0
                #count=0
                # print("iteration: " + str(i + 1))
                #results = []
                results_BME = []
                results_XYZ = []
                X_test_batch = X_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_test_str_batch = X_test_str[i * self.batch_size : (i + 1) * self.batch_size]
                X_test_tag_batch = X_test_tag[i * self.batch_size : (i + 1) * self.batch_size]
                y_intent_test_batch = y_intent_test[i * self.batch_size : (i + 1) * self.batch_size]
                y_test_batch = y_test[i * self.batch_size: (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_test_str_batch = list(X_test_str_batch)
                    X_test_tag_batch = list(X_test_tag_batch)
                    y_intent_test_batch = list(y_intent_test_batch)
                    y_test_batch = list(y_test_batch)

                    last_size = len(X_test_batch)

                    X_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_str_batch += [['x' for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_tag_batch += [[[0, 0, 0, 0, 0, 0, 0, 0, 0] for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    y_intent_test_batch += [0 for i in range(self.batch_size - last_size)]
                    y_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]

                    X_test_batch = np.array(X_test_batch)
                    X_test_str_batch = np.array(X_test_str_batch)
                    X_test_tag_batch = np.array(X_test_tag_batch)
                    y_intent_test_batch = np.array(y_intent_test_batch)
                    y_test_batch = np.array(y_test_batch)

                    results_BME, results_XYZ, results_UVW, y_predictions, correct_batch, count_batch, slot_precision_batch,slot_recall_batch,slot_f1_batch = \
                        self.predict_batch(sess, X_test_batch, X_test_str_batch, X_test_tag_batch, y_intent_test_batch, y_test_batch, id2label, id2char)
                    correct+=correct_batch
                    count+=count_batch
                    acc = 1.0 * correct / count
                    results_BME = results_BME[:last_size]
                    results_XYZ = results_XYZ[:last_size]
                    results_UVW = results_UVW[:last_size]
                    total_f1 += slot_f1_batch
                    total_p += slot_precision_batch
                    total_r += slot_recall_batch
                    total_acc += acc
                else:
                    X_test_batch = np.array(X_test_batch)
                    X_test_tag_batch = np.array(X_test_tag_batch)
                    y_intent_test_batch = np.array(y_intent_test_batch)
                    y_test_batch = np.array(y_test_batch)
                    results_BME, results_XYZ, results_UVW, y_predictions, correct_batch, count_batch,slot_precision_batch,slot_recall_batch,slot_f1_batch =\
                        self.predict_batch(sess, X_test_batch, X_test_str_batch, X_test_tag_batch, y_intent_test_batch, y_test_batch, id2label, id2char)
                    correct+=correct_batch
                    count+=count_batch
                    acc=1.0* correct/count
                    total_f1 += slot_f1_batch
                    total_p += slot_precision_batch
                    total_r += slot_recall_batch
                    total_acc += acc
            # print("test intent acc: ", total_acc/num_iterations)
            # print("test slot precision: ", total_p/num_iterations)
            # print("test slot recall: ", total_r/num_iterations)
            # print("test slot f1: ", total_f1/num_iterations)
                for j in range(len(y_predictions)):
                    doc = ''.join(X_test_str_batch[j])
                    if len(doc)>=1 and doc[0]!= 'x':
                        out_intent_file.write(doc + "<@>" + intentid2label[y_predictions[j]]+"\n")
                for i in range(len(results_BME)):
                    doc = ''.join(X_test_str_batch[i])
                    outfile.write(doc + "<@>" +" ".join(results_BME[i]) + "<@>" + " ".join(results_XYZ[i])  + "<@>" + " ".join(results_UVW[i]) + "\n")
                    #outfile.write(doc + "<@>" +" ".join(results[i]).encode("utf-8") + "\n")

            print("test intent acc: ", total_acc / num_iterations)
            print("test slot precision: ", total_p / num_iterations)
            print("test slot recall: ", total_r / num_iterations)
            total_f1 = 2.0*total_p*total_r/(total_r+total_p)
            print("test slot f1: ", total_f1 / num_iterations)

    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            # last_max_node = 0
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths

    def predict_batch(self, sess, X, X_str, X_tag, y_intent, y_test, id2label, id2char):
        results_BME = []
        results_XYZ = []
        results_UVW = []
        #print("X",X)
        #print("X_str",X_str)
        #print("X_tag",X_tag)
        #print("y_intent",y_intent)
        slot_precision = -1.0
        slot_recall = -1.0
        slot_f1 = -1.0
        hit_num = 0.0
        pred_num = 0.0
        true_num = 0.0
        if self.crf_flag==2:
            length, max_scores, max_scores_pre, intent_prediction = sess.run(
                [self.sequence_len, self.max_scores, self.max_scores_pre, self.intent_prediction],
                            feed_dict={self.inputs:X, self.input_tag:X_tag, self.intent_target:y_intent})
            predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
        elif self.crf_flag==3:
            slot_train_logits, transition_params_train, length_dev, intent_prediction = \
                sess.run([self.slot_logits,
                          self.transition_params,
                          self.sequence_len,
                          self.intent_prediction,
                          # self.sum_loss
                          ],
                         feed_dict={
                             # model_dev.targets_transition:transition_batch,
                             self.inputs: X,
                             # self.slot_targets: y_valid_batch,
                             # model_dev.targets_weight:y_val_weight_batch,
                             self.input_tag: X_tag,
                             self.intent_target: y_intent,
                             # model_dev.sequence_len:seq_len_valid_batch
                         })
            label_list = []
            for logit, seq_len in zip(slot_train_logits, length_dev):
                if seq_len==0: # padding 0 at last of the data
                    break
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params_train)
                label_list.append(viterbi_seq)
            predicts = label_list

        for i in range(len(predicts)):
            x = ''.join(X_str[i])
            x_ = [str(id2char[val]) for val in X[i]]
            y = [str(id2label[val]) for val in y_test[i]]
            y_pred_str = ''.join([id2label[val] for val in predicts[i] if val != 11 and val != 0])
            y_hat = [id2label[val] for val in predicts[i]]
            true_labels = helper.extract_entity(x_, y)
            pred_labels = helper.extract_entity(x_, y_hat)
            if False:#len(true_labels) == 0 and len(pred_labels) == 0:
                hit_num += 1
                pred_num += 1
                true_num += 1
            else:
                hit_num += len(set(true_labels) & set(pred_labels))
                pred_num += len(set(pred_labels))
                true_num += len(set(true_labels))

            #entitys = helper.extractEntity(x, y_pred)
            entitys_BME = helper.extractEntity_BME(x, y_pred_str)
            entitys_XYZ = helper.extractEntity_XYZ(x, y_pred_str)
            entitys_UVW = helper.extractEntity_UVW(x, y_pred_str)
            #results.append(entitys)
            results_BME.append(entitys_BME)
            results_XYZ.append(entitys_XYZ)
            results_UVW.append(entitys_UVW)
            correct=0
            count=0
            for j in range(len(intent_prediction)):
                # print("predictions",predictions[j])
                # print("y_intent",y_intent[j])
                if intent_prediction[j]==y_intent[j] and y_intent[j]!=0:
                    correct+=1
                if y_intent[j]!=0:
                    count+=1
                # else:
                #     print("it is padding")
        if pred_num != 0.0:
            slot_precision = 1.0 * hit_num / pred_num
        if true_num != 0.0:
            slot_recall = 1.0 * hit_num / true_num
        if slot_precision > 0 and slot_recall > 0:
            slot_f1 = 2.0 * (slot_precision * slot_recall) / (slot_precision + slot_recall)
        return results_BME, results_XYZ, results_UVW, intent_prediction, correct, count ,slot_precision,slot_recall,slot_f1

    def evaluate(self, X, y_true, y_intent_true, y_pred, y_intent_pred, id2char, id2label):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        hit_num = 0.0
        pred_num = 0.0
        true_num = 0.0
        correct = 0.0
        acc = 0.0
        try:
            for i in range(len(y_true)):
                x = [str(id2char[val]) for val in X[i]]
                y = [str(id2label[val]) for val in y_true[i]]
                y_hat = [id2label[val] for val in y_pred[i]]
                true_labels = helper.extract_entity(x, y)
                pred_labels = helper.extract_entity(x, y_hat)
                if False:#len(true_labels)==0 and len(pred_labels)==0:
                    hit_num += 1
                    pred_num += 1
                    true_num += 1
                else:
                    hit_num += len(set(true_labels) & set(pred_labels))
                    pred_num += len(set(pred_labels))
                    true_num += len(set(true_labels))
        except:
            print(y_pred)
        if pred_num != 0.0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0.0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        for i in range(len(y_intent_true)):
            if y_intent_true[i]==y_intent_pred[i]:
                correct+=1
            acc = 1.0*(correct)/len(y_intent_pred)
        return precision, recall, f1, acc

#
# model = BILSTM_CRF(num_chars=100, num_classes=4, num_intent_classes=4,
# 			   num_steps=100, num_epochs=4, embedding_matrix=None, is_training=True)
