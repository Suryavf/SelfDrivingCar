import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from ReinforcementLearning.network.base import  BaseModel

def conv2d_layer(x, output_dim, kernel_size, stride, initializer=None, padding="VALID", data_format="NCHW",
                 summary_tag=None,
                 scope_name="conv2d", activation=tf.nn.relu):
    with tf.variable_scope(scope_name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=tf.truncated_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

        if activation != None:
            out = activation(out)
        summary = None
        if summary_tag is not None:
            # TODO general definitions
            if output_dim == 32:
                ix = 4
                iy = 8
            elif output_dim == 64:
                ix = 8
                iy = 8

            img = tf.slice(out, [0, 0, 0, 0], [1, -1, -1, -1])
            if data_format == "NCHW":
                img = tf.transpose(img, [0, 2, 3, 1])
            out_shape = img.get_shape().as_list()
            img = tf.reshape(img, [out_shape[1], out_shape[2], out_shape[3]])
            out_shape[1] += 4
            out_shape[2] += 4
            img = tf.image.resize_image_with_crop_or_pad(img, out_shape[1], out_shape[2])
            img = tf.reshape(img, [out_shape[1], out_shape[2], ix, iy])
            img = tf.transpose(img, [2, 0, 3, 1])
            img = tf.reshape(img, [1, ix * out_shape[1], iy * out_shape[2], 1])
            summary = tf.summary.image(summary_tag, img)
        return w, b, out, summary


def fully_connected_layer(x, output_dim, scope_name="fully", initializer=tf.random_normal_initializer(stddev=0.02),activation=tf.nn.relu):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope_name):
        w = tf.get_variable("w", [shape[1], output_dim], dtype=tf.float32,
                            initializer=initializer)
        b = tf.get_variable("b", [output_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        out = tf.nn.xw_plus_b(x, w, b)
        if activation is not None:
            out = activation(out)

        return w, b, out


def huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), delta * tf.abs(x) - 0.5* delta)


class DQN(BaseModel):

    def __init__(self, n_actions, config):
        super(DQN, self).__init__(config, "dqn")
        self.n_actions = n_actions
        self.history_len = config.history_len
        self.cnn_format = config.cnn_format
        self.all_tf = not True


    def train_on_batch_target(self, state, action, reward, state_, terminal, steps):
        state_ = state_ / 255.0
        state = state / 255.0
        target_val = self.q_target_out.eval({self.state_target: state_}, session=self.sess)
        max_target = np.max(target_val, axis=1)
        target = (1. - terminal) * self.gamma * max_target + reward
        _, q, train_loss, q_summary, image_summary = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.avg_q_summary, self.merged_image_sum],
            feed_dict={
                self.state: state,
                self.action: action,
                self.target_val: target,
                self.lr: self.learning_rate
            }
        )
        if self.train_steps % 1000 == 0:
            self.file_writer.add_summary(q_summary, self.train_steps)
            self.file_writer.add_summary(image_summary, self.train_steps)
        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean(), train_loss

    def train_on_batch_all_tf(self, state, action, reward, state_, terminal, steps):
        state = state/255.0
        state_= state_/255.0
        _, q, train_loss, q_summary, image_summary = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.avg_q_summary, self.merged_image_sum], feed_dict={
                self.state: state,
                self.action: action,
                self.state_target:state_,
                self.reward: reward,
                self.terminal: terminal,
                self.lr: self.learning_rate,
                self.dropout: self.keep_prob
            }
        )
        if self.train_steps % 1000 == 0:
            self.file_writer.add_summary(q_summary, self.train_steps)
            self.file_writer.add_summary(image_summary, self.train_steps)
        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean(), train_loss

    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.history_len, self.screen_height, self.screen_width],
                                    name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")

        self.state_target = tf.placeholder(tf.float32,
                                           shape=[None, self.history_len, self.screen_height, self.screen_width],
                                           name="input_target")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")

        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")
        self.target_val_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])

        self.learning_rate_step = tf.placeholder("int64", None, name="learning_rate_step")

    def add_logits_op_train(self):
        self.image_summary = []
        if self.cnn_format == "NHWC":
            x = tf.transpose(self.state, [0, 2, 3, 1])
        else:
            x = self.state
        w, b, out, summary = conv2d_layer(x, 32, [8, 8], [4, 4], scope_name="conv1_train", summary_tag="conv1_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc1"] = w
        self.w["bc1"] = b
        self.image_summary.append(summary)

        w, b, out, summary = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_train", summary_tag="conv2_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc2"] = w
        self.w["bc2"] = b
        self.image_summary.append(summary)

        w, b, out, summary = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_train", summary_tag="conv3_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc3"] = w
        self.w["bc3"] = b
        self.image_summary.append(summary)

        shape = out.get_shape().as_list()
        out_flat = tf.reshape(out, [-1, reduce(lambda x, y: x * y, shape[1:])])

        w, b, out = fully_connected_layer(out_flat, 512, scope_name="fully1_train")

        self.w["wf1"] = w
        self.w["bf1"] = b

        w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_train", activation=None)

        self.w["wout"] = w
        self.w["bout"] = b

        self.q_out = out
        self.q_action = tf.argmax(self.q_out, axis=1)

    def add_logits_op_target(self):
        if self.cnn_format == "NHWC":
            x = tf.transpose(self.state_target, [0, 2, 3, 1])
        else:
            x = self.state_target
        w, b, out, _ = conv2d_layer(x, 32, [8, 8], [4, 4], scope_name="conv1_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc1"] = w
        self.w_target["bc1"] = b

        w, b, out, _ = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc2"] = w
        self.w_target["bc2"] = b

        w, b, out, _ = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc3"] = w
        self.w_target["bc3"] = b

        shape = out.get_shape().as_list()
        out_flat = tf.reshape(out, [-1, reduce(lambda x, y: x * y, shape[1:])])

        w, b, out = fully_connected_layer(out_flat, 512, scope_name="fully1_target")

        self.w_target["wf1"] = w
        self.w_target["bf1"] = b

        w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_target", activation=None)

        self.w_target["wout"] = w
        self.w_target["bout"] = b

        self.q_target_out = out
        self.q_target_action = tf.argmax(self.q_target_out, axis=1)

    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])

    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='q_acted')
        self.delta = train - self.target_val
        self.loss = tf.reduce_mean(huber_loss(self.delta))

        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.n_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.merged_image_sum = tf.summary.merge(self.image_summary, "images")
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def add_loss_op_target_tf(self):
        self.reward = tf.cast(self.reward, dtype=tf.float32)
        target_best = tf.reduce_max(self.q_target_out, 1)
        masked = (1.0 - self.terminal) * target_best
        target = self.reward + self.gamma * masked

        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1)
        delta = target - train
        self.loss = tf.reduce_mean(huber_loss(delta))
        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.n_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.merged_image_sum = tf.summary.merge(self.image_summary, "images")

    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        if self.all_tf:
            self.add_loss_op_target_tf()
        else:
            self.add_loss_op_target()
        self.add_train_op(self.lr_method, self.lr, self.loss, clip=10)
        self.initialize_session()
        self.init_update()

    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)