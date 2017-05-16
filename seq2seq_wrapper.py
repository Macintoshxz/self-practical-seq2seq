import tensorflow as tf
import numpy as np
import sys


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, 
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001, 
            epochs=100000, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name


        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():

            # placeholders, create a bunch of placeholders for feeding input sequences, labels and decoder inputs.The labels correspond to the real output sequence. 
            tf.reset_default_graph()
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [ tf.placeholder(shape=[None,],                       #  Inserts a placeholder for a tensor that will be always fed. fed using sess.run()
                            dtype=tf.int64,                 
                            name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
            self.labels = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int64, 
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2,  ... y_t-1 ]
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1] # Decoder


            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32) # define a placeholder, for control the DROPOUT to avoid overfitting
                                                        # It essentially just drops some of the unit activations in a layer, by making them zero.
            # define the basic cell
            basic_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                    tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),   # Basic cells
                    output_keep_prob=self.keep_prob)

                                                                        # state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state,
                                                                        # which is cell state and memory state, 相当于c和h。 these tensor代表了combined internal state of cell
                                                                        # and should be passed together. 这种是新的tf的工作方式，用tuples， 比较快。
                                                                        # 
                                                                        # 老的方式 是If False, they are concatenated along the column axis. The latter behavior will 
                                                                        # soon be deprecated.                                                                        
                                                                        # 在Dataanalysis的ipnb里，留了一个简单的例子来解释 state_is_tuple为True和False的区别，总结来说是：

                                                                        # state = tf.zeros([1,LSTM_CELL_SIZE*2])
                                                                        # 新的方式是用两个长度为2的tuple/tensors来表示state
                                                                        # 旧的方式是用一个长度为4的tensor来表示 [0,0,0,0] becomes ([0,0],[0,0])

            # stack LSTM cells together : n layered model
            stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True) #The LSTM cellsare stacked together to form a stacked LSTM,
                                                                                                                   # using MultiRNNCell


            # for parameter sharing between training model 
            #  and testing model, does word embedding internally
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model 
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input 
                #  to the next timestep
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=True)      # feed_previous  causes the decoder of the model to use the output of previous timestep as input to the current timestep, 
                                             # while during training, the input to a timestep (in the decoder) is taken from the labels sequence (the real output sequence).


            # now, for training,
            #  build loss function ( high level function sequence_loss) to get the expression for loss. Then, we build a train operation that minimizes the loss.

            # weighted loss
            #  TODO : add parameter hint
            loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('<log> Building Graph ')
        # build comput graph
        __graph__()
        sys.stdout.write('</log>')



    '''
        Training and Evaluation

    '''

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5) # A dropout of 0.5 is used during training
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.) # NO Dropout
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, sess=None ):
        
        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())            #这一步是必须的， 初始化

        sys.stdout.write('\n<log> Training started </log>\n')

        # run M epochs
        # one epoch = one forward pass and one backward pass of all the training examples
        # number of iterations = number of passes, each pass using [batch size] number of examples.
        # Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

        for i in range(self.epochs):
            try:
                self.train_batch(sess, train_set)
                if i and i% (self.epochs//100) == 0: # TODO : make this tunable by the user
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 16) # TODO : and this
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess

    def restore_last_session(self): 
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction                # does a forward step and returns the indices of most probable words emitted by the model
    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


