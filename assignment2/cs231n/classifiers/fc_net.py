from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params["W1"] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params["b2"] = np.zeros(num_classes)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]
        A1, cache1 = affine_relu_forward(X, W1, b1)
        Z2, cache2 = affine_forward(A1, W2, b2)
        scores = Z2

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N = X.shape[0]
        X_reshape = X.reshape(N,-1)
        exp_scores = np.exp(Z2)
        sum_row = np.sum(exp_scores, axis=1)
        sum_row = sum_row.reshape((sum_row.shape[0], 1))
        scores = exp_scores / sum_row  # (N,C)
        loss = np.sum(-np.log(scores[[range(N)], y]))
        loss /= N
        reg = self.reg
        loss += 0.5 * reg * np.sum(np.square(W1))
        loss += 0.5 * reg * np.sum(np.square(W2))
        scores[[range(N)], y] -= 1  # (N,C)  X(N,D)
        dZ2 = scores
        dW2 = A1.T.dot(dZ2)
        dW2 /= N
        dW2 += 2 * W2 * reg * 0.5
        db2 = 1 / N * np.sum(dZ2, axis=0)
        dA1 = dZ2.dot(W2.T)
        fc_cache, relu_cache = cache1
        Z1 = relu_cache
        dA1_der_Z1 = np.where(Z1 > 0, 1, 0)
        dZ1 = dA1 * dA1_der_Z1
        dW1 =  X_reshape.T.dot(dZ1)
        dW1 /= N
        # print(dW1.shape)
        # print(W1.shape)
        # print(reg)
        dW1 += 2 * W1 * reg * 0.5
        db1 = 1 / N * np.sum(dZ1, axis=0)
        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for i in range(self.num_layers):
            if i ==0:
               self.params["W"+str(i+1)] = weight_scale * np.random.randn(input_dim,hidden_dims[i])
               self.params["b" + str(i+1)] = np.zeros(hidden_dims[i])
               if self.normalization is not None:
                   self.params["gamma" + str(i + 1)] =np.ones(hidden_dims[i])
                   self.params["beta" + str(i + 1)] = np.zeros(hidden_dims[i])
            elif i == self.num_layers-1:
                self.params["W" + str(i + 1)] = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
                self.params["b" + str(i + 1)] = np.zeros(num_classes)
            else:
                self.params["W" + str(i + 1)] = weight_scale * np.random.randn(hidden_dims[i - 1],hidden_dims[i] )
                self.params["b" + str(i + 1)] = np.zeros(hidden_dims[i])
                if self.normalization is not None:
                    self.params["gamma" + str(i + 1)] = np.ones(hidden_dims[i])
                    self.params["beta" + str(i + 1)] = np.zeros(hidden_dims[i])

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
         Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X_R = X.reshape(X.shape[0],-1) #(N,D)
        #N,D = X_R.shape
        caches ={}
        outs = {}
        A_pre = X_R
        # for 循环完成隐藏层的前向传播
        for i in range(1,self.num_layers):
            if self.normalization is not None:
                # 第一步归一化
                x = None
                N, D =  A_pre.shape
                if self.normalization == "batchnorm":
                    relu_out, cache = affine_bn_relu_forward(A_pre, self.params["W" + str(i)], self.params["b" + str(i)], self.params["gamma" + str(i)], self.params["beta" + str(i)], self.bn_params[i-1])
                    caches["abrcache" + str(i)] = cache
                    outs["abrout" + str(i)] = relu_out
                    x = relu_out
                if self.normalization == "layernorm":
                    relu_out, cache= affine_ln_relu_forward(A_pre, self.params["W" + str(i)], self.params["b" + str(i)], self.params["gamma" + str(i)], self.params["beta" + str(i)], self.bn_params[i-1])
                    caches["alrcache" + str(i)] = cache
                    outs["alrout" + str(i)] =  relu_out
                    x =  relu_out
                A_pre = x
            else:
               # 第二步 前向传播
               A_now, cache = affine_relu_forward(A_pre, self.params["W" + str(i)], self.params["b" + str(i)])
               outs["A" + str(i)] = A_now
               caches["arfcache" + str(i)] = cache
               A_pre = A_now
            # 第三步 dropout
            if self.use_dropout:
                out, cache = dropout_forward(A_pre, self.dropout_param)
                outs["drop_out" + str(i)] = out
                caches["drop_cache" + str(i)] = cache
                A_pre = out
        Z, cache = affine_forward(A_pre,self.params["W" + str(self.num_layers)] , self.params["b" + str(self.num_layers)])
        scores = Z
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N, D = X_R.shape
        loss,dZ = softmax_loss(scores, y)
        reg = self.reg
        for i in range(self.num_layers):
            loss += 0.5 * reg * np.sum(np.square(self.params["W" + str(i+1)]))
        dx, dw, db = affine_backward(dZ, cache)
       # dw /= N
        dw += 2 * self.params["W"+str(self.num_layers)] * reg * 0.5
       # db = 1 / N * db
        grads["W"+str(self.num_layers)] = dw
        grads["b" + str(self.num_layers)] = db
        dout =dx
        for i in range(self.num_layers-1,0,-1):
            if self.use_dropout:
                dx =dropout_backward(dout, caches["drop_cache" + str(i)])
                dout = dx

            if self.normalization is not None:
                if self.normalization == "batchnorm":
                    dx, dw, db, dgamma, dbeta  = affine_bn_relu_backward(dout, caches["abrcache" + str(i)])

                if self.normalization == "layernorm":
                    dx, dw, db, dgamma, dbeta = affine_ln_relu_backward(dout, caches["alrcache" + str(i)])

                grads["gamma" + str(i)] = dgamma
                grads["beta" + str(i)] = dbeta
                dw += 2 * self.params["W" + str(i)] * reg * 0.5
                # db = 1 / N * db
                grads["W" + str(i)] = dw
                grads["b" + str(i)] = db
                dout = dx
            else:
                dx, dw, db = affine_relu_backward(dout, caches["arfcache" + str(i)])
                # dw /= N
                dw += 2 * self.params["W" + str(i)] * reg * 0.5
                # db = 1 / N * db
                grads["W" + str(i)] = dw
                grads["b" + str(i)] = db
                dout = dx


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    # def loss(self, X, y=None):
    #     """
    #     Compute loss and gradient for the fully-connected net.
    #     Input / output: Same as TwoLayerNet above.
    #     """
    #     X = X.astype(self.dtype)
    #     mode = "test" if y is None else "train"
    #
    #     # Set train/test mode for batchnorm params and dropout param since they
    #     # behave differently during training and testing.
    #     if self.use_dropout:
    #         self.dropout_param["mode"] = mode
    #     if self.normalization == "batchnorm":
    #         for bn_param in self.bn_params:
    #             bn_param["mode"] = mode
    #     scores = None
    #     ############################################################################
    #     # TODO: Implement the forward pass for the fully-connected net, computing  #
    #     # the class scores for X and storing them in the scores variable.          #
    #     #                                                                          #
    #     # When using dropout, you'll need to pass self.dropout_param to each       #
    #     # dropout forward pass.                                                    #
    #     #                                                                          #
    #     # When using batch normalization, you'll need to pass self.bn_params[0] to #
    #     # the forward pass for the first batch normalization layer, pass           #
    #     # self.bn_params[1] to the forward pass for the second batch normalization #
    #     # layer, etc.                                                              #
    #     ############################################################################
    #     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #
    #     layer_ret = []
    #
    #     # Forward for: {affine - relu - [dropout]} x (self.num_layers-1)
    #     if self.normalization is None:
    #         layer_ret.append((affine_relu_forward(X, self.params['W1'], self.params['b1'])))
    #         if self.use_dropout:
    #             layer_ret.append((dropout_forward(layer_ret[-1][0], self.dropout_param)))
    #
    #         for i in range(2, self.num_layers):
    #             layer_ret.append(
    #                 (affine_relu_forward(layer_ret[-1][0], self.params['W' + str(i)], self.params['b' + str(i)])))
    #             if self.use_dropout:
    #                 layer_ret.append((dropout_forward(layer_ret[-1][0], self.dropout_param)))
    #
    #     # Forward for: {affine - batch/layer norm - relu - [dropout]} x (self.num_layers-1)
    #     elif self.normalization:
    #         # step 1. select forward function
    #         forward_func = None
    #         if self.normalization == "batchnorm":
    #             forward_func = affine_bn_relu_forward
    #         elif self.normalization == "layernorm":
    #             forward_func = affine_ln_relu_forward
    #
    #         # step 2. perform forward
    #         layer_ret.append((forward_func(X, self.params['W1'], self.params['b1'],
    #                                        self.params['gamma1'], self.params['beta1'], self.bn_params[0])))
    #         if self.use_dropout:
    #             layer_ret.append((dropout_forward(layer_ret[-1][0], self.dropout_param)))
    #
    #         for i in range(2, self.num_layers):
    #             layer_ret.append((forward_func(layer_ret[-1][0], self.params['W' + str(i)], self.params['b' + str(i)],
    #                                            self.params['gamma' + str(i)], self.params['beta' + str(i)],
    #                                            self.bn_params[i - 1])))
    #             if self.use_dropout:
    #                 layer_ret.append((dropout_forward(layer_ret[-1][0], self.dropout_param)))
    #
    #     # Forward for: - (last) affine - softmax
    #     layer_ret.append((affine_forward(layer_ret[-1][0],
    #                                      self.params['W' + str(self.num_layers)],
    #                                      self.params['b' + str(self.num_layers)])))
    #
    #     scores = layer_ret[-1][0]
    #
    #     pass
    #
    #     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #     ############################################################################
    #     #                             END OF YOUR CODE                             #
    #     ############################################################################
    #
    #     # If test mode return early
    #     if mode == "test":
    #         return scores
    #
    #     loss, grads = 0.0, {}
    #     ############################################################################
    #     # TODO: Implement the backward pass for the fully-connected net. Store the #
    #     # loss in the loss variable and gradients in the grads dictionary. Compute #
    #     # data loss using softmax, and make sure that grads[k] holds the gradients #
    #     # for self.params[k]. Don't forget to add L2 regularization!               #
    #     #                                                                          #
    #     # When using batch/layer normalization, you don't need to regularize the scale   #
    #     # and shift parameters.                                                    #
    #     #                                                                          #
    #     # NOTE: To ensure that your implementation matches ours and you pass the   #
    #     # automated tests, make sure that your L2 regularization includes a factor #
    #     # of 0.5 to simplify the expression for the gradient.                      #
    #     ############################################################################
    #     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #
    #     # Compute loss for all kinds of hidden layer patterns
    #     loss, dout = softmax_loss(scores, y)
    #     reg = self.reg
    #     for i in range(self.num_layers):
    #         w = self.params['W' + str(i + 1)]
    #         loss += 0.5 * reg * np.sum(w * w)
    #
    #     # Compute grads
    #     # Backward for: - (last) affine - softmax
    #     idx = self.num_layers
    #
    #     _, cache = layer_ret[-1]
    #     dout, dw, db = affine_backward(dout, cache)
    #     grads['W' + str(idx)] = dw + reg * self.params['W' + str(idx)]
    #     grads['b' + str(idx)] = db
    #     idx -= 1
    #     del layer_ret[-1]  # for memory leak consideration
    #
    #     # Backward for: {affine - relu - [dropout]} x (self.num_layers-1)
    #     if self.normalization is None:
    #         while len(layer_ret) > 0:
    #             if self.use_dropout:
    #                 _, cache = layer_ret[-1]
    #                 dout = dropout_backward(dout, cache)
    #                 del layer_ret[-1]
    #             _, cache = layer_ret[-1]
    #             dout, dw, db = affine_relu_backward(dout, cache)
    #             grads['W' + str(idx)] = dw + reg * self.params['W' + str(idx)]
    #             grads['b' + str(idx)] = db
    #             idx -= 1
    #             del layer_ret[-1]
    #
    #     # Backward for: {affine - batch/layer norm - relu} x (self.num_layers-1)
    #     elif self.normalization:
    #         # step 1. select backward function
    #         if self.normalization == "batchnorm":
    #             backward_func = affine_bn_relu_backward
    #         elif self.normalization == "layernorm":
    #             backward_func = affine_ln_relu_backward
    #
    #         # step 2. perform backward
    #         while len(layer_ret) > 0:
    #             if self.use_dropout:
    #                 _, cache = layer_ret[-1]
    #                 dout = dropout_backward(dout, cache)
    #                 del layer_ret[-1]
    #             _, cache = layer_ret[-1]
    #             dout, dw, db, dgamma, dbeta = backward_func(dout, cache)
    #             grads['W' + str(idx)] = dw + reg * self.params['W' + str(idx)]
    #             grads['b' + str(idx)] = db
    #             grads['gamma' + str(idx)] = dgamma
    #             grads['beta' + str(idx)] = dbeta
    #             idx -= 1
    #             del layer_ret[-1]
    #
    #     pass
    #
    #     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #     ############################################################################
    #     #                             END OF YOUR CODE                             #
    #     ############################################################################
    #
    #     return loss, grads