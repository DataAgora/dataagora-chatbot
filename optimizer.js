var tf = require('@tensorflow/tfjs-node')

class AdamWrapper extends tf.train.adam {
    constructor (learningRate, beta1, beta2, train_vars, loss_function) {
        super(learningRate, beta1, beta2);
        this.train_vars = train_vars; 
        this.loss_function = loss_function;
    }

    

    
}