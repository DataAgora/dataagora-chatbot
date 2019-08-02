var tf = require('@tensorflow/tfjs-node');

class LayerNormalization extends tf.layers.Layer {
    constructor (center=true, scale=true, gammaInitializer=tf.initializers.ones(), epsilon=null,
        betaInitializer=tf.initializers.zeros(), gammaRegularizer=null, betaRegularizer=null,
        gammaConstraint=null, betaConstraint=null, ...args) {

            super(...args);

            this.supports_masking = true;
            this.center = center;
            this.scale = scale;
            if (epsilon == null)
                epsilon = 1e-7 * 1e-7;
            this.epsilon = epsilon;
            this.gammaInitializer = gammaInitializer;
            this.betaInitializer = betaInitializer;
            this.gammaRegularizer = gammaRegularizer;
            this.betaRegularizer = betaRegularizer;
            this.gammaConstraint = gammaConstraint;
            this.betaConstraint = betaConstraint;
            
            this.gamma = null
            this.beta = null
    }

    computeOutputShape(inputShape) {
        //console.log(inputShape, "HULLO")
        return inputShape;
    }

    computeMask(inputs, inputMask) {
        return inputMask;
    }

    build(inputShape) {
        var shape = inputShape.slice(inputShape.length - 1);
        if (this.scale) {
            this.gamma = this.addWeight(
                `gamma`,
                shape,
                undefined,
                this.gammaInitializer
            );
        }
        if (this.center) {
            this.beta = this.addWeight(
                `beta`,
                shape,
                undefined,
                this.betaInitializer
            );
        }
        super.build(inputShape);
    }

    call(inputs, training=null) {
        console.log("inputs", inputs);
        //console.log(inputs, "yeet")
        inputs = inputs[0];
        var mean = tf.mean(inputs, inputs.shape.length-1, true);
        var variance = tf.mean(tf.square(tf.sub(inputs, mean)), inputs.shape.length - 1, true);
        var std = tf.sqrt(tf.add(variance, this.epsilon));
        var outputs = tf.div(tf.sub(inputs, mean), std);
        if (this.scale) {
            outputs = tf.mul(outputs, this.gamma.val);
        }
        if (this.center) {
            outputs = tf.add(outputs, this.beta.val)
        }

        console.log("outputs", outputs)
        return outputs;
    }
}

module.exports = {
    LayerNormalization: LayerNormalization
}