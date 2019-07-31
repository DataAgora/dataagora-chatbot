var tf = require('@tensorflow/tfjs-node');

class EmbeddingSim extends tf.layers.Layer {

    constructor(useBias=false, initializer=tf.initializers.zeros, regularizer=null, constraint=null, stopGradient=false, ...args) {
        super(...args);
        this.supportsMasking = true;
        this.useBias = useBias;
        this.initializer = initializer;
        this.regularizer = regularizer;
        this.constraint = constraint;
        this.stopGradient = stopGradient
    }

    getConfig() {
        var baseConfig = super.getConfig();
        baseConfig["useBias"] = this.useBias;
        baseConfig["initializer"] = this.initializer;
        baseConfig["regularizer"] = this.regularizer;
        baseConfig["constraint"] = this.constraint;
        baseConfig["stopGradient"] = this.stopGradient;

        return baseConfig;
    }

    build(inputShape) {
        if (this.useBias) {
            var embedShape = inputShape[1];
            var tokenNum = parseInt(embedShape[0])
            this.bias = this.addWeight(
                "bias",
                (tokenNum),
                undefined,
                this.initializer,
                this.regularizer,
                this.constraint
            )
        }
        super.build(inputShape);
    }

    computeOutputShape(inputShape) {
        var featureShape = inputShape[0];
        var embedShape = inputShape[1];
        var tokenNum = embedShape[0];
        return featureShape.slice(0, featureShape.length - 1).concat([tokenNum])
    }

    computeMask(inputs, mask=null) {
        if (mask == null) {
            return null;
        }
        return mask[0];
    }

    call(inputs, mask=null, ...args) {
        var embeddings = inputs[1];
        inputs = inputs[0];
        var outputs = tf.dot(inputs, tf.transpose(embeddings));
        return tf.layers.softmax(outputs)
    }
}

module.exports = {
    EmbeddingSim: EmbeddingSim
}