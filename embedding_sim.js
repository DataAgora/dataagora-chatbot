var tf = require('@tensorflow/tfjs-node');

class EmbeddingSim extends tf.layers.Layer {

    constructor(use_bias=false, initializer=tf.initializers.zeros, regularizer=null, constraint=null, stop_gradient=false, ...args) {
        super(...args);
        this.supportsMasking = true;
        this.use_bias = use_bias;
        this.initializer = initializer;
        this.regularizer = regularizer;
        this.constraint = constraint;
        this.stop_gradient = stop_gradient
    }

    getConfig() {
        var base_config = super.getConfig();
        base_config["use_bias"] = this.use_bias;
        base_config["initializer"] = this.initializer;
        base_config["regularizer"] = this.regularizer;
        base_config["constraint"] = this.constraint;
        base_config["stop_gradient"] = this.stop_gradient;

        return base_config;
    }

    build(input_shape) {
        if (this.use_bias) {
            var embed_shape = input_shape[1];
            var token_num = parseInt(embed_shape[0])
            this.bias = this.addWeight(
                "bias",
                (token_num),
                undefined,
                this.initializer,
                this.regularizer,
                this.constraint
            )
        }
        super.build(input_shape);
    }

    computeOutputShape(input_shape) {
        var feature_shape = input_shape[0];
        var embed_shape = input_shape[1];
        var token_num = embed_shape[0];
        return feature_shape.slice(0, feature_shape.length - 1).concat([token_num])
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

class EmbeddingRet extends tf.layers.Layer {
    constructor(...args) {
        super(...args);
        this.embedding_object = tf.layers.embedding(...args);
    }

    computeOutputShape(input_shape) {
        return [
            this.embedding_object.computeOutputShape(input_shape),
            (this.inputDim, this.outputDim)
        ]
    }

    computeMask(inputs, mask=null) {
        return [
            this.embedding_object.computeMask(inputs, mask),
            null
        ]
    }

    call(inputs) {
        return [
            this.embedding_object.call(inputs),
            tf.tensor(this.embedding_object.embeddings.arraySync())
        ]
    }

    apply(...args) {
        return this.embedding_object.apply(...args);
    }

    countParams(...args) {
        return this.embedding_object.countParams(...args);
    }

    build(...args) {
        return this.embedding_object.build(...args);
    }

    getWeights(...args) {
        return this.embedding_object.getWeights(...args);
    }

    setWeights(...args) {
        return this.embedding_object.setWeights(...args);
    }

    addWeight(...args) {
        return this.embedding_object.addWeight(...args);
    }

    addLoss(...args) {
        return this.embedding_object.addLoss(...args);
    }

    getConfig(...args) {
        return this.embedding_object.getConfig(...args);
    }

    dispose(...args) {
        return this.embedding_object.dispose(...args);
    }
}

module.exports = {
    EmbeddingSim: EmbeddingSim
}