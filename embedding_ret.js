var tf = require('@tensorflow/tfjs-node');

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
    EmbeddingRet: EmbeddingRet,
}