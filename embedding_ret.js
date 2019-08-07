// var tf = require('@tensorflow/tfjs-node');

export class EmbeddingRet extends tf.layers.Layer {
    constructor(...args) {
        super(...args);
        this.embeddingObject = tf.layers.embedding(...args);
    }

    computeOutputShape(inputShape) {
        return [
            this.embeddingObject.computeOutputShape(inputShape),
            [this.embeddingObject.inputDim, this.embeddingObject.outputDim]
        ]

        
    }

    computeMask(inputs, mask=null) {
        return [
            this.embeddingObject.computeMask(inputs, mask),
            null
        ]
    }

    call(inputs) {
        return [
            this.embeddingObject.call(inputs),
            this.embeddingObject.embeddings.val.clone()
        ]
    }

    // apply(...args) {
    //     return this.embeddingObject.apply(...args);
    // }

    countParams(...args) {
        return this.embeddingObject.countParams(...args);
    }

    build(...args) {
        return this.embeddingObject.build(...args);
    }

    getWeights(...args) {
        return this.embeddingObject.getWeights(...args);
    }

    setWeights(...args) {
        return this.embeddingObject.setWeights(...args);
    }

    addWeight(...args) {
        return this.embeddingObject.addWeight(...args);
    }

    addLoss(...args) {
        return this.embeddingObject.addLoss(...args);
    }

    getConfig(...args) {
        return this.embeddingObject.getConfig(...args);
    }

    dispose(...args) {
        return this.embeddingObject.dispose(...args);
    }
}

// module.exports = {
//     EmbeddingRet: EmbeddingRet,
// }