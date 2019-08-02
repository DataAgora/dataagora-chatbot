var tf = require('@tensorflow/tfjs-node');
var biasAdd = require('./utils').biasAdd;

class EmbeddingSim extends tf.layers.Layer {

    constructor(useBias=false, initializer=tf.initializers.zeros, regularizer=null, constraint=null, stopGradient=false, ...args) {
        super(...args);
        this.supportsMasking = true;
        this.useBias = useBias;
        this.initializer = initializer;
        this.regularizer = regularizer;
        this.constraint = constraint;
        this.stopGradient = stopGradient
        this.bias = null;
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

    dot(x, y) {
        var newArr = [];
        x = x.arraySync();
        y = y.arraySync();
        for (var i = 0; i < x.length; i++) {
            // console.log("HEY", i);
            newArr.push(tf.dot(x[i], y));
        }
        var newTensor = tf.tensor(newArr)
        //console.log("FINISHED");
        return newTensor;
    }

    call(inputs, mask=null, ...args) {
        var embeddings = inputs[1];
        inputs = inputs[0];
        // console.log(embeddings)
        // console.log(inputs)
        // console.log(1)
        var outputs = this.dot(inputs, tf.transpose(embeddings));
        // console.log(2);
        if (this.useBias) {
            outputs = biasAdd(outputs, this.bias);
        }
        // console.log("MADE IT HERE");
        return tf.layers.softmax(outputs)
    }
}

module.exports = {
    EmbeddingSim: EmbeddingSim
}