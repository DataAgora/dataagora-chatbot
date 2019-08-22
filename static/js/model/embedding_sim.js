// var tf = require('@tensorflow/tfjs-node');
// var biasAdd = require('./utils').biasAdd;

import {biasAdd, dot} from './utils.js'

export class EmbeddingSim extends tf.layers.Layer {

    constructor(useBias=false, initializer=tf.initializers.zeros, regularizer=null, constraint=null, stopGradient=false, ...args) {
        if (typeof useBias == 'object') {
            var config = useBias;
            useBias = false;
            super({trainable:config.trainable, name:config.name});
        } else {
            super(...args);
        }

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

    call(inputs, mask=null, ...args) {
        var embeddings = inputs[1];
        inputs = inputs[0];
        // console.log("INPUTS", inputs.arraySync());
        // console.log("EMBEDDINGS", embeddings.arraySync())

        // console.log(embeddings)
        // console.log(inputs)
        var outputs = dot(inputs, tf.transpose(embeddings));
        if (this.useBias) {
            outputs = biasAdd(outputs, this.bias);
        }
        return tf.softmax(outputs);
        //console.log("AYEE")
        // var output_arr = tf.split(outputs, 2);
        // console.log("3");
        // var new_output_arr = [];
        // output_arr.forEach(element => {
        //     new_output_arr.push(tf.softmax(element));
        // });
        // console.log("4");
        // return tf.concat(new_output_arr)
    }

    static get className() {
        return 'EmbeddingSim';
    }
}
tf.serialization.registerClass(EmbeddingSim)
// module.exports = {
//     EmbeddingSim: EmbeddingSim
// }