var tf = require('@tensorflow/tfjs-node');
var batchDot = require('./utils').batchDot;

class ScaledDotProductAttention extends tf.layers.Layer {
    constructor (returnAttention=false, historyOnly=false, ...args) {
        super(...args);
        this.supportsMasking = true;
        this.returnAttention = returnAttention;
        this.historyOnly = historyOnly;
    }

    getConfig() {
        var baseConfig = super.getConfig();
        baseConfig['returnAttention'] = this.returnAttention;
        baseConfig['historyOnly'] = this.historyOnly;

        return baseConfig;
    }

    computeOutputShape(inputShape) {
        var queryShape, keyShape, valueShape;
        if (Array.isArray(inputShape)) {
            queryShape = inputShape[0];
            keyShape = inputShape[1];
            valueShape = inputShape[2];
        } else {
            queryShape = keyShape = valueShape = inputShape;
        }

        var outputShape = queryShape.slice(0, queryShape.length - 1).concat([keyShape[1]]);

        if (this.returnAttention) {
            var attentionShape = queryShape.slice(0, 2).concat([keyShape[1]]);
            return [outputShape, attentionShape]
        } else {
            return outputShape
        }

    }

    computeMask(self, inputs, mask=null) {
        if (Array.isArray(mask)) {
            mask = mask[0];
        }
        if (this.returnAttention) {
            return [mask, null];
        } else {
            return mask;
        }
    }

    call(inputs, mask=null, ...args) {
        //console.log("YAY");
        var query, key, value;
        if (Array.isArray(inputs)) {
            query = inputs[0];
            key = inputs[1];
            value = inputs[2];
        } else {
            query = key = value = inputs;
        }

        mask = null;

        // console.log("query", query)
        // console.log("key", key)
        var tempQuery = query.arraySync();
        var tempKey = key.arraySync();
        var featureDim = query.shape[query.shape.length - 1];
        // var batch_dot_result = [];
        // for (var i = 0; i < tempQuery.length; i++) {
        //     for (var j = 0; j < tempKey.length; j++) {
        //         var new_arr = [];
        //         for (var k = 0; k < tempQuery[0].length; k++) {
        //             for (var m = 0; m < tempKey[0].length; m++) {
        //                 new_arr.push(tf.dot(tempQuery[i][k], tempKey[j][m]).arraySync());
        //             }
        //         }
        //         batch_dot_result.push(new_arr);
        //     }
        // }
        var batch_dot_result = batchDot(query, key, 2);
        //console.log(batch_dot_result.shape)
        var e = tf.div(batch_dot_result, tf.sqrt(tf.cast(featureDim, "float32")));
        //console.log(e.shape)
        e = tf.exp(tf.sub(e, tf.max(e, e.shape.length-1, true)));

        if (this.historyOnly) {
            var queryLen = query.shape[1];
            var keyLen = key.shape[1];
            var indices = tf.expandDims(tf.range(0, keyLen), 0);
            var upper = tf.expandDims(tf.range(0, queryLen), tf.range(0, queryLen).shape.length - 1);
            var result = tf.expandDims(tf.cast(tf.lessEqual(indices, upper), "float32"), 0);
            //console.log(result.shape, e.shape)
            e = tf.mul(e, result);
        }

        if (mask != null) {
            e = tf.mul(e, tf.cast(tf.expandDims(mask, mask.shape.length - 2), "float32"));
        }

        var a = tf.div(e, tf.add(tf.sum(e, e.shape.length - 1, true), 1e-7));
        var v = [];
        //console.log(a, "AAA")
        // var a_arr = a.arraySync();
        // var tempValue = value.arraySync();
        // for (var i = 0; i < a_arr.length; i++) {
        //     v.push(tf.dot(a_arr[i], tempValue[i]).arraySync());
        // }
        // v = tf.tensor(v);
        v = batchDot(a, value, 1);
        //console.log(v)
        if (this.returnAttention) {
            return [v, a];
        } else {
            return v;
        }
    }
}

module.exports = {
    ScaledDotProductAttention:ScaledDotProductAttention
}