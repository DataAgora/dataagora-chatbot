var tf = require('@tensorflow/tfjs-node');
var assert = require('assert');

function biasAdd(x, bias) {
    var biasShape = bias.shape;
    assert(biasShape.length == 1 || biasShape.length == x.shape.length - 1);

    assert(x.shape.length == 3);
    // switch (x.shape.length) {
    //     case 5:
    //         if (biasShape.length == 1) {
    //             x = tf.add(x, tf.reshape(bias, [1, 1, 1, biasShape[0]]));
    //         } else {
    //             x = tf.add(x, tf.reshape(bias, [1].concat(biasShape)));
    //         }
    //         break;

    //     case 4:

    // }
    x = tf.add(x, tf.reshape(bias, [1, 1, biasShape[0]]));
    return x;
}

function batchDot(x, y, axis) {
    var axes = [axis, axis];

    var x_dim = x.shape.length;
    var y_dim = y.shape.length;

    assert (x_dim == y_dim);

    var diff = 0;


    if (x_dim == 2 && y_dim == 2) {
        if (axes[0] == axes[1]) {
            out = tf.mul(x, y).sum()
        } else {
            assert(false);
        }
    } else {
        var out;
        var adj_x, adj_y;
        if (axes[0] == x_dim - 1) {
            adj_x = false;
        } else {
            adj_x = true;
        }
        if (axes[1] == y_dim - 1) {
            adj_y = true;
        } else {
            adj_x = false;
        }
        out = tf.matMul(x, y, adj_x, adj_y);
    }

    

    if (out.shape.length == 1) {
        out = tf.expandDims(out, 1);
    }

    return out;
}

// var x = tf.initializers.ones().apply([24, 1024, 64])
// var y = tf.initializers.ones().apply([24, 1024, 64])
// console.log(batchDot(x, y, 2));

module.exports = {
    batchDot:batchDot,
    biasAdd:biasAdd
}