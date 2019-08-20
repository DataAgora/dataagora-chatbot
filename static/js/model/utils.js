

export function biasAdd(x, bias) {
    var biasShape = bias.shape;
    console.assert(biasShape.length == 1 || biasShape.length == x.shape.length - 1);

    console.assert(x.shape.length == 3);
    x = tf.add(x, tf.reshape(bias, [1, 1, biasShape[0]]));
    return x;
}

export function batchDot(x, y, axis) {
    var axes = [axis, axis];

    var x_dim = x.shape.length;
    var y_dim = y.shape.length;

    console.assert (x_dim == y_dim);

    var diff = 0;


    if (x_dim == 2 && y_dim == 2) {
        if (axes[0] == axes[1]) {
            out = tf.mul(x, y).sum()
        } else {
            console.assert(false);
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

export function dot(x, y) {
    var x_dim = x.shape.length;
    var y_dim = y.shape.length;

    if (x_dim <= 2 && y_dim <= 2) {
        return tf.dot(x_dim, y_dim);
    }

    var y_permute_dim = range(0, y_dim);

    y_permute_dim = [pop(y_permute_dim, y_permute_dim.length - 2)].concat(y_permute_dim);

    var xt = tf.reshape(x, [x.size/x.shape[x_dim - 1], x.shape[x_dim - 1]]);
    var yt = tf.reshape(
        tf.transpose(y, y_permute_dim),
        [y.shape[y_dim - 2], y.size/y.shape[y_dim - 2]]
    );
    return tf.reshape(
        tf.dot(xt, yt),
        x.shape.slice(0, x_dim - 1).concat(y.shape.slice(0, y_dim - 2)).concat(y.shape.slice(y_dim - 1))
    );


}

export function gelu(x) {
    return tf.mul(
        tf.mul(0.5, x), tf.add(
            1.0, tf.tanh(
                tf.mul(
                    Math.sqrt(2.0/Math.PI), tf.add(
                        x, tf.mul(
                            0.044715, tf.mul(
                                x, tf.mul(x, x)
                            )
                        )
                    )
                )
            )
        )
    )
}

export function enumerate(list) {
    return zip(range(0, list.length), list);
}
    
export function zip(arr1, arr2) {
    return arr1.map(function(e, i) {
        return [e, arr2[i]];
    })
}

export function range(start, finish) {
    var size = finish - start;
    return [...Array(size).keys()].map(i => i + start);
}

export function pop(arr, i) {
    var removed = arr.splice(i, 1);
    return removed[0]
}

export function swap(json){
    var ret = {};
    for(var key in json){
      ret[json[key]] = key;
    }
    return ret;
}

export function strip(str) {
    return str.replace(/^\s+|\s+$/g, '');
}

export function arraysEqual(a1,a2) {
    /* WARNING: arrays must not contain {objects} or behavior may be undefined */
    return JSON.stringify(a1)==JSON.stringify(a2);
}
