var tf = require('@tensorflow/tfjs');
var assert = require('assert');

function split_heads(x) {
    // From [batch, sequence, features] to [batch, heads, sequence, features]
    return tf.transpose(this.split_states(x, hparams.n_head), [0, 2, 1, 3])
}

function merge_heads(x) {
    // Reverse of split_heads
    return this.merge_states(tf.transpose(x, [0, 2, 1, 3]))
}

function mask_attn_weights(w) {
    // w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    var arr = this.shape_list(w);
    var nd = arr[arr.length - 2];
    var ns = arr[arr.length - 1];
    var b = this.attention_mask(nd, ns, dtype=w.dtype)
    b = tf.reshape(b, [1, 1, nd, ns])
    var w = tf.sub(tf.mul(w,b) - tf.mul(tf.cast(1e10, w.dtype)*(tf.sub(1, b))))
    return w
}

function multihead_attn (q, k, v) {
    // q, k, v have shape [batch, heads, sequence, features]
    var w = tf.matMul(q, k, transpose_b=true)
    w = tf.mul(w, tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype)))

    w = this.mask_attn_weights(w)
    w = this.softmax(w, w.shape.length)
    var a = tf.matMul(w, v)
    return a
}

class Model {
    constructor () {
        this.variables = [];
    }

    default_hparams = {
        n_vocab:50257,
        n_ctx:1024,
        n_embd:768,
        n_head:12,
        n_layer:12
    }

    shape_list(x) {
        return x.shape
    }

    softmax(x, axis=-1) {
        if (axis == -1) {
            axis = x.shape.length - 1
        }
        x = tf.sub(x, tf.max(x, axis, true))
        var ex = tf.exp(x)
        return tf.div(ex, tf.sum(ex, axis, true))
    }

    gelu(x) {
        return 0.5*x*(1+tf.tanh(Math.sqrt(2/Math.pi)*(x+0.044715*tf.pow(x, 3))))
    }

    norm(x, axis=-1, epsilon=1e-5) {
        if (axis == -1) {
            axis = x.shape.length - 1
        }

        var n_state = x.shape[x.shape.length - 1]
        var g = tf.variable(tf.initializers.ones().apply([n_state], 'float32'))
        var b = tf.variable(tf.initializers.zeros().apply([n_state], 'float32'))
        this.variables.push(g);
        this.variables.push(b);
        var u = tf.mean(x, axis, true)
        var s = tf.mean(tf.square(tf.sub(x,u)), axis, true)
        x = tf.mul(tf.sub(x, u) , tf.rsqrt(tf.add(s, epsilon)))
        x = tf.add(tf.mul(x,g), b)
        return x;
    }

    split_states(x, n) {
        var arr = shape_list(x)
        var start = arr.slice(0, arr.length - 1);
        var m = arr[arr.length - 1];
        return tf.reshape(x, start.concat([n, Math.floor(m/n)]));
    }

    merge_states(x) {
        var arr = shape_list(x);
        var start = arr.slice(0, arr.length - 2);
        var a = arr[arr.length - 2];
        var b = arr[arr.length - 1];
        return tf.reshape(x, start.concat([tf.mul(a,b)]));
    }

    conv1d(x, nf, w_init_stdev=0.02) {
        var arr = this.shape_list(x);
        var start = arr.slice(0, arr.length - 1);
        var nx = arr[arr.length - 1];
        var w = tf.variable(tf.initializers.randomNormal({mean:0, stddev:w_init_stdev}).apply([1, nx, nf], 'float32'));
        var b = tf.variable(tf.initializers.zeros().apply([nf], "float32"));
        this.variables.push(w);
        this.variables.push(b);
        console.log(nx, nf, x.size, w.size, x.size/nx, w.size/nf)
        var c = tf.reshape(tf.add(tf.matMul(tf.reshape(x, [x.size/nx, nx]), tf.reshape(w, [w.size/nf, nf])), b), start.concat([nf]));
        return c;            
    }

    attention_mask(nd, ns, dtype) {
        var i = tf.range(0, nd).arraySync();
        i2 = []
        i = i.forEach(element => {
            i2.push([element])
        })
        i = i2
        i = tf.tensor(i)
        var j = tf.range(0, ns)
        var m = i >= j - ns + nd
        return tf.cast(m, dtype)
    }

    

    attn(x, n_state, past=null, hparams=null) {
        assert (x.shape.length == 3);  
        assert (n_state % hparams.n_head == 0);
        if (past != null)
            assert(past.shape.length == 5)  // Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        var c = this.conv1d(x, n_state*3)
        var split_heads
        var result_arr = tf.split(c, 3, 2).map(this.split_heads);
        var q = result_arr[0];
        var k = result_arr[1];
        var v = result_arr[2];
        var present = tf.stack([k, v], 1)
        if (past != null) {
            result_arr = tf.unstack(past, axis=1);
            var pk = result_arr[0];
            var pv = result_arr[1];
            k = tf.concat([pk, k], -2);
            v = tf.concat([pv, v], -2);
        }
        var a = this.multihead_attn(q, k, v)
        a = this.merge_heads(a)
        a = this.conv1d(a, n_state)
        return [a, present]
            
    }


    mlp(x, n_state, hparams) {
        var nx = x.shape[x.shape.length - 1].value
        var h = gelu(conv1d(x, n_state))
        var h2 = conv1d(h, nx)
        return h2
    }


    block(x, past, hparams) {
        console.log(this)
        var nx = x.shape[x.shape.length - 1]
        var result_arr = this.attn(this.norm(x), nx, past=past, hparams=hparams)
        var a = result_arr[0];
        var present = result_arr[1];
        var x = x + a
        var m = this.mlp(norm(x), nx*4, hparams=hparams)
        x = x + m
        return [x, present]
    }
        

    past_shape(hparams, batch_size=null, sequence=null) {
        return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, Math.floor(hparams.n_embd / hparams.n_head)]
    }
    
    expand_tile(value, size) {
        //value = tf.convert_to_tensor(value)
        var ndims = value.shape.length
        return tf.tile(tf.expandDims(value), [size].concat(new Array(ndims).fill(1)))
    }

    positions_for(tokens, past_length) {
        var batch_size = tokens.shape[0]
        var nsteps = tokens.shape[1]
        return this.expand_tile(tf.tensor(tf.range(past_length, nsteps + past_length).arraySync()), batch_size)
    }

    model(hparams, X, past=null, reuse=false) {
        var results = {}
        var result_arr = this.shape_list(X);
        var batch = result_arr[0];
        var sequence = result_arr[1];
        var wpe = tf.variable(tf.initializers.randomNormal({mean:0, stddev:0.01}).apply([hparams.n_ctx, hparams.n_embd]));
        var wte = tf.variable(tf.initializers.randomNormal({mean:0, stddev:0.02}).apply([hparams.n_vocab, hparams.n_embd]));

        this.variables.push(wpe);
        this.variables.push(wte);

        var past_length;
        if (past == null) {
            past_length = 0;
        } else {
            past_length = past.shape[past.shape.length - 2]
        }
        
        var h = tf.add(tf.gather(wte, X), tf.gather(wpe, tf.cast(this.positions_for(X, past_length), "int32")))

        // Transformer
        var presents = [];
        var pasts; 
        if (past != null) {
            pasts = tf.unstack(past, axis=1).arraySync()
        } else {
            pasts = new Array(hparams.n_layer).fill(null);
        }
        assert (pasts.length == hparams.n_layer);
        
        console.log(this)
        for (var i = 0; i < pasts.length; i++) {
            past = pasts[i];
            result_arr = this.block(h, past=past, hparams = hparams);
            h = result_arr[0];
            presents.push(result_arr[1]);
        };
        
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        // Language model loss.  Do tokens <n predict token n?
        var h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        var logits = tf.matMul(h_flat, wte, transpose_b=true)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results            
    }

    loss_function(labels, x) {
        var output = this.model(this.default_hparams, x);
        var logits = [];
        output["logits"].arraySync().forEach(element => {
            logits.push(element.slice(0, element.length - 1));
        })
        var loss = tf.mean(
            tf.losses.softmaxCrossEntropy(labels, logits)
        )
        return loss
    }
}

module.exports = {
    Model: Model
}
