var tf = require("@tensorflow/tfjs-node");
var fs = require('fs');
var Sampler = require('./sampler').Sampler;
require('longjohn');

var EmbeddingRet = require('./embedding_ret').EmbeddingRet;
var PositionEmbedding = require('./position_embedding').PositionEmbedding;
var MultiHeadAttention = require('./multi_head_attention').MultiHeadAttention;
var FeedForward = require('./feed_forward').FeedForward;
var LayerNormalization = require('./layer_normalization').LayerNormalization;
var EmbeddingSim = require('./embedding_sim').EmbeddingSim;

function getModel(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, batch_size=null, fixed_input_shape=false) {

    var inputLayer = tf.layers.input({shape: (batch_size, n_ctx)});

    var embeddingRet = new EmbeddingRet({
        inputDim:n_vocab,
        outputDim:n_embd,
        maskZero:false,
        name:'Embed-Token'
    }).apply(inputLayer);

    var positionEmbedding = new PositionEmbedding(
        n_ctx,
        n_embd,
        PositionEmbedding.MODE_ADD,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {name:'Embed-Token-Pos'}
    ).apply(embeddingRet[0]);

    var lastLayer = getEncoderComponent(
        'Encode-1',
        positionEmbedding,
        n_head,
        n_embd*4,
        null,
        gelu,
    )

    // var normLayer = new LayerNormalization(
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     {name:'Norm'}
    // ).apply(lastLayer)

    // var outputLayer = new EmbeddingSim(
    //     false,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     {name:'Output'}
    // ).apply([normLayer, embeddingRet[1]])

    const model = tf.model({inputs:inputLayer, outputs:lastLayer});
    return model;
}

function getEncoderComponent(name, inputLayer, headNum, hiddenDim, attentionActivation=null, feedForwardActivation=tf.relu, trainable=true) {

    var attentionName = name.concat('-MultiHeadAtt');
    var feedForwardName = name.concat('-FeedForward');

    //console.log(inputLayer, "myinp[ut");
    var attentionLayer = wrapLayer(
        attentionName,
        inputLayer,
        attentionBuilder(
            attentionName,
            headNum,
            attentionActivation,
            true,
            trainable
        ),
        trainable
    );

    var feedForwardLayer = wrapLayer(
        feedForwardName,
        attentionLayer,
        feedForwardBuilder(
            feedForwardName,
            hiddenDim,
            feedForwardActivation,
            trainable
        ),
        trainable,
        true
    );

    // var feedForwardLayer = new FeedForward(
    //     hiddenDim,
    //     feedForwardActivation,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     undefined,
    //     {name:name, trainable:trainable}
    // ).apply(attentionLayer);

    return feedForwardLayer;
}

function wrapLayer(name, inputLayer, buildFunc, trainable=true, fake=false) {
    var normalLayer = new LayerNormalization(
        undefined,
        undefined, 
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {trainable:trainable, name:name.concat('-Norm')}
    ).apply(inputLayer);

    //console.log(inputLayer, "input");
    //console.log(normalLayer, "normal")
    var buildOutput = buildFunc(normalLayer);


    
    var addLayer = tf.layers.add({name:name.concat('-Add')});
    if (fake) {
        console.log("SANITY");
        addLayer.call = function(inputs) {
            console.log(inputs)
            return tf.add(inputs[0], inputs[1]);
        }
    }
    return addLayer.apply([inputLayer, buildOutput]);
    
    
}

function attentionBuilder(name, headNum, activation, historyOnly, trainable=true) {

    function attentionBuilderHelper(x) {
        return new MultiHeadAttention(
            headNum,
            activation,
            historyOnly,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            {name:name, trainable:trainable}
        ).apply(x);
    }

    return attentionBuilderHelper;
}

function feedForwardBuilder(name, hiddenDim, activation, trainable=true) {

    function feedForwardBuilderHelper(x) {
        return new FeedForward(
            hiddenDim,
            activation,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            {name:name, trainable:trainable}
        ).apply(x);
    }

    return feedForwardBuilderHelper;

}

function gelu(x) {
    return tf.mul(
        tf.mul(0.5, x), tf.add(
            1.0, tf.tanh(
                tf.mul(
                    Math.sqrt(2.0/Math.pi), tf.add(
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

model = getModel();


var chunks = JSON.parse(fs.readFileSync("output.txt", "utf-8"));

var dataSampler = new Sampler(chunks);

var sampleBatch = dataSampler.sampleBatch(2);

var x  = sampleBatch;

var y = [sampleBatch[0].slice(1), sampleBatch[0].slice(1)];

console.log(model)

console.log(model.predict(tf.tensor(x)));