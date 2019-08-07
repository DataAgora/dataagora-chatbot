// var tf = require("@tensorflow/tfjs-node");
// var fs = require('fs');
// var Sampler = require('./sampler').Sampler;
// var longjohn = require('longjohn');

// var EmbeddingRet = require('./embedding_ret').EmbeddingRet;
// var PositionEmbedding = require('./position_embedding').PositionEmbedding;
// var MultiHeadAttention = require('./multi_head_attention').MultiHeadAttention;
// var FeedForward = require('./feed_forward').FeedForward;
// var LayerNormalization = require('./layer_normalization').LayerNormalization;
// var DummyLayer = require('./dummy_layer').DummyLayer;

// var EmbeddingSim = require('./embedding_sim').EmbeddingSim;

import {EmbeddingRet} from './embedding_ret.js';
import {PositionEmbedding} from './position_embedding.js';
import {MultiHeadAttention} from './multi_head_attention.js';
import {FeedForward} from './feed_forward.js';
import {LayerNormalization} from './layer_normalization.js';
import {EmbeddingSim} from './embedding_sim.js'
import {Sampler} from './sampler.js';
import {gelu} from './utils.js';
import {get_encoder} from './encoder.js';

function getModel(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, batch_size=null, fixed_input_shape=false) {

    var inputLayer = tf.layers.input({shape: (batch_size, n_ctx)});

    var embeddingRet = new EmbeddingRet({
        inputDim:n_vocab,
        outputDim:n_embd,
        maskZero:false,
        name:'Embed-Token'
    }).apply(inputLayer);

    var embedToken = embeddingRet[0];
    var embeddings = embeddingRet[1];

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
    ).apply(embedToken);

    var lastLayer = positionEmbedding
    
    for (var i = 0; i < 1; i++) {
        lastLayer = getEncoderComponent(
            'Encode-'.concat([i + ""]),
            lastLayer,
            n_head,
            n_embd*4,
            null,
            gelu,
        )
    }

    var normLayer = new LayerNormalization(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {name:'Norm'}
    ).apply(lastLayer)

    var outputLayer = new EmbeddingSim(
        false,
        undefined,
        undefined,
        undefined,
        undefined,
        {name:'Output'}
    ).apply([normLayer, embeddings])

    const model = tf.model({inputs:inputLayer, outputs:outputLayer});
    return model;
}

function getEncoderComponent(name, inputLayer, headNum, hiddenDim, attentionActivation=null, feedForwardActivation=tf.relu, trainable=true) {

    var attentionName = name.concat('-MultiHeadAtt');
    var feedForwardName = name.concat('-FeedForward');

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
        {trainable:trainable, name:name.concat('-Norm1')}
    ).apply(inputLayer);

    var attentionLayer = new MultiHeadAttention(
        headNum,
        attentionActivation,
        true,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {name:name.concat('-atten'), trainable:trainable}
    ).apply(normalLayer);

    var addLayer = tf.layers.add({name:name.concat('-Add1')});

    var lastoLayer = addLayer.apply([inputLayer, attentionLayer]);

    normalLayer = new LayerNormalization(
        undefined,
        undefined, 
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {trainable:trainable, name:name.concat('-Norm2')}
    ).apply(lastoLayer);

    var feedForwardLayer = new FeedForward(
        hiddenDim,
        feedForwardActivation,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {name:name.concat('-feed'), trainable:trainable}
    ).apply(normalLayer);

    

    addLayer = tf.layers.add({name:name.concat('-Add2')});

    return feedForwardLayer;
}

async function forwardPass(text) {
    // var model = getModel();
    var encoder = await get_encoder();
    console.log(encoder)
    var encodings = encoder.encode("Hi, my name is Bill. What's your name?");
    console.log(encodings);
    console.log(encoder.decode(encodings))
    // var chunks = data;

    // var dataSampler = new Sampler(chunks);

    // var sampleBatch = dataSampler.sampleBatch(1);

    // var x  = sampleBatch;

    // console.log(sampleBatch)

    // console.log(model)

    // var result = model.predict(tf.tensor(x));

    // var array = result.dataSync();

    // console.log(array);
}

forwardPass();