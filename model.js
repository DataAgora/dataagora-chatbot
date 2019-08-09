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
import {get_encoder, range, enumerate} from './encoder.js';

// function setWeights(model, weights) {
//     model.layers.forEach(layer => {
//         var layerWeights = JSON.parse(weights.pop());
//         //console.log(layerWeights.length, layerWeights[0].length)
//         if (layerWeights.length != 0) {
//             //console.log(layerWeights)
//             layerWeights = layerWeights.map(arr => {
//                 return tf.tensor(arr);
//             });
//             //console.log(layer.getWeights());
            
            
//             layer.setWeights(layerWeights);
//         } 
//     });  
// }

async function getWeights() {
    var baseUrl = 'http://localhost:8000/weights/my_weights_';
    var modelWeights = []
    for (var i = 22; i >= 0; i--) {
        var fullUrl = baseUrl.concat(i + ".txt");
        var layerWeights = await fetch(fullUrl);
        layerWeights = await layerWeights.json();
        if (layerWeights.length == 0) {
            modelWeights.push([]);
        } else {
            modelWeights.push(layerWeights.map(weight => {
                return tf.tensor(weight);
            }));
        }
    };
    
    return modelWeights
}

function setWeights(model, weights) {
    //console.log(weights)
    model.layers.forEach(layer => {
        layer.setWeights(weights.pop());
    });  
}

function getModel(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=3, batch_size=null, fixed_input_shape=false) {

    var inputLayer = tf.layers.input({shape: [null]});

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
    
    for (var i = 0; i < n_layer; i++) {
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

    feedForwardLayer = addLayer.apply([feedForwardLayer, normalLayer]);

    return feedForwardLayer;
}

// async function getWeights() {
//     var modelWeights = await fetch('http://localhost:8000/my_weights.txt');
//     modelWeights = await modelWeights.text();
//     modelWeights = modelWeights.split('\n')
//     modelWeights.pop();
//     modelWeights = modelWeights.reverse();
//     return modelWeights
// }

async function forwardPass(text) {
    var texts = ["How are you?"];
    var batch_size = texts.length;
    var modelWeights = await getWeights();
    var model = getModel();
    console.log(model);
    setWeights(model, modelWeights);
    model.save('downloads://chatbot')
    var encoder = await get_encoder();
    console.log(encoder);
    var encodings = encoder.encode(texts[0]);
    var text_lens = [encodings.length];
    var max_len = Math.max(text_lens)
    var input_data = [encodings];
    var textLen = 10;
    range(0, textLen).forEach(shift => {
        var output_data = model.predict(tf.tensor(input_data)).arraySync();
        range(0, batch_size).forEach(index => {
            // console.log(output_data);
            // console.assert(false);
            console.log(output_data)
            var probs = enumerate(output_data[index][max_len + shift - 1]).map(pair => {
                return [pair[1], pair[0]];
            });
            probs = probs.sort().reverse();
            probs = probs[0];
            var indices = [probs[1]];
            probs = [probs[0]];
            probs = tf.tensor1d(probs);
            probs = tf.sub(probs, tf.max(probs));
            probs = tf.exp(probs);
            probs = tf.div(probs, probs.sum());
            var next_token = indices[0];
            input_data[index].push(0);
            input_data[index][max_len + shift] = next_token;
            console.log(next_token)
            console.log(input_data)
        })
    });

    var outputs = encoder.decode(input_data[0].slice(max_len, max_len + textLen));

    console.log(outputs);




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
// getWeights()