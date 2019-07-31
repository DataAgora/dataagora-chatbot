var tf = require("@tensorflow/tfjs-node");
var fs = require('fs');
var Sampler = require('./sampler').Sampler;

var EmbeddingRet = require('./embedding_ret').EmbeddingRet;
var EmbeddingSim = require('./embedding_sim').EmbeddingSim;

function getModel(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, batch_size=null, fixed_input_shape=false) {
    var inputLayer = tf.layers.input({shape: (batch_size, n_ctx)})
    var layer2 = new EmbeddingRet({
        inputDim:n_vocab,
        outputDim:n_embd,
        maskZero:false,
        name:'Embed-Token'
    }).apply(inputLayer)
    const model = tf.model({inputs:inputLayer, outputs:layer2});
    return model;
}

model = getModel();


var chunks = JSON.parse(fs.readFileSync("output.txt", "utf-8"));

var dataSampler = new Sampler(chunks);

var sampleBatch = dataSampler.sampleBatch(2);

var x  = sampleBatch;
console.log([sampleBatch.length, sampleBatch[0].length])
var y = [sampleBatch[0].slice(1), sampleBatch[0].slice(1)];

console.log(model.predict(tf.tensor(x)));