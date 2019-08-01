var tf = require("@tensorflow/tfjs-node");
var fs = require('fs');
var Sampler = require('./sampler').Sampler;

var EmbeddingRet = require('./embedding_ret').EmbeddingRet;
var PositionEmbedding = require('./position_embedding').PositionEmbedding;
var EmbeddingSim = require('./embedding_sim').EmbeddingSim;

function getModel(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, batch_size=null, fixed_input_shape=false) {
    var inputLayer = tf.layers.input({shape: (batch_size, n_ctx)})
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
    ).apply(embeddingRet);
    const model = tf.model({inputs:inputLayer, outputs:positionEmbedding});
    return model;
}

model = getModel();


var chunks = JSON.parse(fs.readFileSync("output.txt", "utf-8"));

var dataSampler = new Sampler(chunks);

var sampleBatch = dataSampler.sampleBatch(2);

var x  = sampleBatch;

var y = [sampleBatch[0].slice(1), sampleBatch[0].slice(1)];

console.log(model)

console.log(model.predict(tf.tensor(x)));