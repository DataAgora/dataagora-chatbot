var fs = require('fs');
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

var encode_js = require('./encoder');
var Sampler = require('./sampler').Sampler;

var file = fs.readFileSync('shakespeare.txt', 'utf8')
var get_encoder = encode_js.get_encoder;

//var Model = require('./logits').Model;

var encoder = get_encoder();

//var chunks = encoder.load_dataset(file);
var chunks = JSON.parse(fs.readFileSync("output.txt", "utf-8"));

var dataSampler = new Sampler(chunks);

var sampleBatch = dataSampler.sampleBatch(2);

var x  = sampleBatch;
console.log([sampleBatch.length, sampleBatch[0].length])
var y = [sampleBatch[0].slice(1), sampleBatch[0].slice(1)];

console.log("X", x)
console.log("Y", y)

async function loadModel () {
    console.log("Loading model...")
    const handler = tfnode.io.fileSystem('model/model.json');
    const model = await tf.loadLayersModel(handler);
    console.log(model.layers[1].batchInputShape)
    var model_json = fs.readFileSync('model/model.json')
    var optimization_data = JSON.parse(model_json)['modelTopology']['training_config']
    //optimizer_config['loss'] = 'softmaxCrossEntropy'
    optimization_data['loss'] = 'categoricalCrossentropy'

    sampleBatch[0] = tf.tensor(sampleBatch[0])
    sampleBatch[1] = tf.tensor(sampleBatch[1])
    //var my_model = new Model();
    //var logits = my_model.model(my_model.default_hparams, x_tensor);
    
    await _compileModel(model, optimization_data)
    console.log("Training start...")
    await model.fit(sampleBatch, tf.tensor(y), {batchSize:1})
    console.log("Training Complete!")
    // console.log(model)
}

async function _compileModel(model, optimization_data) {
    var optimizer;
    var optimizer_config = optimization_data['optimizer_config']
    if (optimization_data['optimizer_config']['class_name'] == 'SGD') {
        // SGD
        optimizer = tf.train.sgd(optimization_data['optimizer_config']['config']['lr']);
    } else if (optimization_data['optimizer_config']['class_name'] == 'Adam') {
        optimizer = tf.train.adam(optimizer_config['config']['lr'], optimizer_config['config']['beta1'], optimizer_config['config']['beta2']);
    } else {
        // Not supported!
        throw "Optimizer not supported!";
    }

    model.compile({
        optimizer: optimizer,
        loss: optimization_data['loss'],
        metrics: optimization_data['metrics']
    });
    //console.log("Model compiled!", model);
    return model;
};

loadModel()