import {EmbeddingRet} from './embedding_ret.js';
import {PositionEmbedding} from './position_embedding.js';
import {MultiHeadAttention} from './multi_head_attention.js';
import {FeedForward} from './feed_forward.js';
import {LayerNormalization} from './layer_normalization.js';
import {EmbeddingSim} from './embedding_sim.js'

// // // console.log(tf.initializers.randomUniform({
// // //         minval:-0.05, 
// // //         maxval:0.05
// // //     }).apply([1])
// // // )

// // var a = tf.initializers.glorotNormal();
// // console.log(a.apply([1, 2]))

// async function loadModel() {
//     var a = await tf.loadLayersModel("http://localhost:8000/chatbot/chatbot.json")
//     console.log(a);
// }

// loadModel()

async function doStuff() {
    const model = createConvModel();
    console.log('Prediction from original model:');
    console.log(model.layers[0].getWeights()[0].arraySync())
    console.log(model);
    const saveResults = await model.save(tf.io.browserHTTPRequest(
        'http://localhost:5002/upload',
        {method: 'POST', headers: {'Access-Control-Allow-Origin': '*'}}));
    const myBinaryFile = await fetch('http://localhost:5002/model.weights.bin')    
    const myBuffer = await myBinaryFile.arrayBuffer() 
    
    const dv = new DataView(myBuffer);
    var f32 = new Float32Array(myBuffer.byteLength / 4);
    const littleEndian = true;
    
    console.log("HERe", f32.length);
    for (let i = 0; i < f32.length; i++) {
        f32[i] = dv.getFloat32(i*4, littleEndian);
    }   

    const myWeightsJSON = JSON.stringify(f32,null,4);

    console.log(myWeightsJSON);
    const loadedModel = await tf.loadLayersModel('http://localhost:5002/chatbot12.json');
    console.log(loadedModel.layers[0].getWeights()[0].arraySync())
    console.log('Prediction from loaded model:');
}

async function doStuff2() {
    var model = await tf.loadLayersModel('http://localhost:5002/chatbot12.json');
    console.log(model)
    console.log('Original model:');
    console.log("first model weight", model.layers[1].getWeights()[0].arraySync()[0][0]);
    // model.layers.forEach(layer => {
    //     layer.setWeights(
    //         layer.getWeights().map(weightArr => {
    //             return tf.initializers.zeros().apply(weightArr.shape);
    //         })
    //     )
    // });
    console.log("new first model weight", model.layers[1].getWeights()[0].arraySync()[0][0]);
    //model.predict(tf.ones([1, 3])).print();
    
    async function stuff3(i) {
        const saveResults = await model.save(tf.io.browserHTTPRequest(
            'http://localhost:5002/upload',
            {method: 'POST', headers: {'Access-Control-Allow-Origin': '*'}}));
        const myBinaryFile = await fetch('http://localhost:5002/model.weights.bin')    
        const myBuffer = await myBinaryFile.arrayBuffer() 
        console.log('Saved model:', i);
        const dv = new DataView(myBuffer);
        var f32 = new Float32Array(500);
        const littleEndian = true;
        
        for (let i = 0; i < 500; i++) {
            f32[i] = dv.getFloat32(i*4, littleEndian);
        }   
    
        console.log("first bin weight", f32[0]);
        const myWeightsJSON = JSON.stringify(f32,null,4);
    
        //console.log(myWeightsJSON)
        
        model = await tf.loadLayersModel('http://localhost:5002/chatbot12.json');
        console.log("first model weight", model.layers[1].getWeights()[0].arraySync()[0][0])
        
    }

    for (var j = 1; j < 1; j++) {
        await stuff3(j);
    }
    // loadedModel.predict(tf.ones([1, 3])).print();
}

export function createModel(sampleLen, charSetSize, lstmLayerSizes) {
    if (!Array.isArray(lstmLayerSizes)) {
      lstmLayerSizes = [lstmLayerSizes];
    }
  
    const model = tf.sequential();
    for (let i = 0; i < lstmLayerSizes.length; ++i) {
      const lstmLayerSize = lstmLayerSizes[i];
      model.add(tf.layers.lstm({
        units: lstmLayerSize,
        returnSequences: i < lstmLayerSizes.length - 1,
        inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
      }));
    }
    model.add(
      tf.layers.dense({ units: charSetSize, activation: 'softmax' }));
  
    return model;
}

function createConvModel() {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    const model = tf.sequential();
  
    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_H, IMAGE_W, 1],
      kernelSize: 3,
      filters: 16,
      activation: 'relu'
    }));
  
    // After the first layer we include a MaxPooling layer. This acts as a sort of
    // downsampling using max values in a region instead of averaging.
    // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  
    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  
    // Max pooling again.
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  
    // Add another conv2d layer.
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({}));
  
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
  
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  
    return model;
  }

async function stuffo() {
  var received = await fetch('https://dataagora-chatbot.s3-us-west-1.amazonaws.com/weights/weights_1_0.json');
  received = await received.json();
  console.log(received);
}

stuffo();
