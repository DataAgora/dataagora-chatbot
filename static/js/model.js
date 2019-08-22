import {EmbeddingRet} from './model/embedding_ret.js';
import {PositionEmbedding} from './model/position_embedding.js';
import {MultiHeadAttention} from './model/multi_head_attention.js';
import {FeedForward} from './model/feed_forward.js';
import {LayerNormalization} from './model/layer_normalization.js';
import {EmbeddingSim} from './model/embedding_sim.js'
import {gelu, range} from './model/utils.js';
import {get_encoder} from './encoder.js';

/*
MODEL
    - The language model that the chatbot uses: built using TFJS, with weights loaded from S3
    - Takes preprocessed input text -> encodes -> feeds into model -> decodes output
    - Based off the Keras GPT-2 model at https://github.com/CyberZHG/keras-gpt-2/tree/master/keras_gpt_2
*/

export class Model {
    constructor(model) { 
        this.model = model;
        var layer1 = model.layers[1];
        layer1.embeddingObject.setWeights(layer1.getWeights());
        this.store = false;    
    }

    // static async getPartialModel(model, layerNum) {
    //     // var new_model = tf.model({inputs:model.inputs[0], outputs:model.layers[layerNum].output});
    //     // await Model.getSetWeights(new_model, layerNum);
    //     return new_model;
    // }

    static async getSetWeights(model, maxLayer=76) {
        var baseUrl = 'https://dataagora-chatbot.s3-us-west-1.amazonaws.com/weights_12/weights_';
        for (var i = 0; i <= maxLayer; i++) {
            var layer = model.layers[i];
            var weightsLength = layer.getWeights().length;
            if (weightsLength != 0) {
                var fullUrl = baseUrl.concat(i + ".json");
                var layerWeights = await fetch(fullUrl);
                layerWeights = await layerWeights.json();
                layer.setWeights(layerWeights.map(weightArr => {
                    return tf.tensor(weightArr);
                }));
            }            
        };
    }

    async forwardPass(texts, top_k = 1, temperature=1) {
        var batch_size = texts.length;
        var model = this.model;
        var encoder = await get_encoder();
        var encodings = encoder.encode(texts[0]);
        var text_lens = [encodings.length];
        var max_len = Math.max(text_lens)
        var input_data = [encodings];
        var textLen = 21;
        console.log(texts)
        range(0, textLen).forEach(shift => {
            var output_data = model.predict(tf.tensor(input_data)).arraySync();
            //console.log(output_data);
            range(0, batch_size).forEach(index => {
                var probs = tf.tensor(output_data[index][max_len + shift - 1]);
                var {values, indices} = tf.topk(probs, top_k);
                probs = values;
                probs = tf.div(probs, temperature);
                probs = tf.sub(probs, tf.max(probs));
                probs = tf.exp(probs);
                probs = tf.div(probs, probs.sum());
                var next_token = indices.arraySync()[0];
                input_data[index].push(0);
                input_data[index][max_len + shift] = next_token;
                console.log(next_token);
            })
        });
    
        var outputs = encoder.decode(input_data[0].slice(max_len, max_len + textLen));
        
        console.log(outputs);
        return outputs;
    }

    async backwardsPass(texts) {
        var batch_size = texts.length;
        var model = this.model;
        var encoder = await get_encoder();
        var encodings = encoder.encode(texts[0]);
        var text_lens = [encodings.length];
        var max_len = Math.max(text_lens)
        var input_data = [encodings];
        var model = this.model;
        var modelJSON = await fetch('weights/model.json');
        modelJSON = await modelJSON.json();
        var optimization_data = modelJSON['modelTopology']['training_config'];
        optimization_data['loss'] = 'categoricalCrossentropy';
        model = this.compileModel(model, optimization_data);
        var encodedOutput = model.predict(tf.tensor(input_data))
        console.log(encodedOutput);
        var start = Date.now();
        var results = await model.trainOnBatch(tf.tensor(input_data), encodedOutput);
        var end = Date.now();
        console.log("Number of minutes: ", (end - start)/60000);
        console.log(results);
        console.log("Done!")
    }

    compileModel(model, optimization_data) {
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

    getModel(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, batch_size=null, fixed_input_shape=false) {

        var inputLayer = tf.layers.input({shape: [null]});
    
        var embeddingRet = new EmbeddingRet({
            inputDim:n_vocab,
            outputDim:n_embd,
            maskZero:false,
            name:'Embed-Token'
        }); 
        
        console.log(embeddingRet)
        embeddingRet = embeddingRet.apply(inputLayer);

        console.log(embeddingRet);
    
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
            lastLayer = this.getEncoderComponent(
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
    
    getEncoderComponent(name, inputLayer, headNum, hiddenDim, attentionActivation=null, feedForwardActivation=tf.relu, trainable=true) {
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
    
        feedForwardLayer = addLayer.apply([feedForwardLayer, lastoLayer]);
    
        return feedForwardLayer;
    }
}