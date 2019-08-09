// var tf = require('@tensorflow/tfjs-node');

export class PositionEmbedding extends tf.layers.Layer {
    static MODE_EXPAND = 'expand';
    static MODE_ADD = 'add';
    static MODE_CONCAT = 'concat';

    constructor(inputDim, outputDim, mode=PositionEmbedding.MODE_EXPAND, embeddingsInitializer = tf.initializers.randomUniform({minval:-0.05, maxval:0.05}),
        embeddingsRegularizer = null, activityRegularizer = null, embeddingsConstraint=null, maskZero=false) {

        // if (!(name) in ...args) {
        //     ...args[name] = "defaultPosition"
        // }
        console.log(inputDim, outputDim)
        super({});

        this.inputDim = inputDim;
        this.outputDim =  outputDim;
        this.mode = mode;
        this.embeddingsInitializer = embeddingsInitializer;
        this.embeddingsRegularizer = embeddingsRegularizer;
        this.activityRegularizer = activityRegularizer;
        this.embeddingsConstraint = embeddingsConstraint;
        this.maskZero = maskZero;
        this.supportsMasking = maskZero;

        this.embeddings = null;
    }

    getConfig() {
        var baseConfig = super.getConfig();
        baseConfig['inputDim'] = this.inputDim;
        baseConfig['outputDim'] = this.outputDim;
        baseConfig['mode'] = this.mode;
        baseConfig['embeddingsInitializer'] = this.embeddingsInitializer;
        baseConfig['maskZero'] = this.maskZero;
        return baseConfig;
    }

    build(inputShape) {
        super.build(inputShape);
        if (this.mode == PositionEmbedding.MODE_EXPAND) {
            this.embeddings = this.addWeight(
                'embeddings',
                [this.inputDim * 2 + 1, this.outputDim],
                undefined,
                this.embeddingsInitializer
            );
        } else {
            this.embeddings = this.addWeight(
                'embeddings',
                [this.inputDim, this.outputDim],
                undefined,
                this.embeddingsInitializer
            );
        }
        
    }

    computeMask(inputs, mask=null) {
        if (this.mode == PositionEmbedding.MODE_EXPAND) {
            if (this.maskZero) {
                return tf.notEqual(inputs, this.maskZero)
            } else {
                return null;
            }
        } else {
            return mask;
        }
    }

    computeOutputShape(inputShape) {
        if (this.mode == PositionEmbedding.MODE_EXPAND) {
            return inputShape.concat([this.outputDim]);
        } else if (this.mode == PositionEmbedding.MODE_CONCAT) {
            return inputShape.slice(0, inputShape.length - 1).concat([inputShape[inputShape.length - 1] + this.outputDim]);
        } else {
            //console.log(inputShape, "rip")
            return inputShape;
        }
    }

    call(inputs, ...args) {
        //console.log(inputs, "hey")
        if (this.mode == PositionEmbedding.MODE_EXPAND) {
            if (inputs.dtype != 'int32') {
                inputs = tf.cast(inputs, "int32");
            }
            return tf.gather(
                this.embeddings,
                tf.add(tf.minimum(tf.maximum(inputs, -1 * this.inputDim), this.inputDim), (this.inputDim))
            )
        }
        var inputShape = inputs[0].shape;
        var batchSize, seqLen, outputDim;
        if (this.mode == PositionEmbedding.MODE_ADD) {
            batchSize = inputShape[0];
            seqLen = inputShape[1];
            outputDim = inputShape[2];
        } else {
            batchSize = inputShape[0];
            seqLen = inputShape[1];
            outputDim = this.outputDim;
        }

        var data = this.embeddings.val.arraySync();
        var tempEmbeddings = [];
        for (var i = 0; i < seqLen; i++) {
            tempEmbeddings.push(data[i].slice(0, this.outputDim))
        }
        data = tempEmbeddings;
        tempEmbeddings = tf.tensor(tempEmbeddings);
        var posEmbeddings = tf.tile(
            tf.expandDims(tempEmbeddings, 0),
            [batchSize, 1, 1]
        )
        //console.log("HELLO")
        if (this.mode == PositionEmbedding.MODE_ADD) {
            return inputs.concat(posEmbeddings);
        } else {
            return tf.concat([inputs, posEmbeddings], 1);
        }
    }

    static get className() {
        return 'PositionEmbedding';
    }
    
}

tf.serialization.registerClass(PositionEmbedding)
// module.exports = {
//     PositionEmbedding: PositionEmbedding
// }