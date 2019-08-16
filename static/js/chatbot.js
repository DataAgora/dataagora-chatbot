var dataagora = require('dataagora-dml'); 
var WebSocket = require('ws');
var tfjs_1 = require("@tensorflow/tfjs");
var fetch = require('node-fetch');
const fs = require('fs');


const repo_id = "3e55b6e37447aca26c807c2aa5961d89";
const cloud_url = "http://" + repo_id + ".au4c4pd2ch.us-west-1.elasticbeanstalk.com";
const ws_url = "ws://" + repo_id + ".au4c4pd2ch.us-west-1.elasticbeanstalk.com";
const model_url = cloud_url + "/model/model.json";

const hyperparams = {
    "batch_size": 8000,
    "epochs": 5,
    "shuffle": True, 
    "label_index": "label"
};

const percentage_averaged = 0.75;

const max_rounds = 5

const NEW_MESSAGE = {
    "type": "NEW_SESSION",
    "repo_id": repo_id,
    "json_model": json_model,
    "hyperparams": hyperparams,
    "selection_criteria": {
        "type": "ALL_NODES",
    },
    "continuation_criteria": {
        "type": "PERCENTAGE_AVERAGED",
        "value": percentage_averaged
    },
    "termination_criteria": {
        "type": "MAX_ROUND",
        "value": max_rounds
    }
}

function preprocess(text) {
    
}

var cloud_websocket = null;
var model = null;

function handle_input(text) {
    if (model == null)
        load_model_local();
    var result = model.predict(text); //needs to be adjusted, based on what model we using.
    dataagora.store(repo_id, new Tensor(text).as2D(5, 5));
    cloud_websocket.send(NEW_MESSAGE);
}

function bootstrap() {
    dataagora.bootstrap(repo_id);
    load_model_local();
    set_up_cloud_websocket();
}

function set_up_cloud_websocket() {
    cloud_websocket = new WebSocket(ws_url);
    
    cloud_websocket.addEventListener("open", function (event) {
        var registrationMessage = {
            "type": "REGISTER",
            "node_type": "DASHBOARD"
        };
        cloud_websocket.send(JSON.stringify(registrationMessage));
    });

    cloud_websocket.addEventListener("message", function (event) {
        var message = JSON.parse(event.data);
        if ("action" in message && message["action"] == "NEW_MODEL") {
            fetch_model_cloud(cloud_url)
        }
    });

    cloud_websocket.addEventListener("close", function (event) {
        if (event.code == 1006) {
            set_up_cloud_websocket();
        }
    });
}

function fetch_model_cloud(url) {
    model = tfjs_1.loadLayersModel(model_url);
    fetch(model_url)
    .then(function (res) { return res.json(); })
    .then(function (out) {
        console.log('Output: ', out);
        model = _compileModel(model, out["modelTopology"]["training_config"]);
        await model.save('./model/my-model')
    }).catch(err => console.error(err));
}

function load_model_local(url) {
    model = await tfjs_1.loadLayersModel('./model/my-model');
    let rawdata = fs.readFileSync('student.json');  
    let model_json = JSON.parse(rawdata);
    model = _compileModel(model, model_json["modelTopology"]["training_config"]);
}

function _lowerCaseToCamelCase (str) {
    return str.replace(/_([a-z])/g, function (g) { return g[1].toUpperCase(); });
}

function _compileModel(model, optimization_data) {
    var optimizer;
    if (optimization_data['optimizer_config']['class_name'] == 'SGD') {
        // SGD
        optimizer = tfjs_1.train.sgd(optimization_data['optimizer_config']['config']['lr']);
    } else if (optimization_data['optimizer_config']['class_name'] == 'Adam') {

    } else {
        // Not supported!
        throw "Optimizer not supported!";
    }
    model.compile({
        optimizer: optimizer,
        loss: _lowerCaseToCamelCase(optimization_data['loss']),
        metrics: optimization_data['metrics']
    });
    console.log("Model compiled!", model);
    return model;
};