var dataagora = require('dataagora-dml'); 
var WebSocket = require('ws');
var tfjs_1 = require("@tensorflow/tfjs");
var fetch = require('node-fetch');


const repo_id = "3e55b6e37447aca26c807c2aa5961d89";
const cloud_url = "http://" + repo_id + ".au4c4pd2ch.us-west-1.elasticbeanstalk.com";
const ws_url = "ws://" + repo_id + ".au4c4pd2ch.us-west-1.elasticbeanstalk.com";
const model_url = cloud_url + "/model/model.json";

var cloud_websocket;
var model;
var registration_message = {
    "node_type": "DASHBOARD",
    "type": "REGISTRATION"
}

function handle_input(text) {
    
}

function bootstrap() {
    dataagora.bootstrap(repo_id);

    set_up_cloud_websocket()
}

function set_up_cloud_websocket() {
    cloud_websocket = new WebSocket(ws_url);
    
    cloud_websocket.addEventListener("open", function (event) {
        var registrationMessage = {
            "type": "REGISTER",
            "node_type": "LIBRARY"
        };
        cloud_websocket.send(JSON.stringify(registrationMessage));
    });

    cloud_websocket.addEventListener("message", function (event) {
        var message = JSON.parse(event.data);
        if ("action" in message && message["action"] == "NEW_MODEL") {
            load_model_cloud(cloud_url)
        }
    });

    cloud_websocket.addEventListener("close", function (event) {
        if (event.code == 1006) {
            set_up_cloud_websocket();
        }
    });
}

function load_model_cloud(url) {
    model = tfjs_1.loadLayersModel(model_url);
    fetch(model_url)
    .then(function (res) { return res.json(); })
    .then(function (out) {
        console.log('Output: ', out);
        model = _compileModel(model, out["modelTopology"]["training_config"]);
    }).catch(err => console.error(err));
}

function load_model_local(url) {

}

function _lowerCaseToCamelCase (str) {
    return str.replace(/_([a-z])/g, function (g) { return g[1].toUpperCase(); });
}

function _compileModel(model, optimization_data) {
    var optimizer;
    if (optimization_data['optimizer_config']['class_name'] == 'SGD') {
        // SGD
        optimizer = tfjs_1.train.sgd(optimization_data['optimizer_config']['config']['lr']);
    }
    else {
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