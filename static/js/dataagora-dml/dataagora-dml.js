"use strict";

import {DataManager} from "./data_manager.js";

export class DataAgoraDML {

    bootstrap(repo_id) {
        return DataManager.bootstrap(repo_id)
    }

    store(repo_id, data) {
        return DataManager.store(repo_id, data);
    }

    isBootstrapped() {
        return DataManager.bootstrapped;
    }
}