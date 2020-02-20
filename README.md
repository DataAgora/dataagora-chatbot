# discreetai-chatbot
DiscreetAI's decentralized chatbot that trains in real time with federated learning. Chatbot is served at chatbot.dataagora.com [deprecated]

Deploying the chatbot is as straightforward as ```python main.py``` on a remote Elastic Beanstalk (EB) server instance with the appropriate .eb files.

All relevant code is inside ```/static/```. Preprocessing and hyperparameters are in ```encoder.json, hparams.json, vocab.bpe```. Within ```js```, model layers are in ```static/js/model/``` and the DML library code is in ```static/js/dataagora-dml/```
