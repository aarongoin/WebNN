# WebNN

WebNN (WNN) is a JavaScript framework for distribution and training of Neural Networks (NN) through the internet. This is a large part of my research, working alongside Ron Cotton into distributed training of neural networks at WSU Vancouver. WNN consists of a NodeJS HTTP server that facilitates the asynchronous distributed training of user-defined NN. It builds on [TensorFlow.js](https://js.tensorflow.org), which leverages the clients' GPU using WebGL (or CPU if WebGL unavailable) to run and train a NN in any web browser that supports WebGL 1.0 or better--which is 96% of browsers in the US and 93% of browsers globally [[Caniuse](https://caniuse.com/#search=webgl)]. 


## Getting Started

1. Clone this repo onto the machine that will be your server.
2. Navigate into the repo and run `npm install` to install all necessary dependencies.
3. Run `node wnn train mnist` to launch the server with the mnist model included with this repo
4. You'll see some output in your terminal as the server prepares the mnist model for training.
5. Once you see the 'Server ready!' output: open your browser and navigate to the url `localhost:8888/watcher` to open the watcher page.
6. In another page or tab, navigate to the url `localhost:8888/client` to begin training the model. You should soon see accuracy and loss metrics populate the charts of the watcher page showing you the training and validation progress.


## Usage

### Commands
WebNN comes with a simple JS command-line script to make certain workflows a little easier:
    new <name>                  - Create blank model
    reset <name>                - Resets model back to initialized stated
    copy <src_name> <dest_name> - Copies model to a new clean version ready for training
    save <src_name> <dest_name> - Copies model including all logs and weights
    train <name> <port>         - Runs training for model
    retrain <name> <port>       - Shortcut for reset <name> followed by train <name> <port>
    list                        - List all models

### Watching Training
Use the Watcher to see how many clients are training, the current learning rate, and the running average (5 samples) of the model validation accuracy and loss. The first chart shows validation metrics, and the second shows actual training metrics. In both charts, accuracy is in orange with labels on the right y-axis, and loss is in blue with labels on the left y-axis.

### Dumb Clients
WebNN prepares dumb clients for your convenience that simply request data from the server and train. But you can run and train models in real applications as well. Look inside /lib/server/clients/client.js to see how these clients function, and how you can use them in your own applications.

### Models
Models you define and create live in the ./models directory of the repo.
Use the `node wnn new YOUR_MODEL_NAME` command to initialize your model. This command will create a directory under ./models for your model, and will include the following template files:

- config.js - this file is where you configure how your model will train, set hyperparameters, and hook up training and validation data sources.
- model.js - this file is where you define your model architecture in a Keras-like format

Also included will be these empty CSV files:

- log.csv - where all server requests will be logged
- training.csv - where all client training metrics are logged
- validation.csv - where all validation metrics are logged
