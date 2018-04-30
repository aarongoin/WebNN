// Adopted from Google's Tensorflow.js MNIST example code

const tf = require('@tensorflow/tfjs');
const FS = require('fs');
const get_pixels = require('get-pixels');
const https = require('https');

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    './models/mnist/data/mnist_images.png';
const MNIST_LABELS_PATH =
    './models/mnist/data/mnist_labels_uint8';

class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
        this.loadIndex = 0;
        this.load();
    }

    async load() {
        get_pixels(MNIST_IMAGES_SPRITE_PATH, undefined, (error, pixels) => {
            if (error) {
                console.log("Failed to get image!");
                return ;
            }
            this.datasetImages = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZE);

            const floatView = pixels.data;

            for (let j = 0; j < floatView.length / 4; j++) {
                this.datasetImages[j] = floatView[j * 4] / 255;
            }
            console.log("Loaded image data.");
            this.onload();

        });

        FS.readFile(MNIST_LABELS_PATH, (error, data) => {
            if (error) {
                console.log('Failed to get labels!')
                return ;
            } else {
                console.log("Loaded labels.");
            }
            
            this.datasetLabels = data;
            this.onload();
            
        });

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
    }

    onload() {
        this.loadIndex++;
        if (this.loadIndex === 2) {
            console.log('Preparing test data...');
            // Slice the the images and labels into train and test sets.
            this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
            this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
            this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
            this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
            console.log('Test data prepared.');
        }
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch( batchSize, [this.trainImages, this.trainLabels], () => {
            this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
            return this.trainIndices[this.shuffledTrainIndex];
        });
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        },);
    }

    nextBatch(batchSize, data, index) {
        const batchNum = new Float32Array([batchSize]);
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label = data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        return Buffer.concat([
            new Uint8Array(batchNum.buffer),
            new Uint8Array(batchImagesArray.buffer),
            batchLabelsArray
        ]);
    }
}

module.exports = MnistData;
