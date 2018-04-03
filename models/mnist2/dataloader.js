/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

//import * as tf from '@tensorflow/tfjs';
const tf = require('@tensorflow/tfjs');
const get_pixels = require('get-pixels');
const https = require('https');

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
        this.loadIndex = 0;
        this.load();
    }

    async load() {
        // Make a request for the MNIST sprited image.
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
            console.log("Image data recieved.");
            this.onload();

        });

        // Make HTTP request to get Labels for images
        https.get(MNIST_LABELS_PATH, (response) => {
            const { statusCode } = response;

            if (statusCode !== 200) {
                console.log('Failed to get labels!')
                return ;
            } else {
                console.log("Getting labels...");
            }

            var data = [];

            response.on('data', (chunk) => {
                data.push(chunk);
            }).on('end', () => {
                this.datasetLabels = Buffer.concat(data);
                console.log("Labels recieved.");
                this.onload();
            });
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
