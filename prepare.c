#include <stdio.h>
#include <assert.h>
#include <string.h>

#define MINIBATCH 64
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define X_WIDTH 784
#define Y_WIDTH 10
#define X_HEADERS 16
#define Y_HEADERS 8


int main(int argc, char **argv) {
	// parse training data into minibatches (length=64)
	FILE* trainX = fopen("./models/mnist/orig/train-images.idx3-ubyte", "rb");
	FILE* trainY = fopen("./models/mnist/orig/train-labels.idx1-ubyte", "rb");
	//FILE* testX = fopen("t10k-images.idx3-ubyte", "rb");
	//FILE* testY = fopen("t10k-labels.idx1-ubyte", "rb");

	assert(trainX != NULL);
	assert(trainY != NULL);
	//assert(testX != NULL);
	//assert(testY != NULL);

	int length = X_WIDTH * NUM_TRAIN - X_HEADERS;

	// skip over header data
	printf("Skipping over headers...\n");
	int i;
	i = X_HEADERS;
	while (i--) {
		fgetc(trainX);
		//fgetc(testX);
	}
	i = Y_HEADERS;
	while (i--) {
		fgetc(trainY);
		//fgetc(testY);
	}

	FILE* trainBatch;
	//FILE* testBatch;

	// parse out as many batches as possible from data (discard data that won't fit into minibatch)
	i = NUM_TRAIN / MINIBATCH;
	char path[] = "./models/mnist/data/\0\0\0\0\0";
	float mb = MINIBATCH;
	float n;
	int b = -1;
	printf("Splitting into minibatches...\n");
	while (++b < i) {
		// create new file to hold this training batch
		sprintf(&path[20], "%d", b);
		trainBatch = fopen(path, "wb");
		fwrite(&mb, sizeof mb, 1, trainBatch); // write minibatch size

		printf("Working...\n");
		// read in minibatch X data
		int l = MINIBATCH * X_WIDTH;
		int w = 28;
		while (l--) {
			// read in data point and normalize
			n = ((float) fgetc(trainX)) / 255;
			// write data to batch file
			fwrite(&n, sizeof n, 1, trainBatch);

			// print input ascii art to console
			if (n > 0) printf("#");
			else printf(" ");
			if (--w == 0) {
				printf("\n");
				w = 28;
			}
		}

		// read in minibatch Y data
		l = MINIBATCH;
		char c;
		while (l--) {
			// read in data point
			c = fgetc(trainY);
			// write Y one-hot-encoded to minibatch file
			for (char label = 0; label < Y_WIDTH; label++) {
				n = label == c ? 1 : 0;
				fwrite(&n, sizeof n, 1, trainBatch);
			}
		}

		fclose(trainBatch);
	}

	fclose(trainX);
	fclose(trainY);

}
