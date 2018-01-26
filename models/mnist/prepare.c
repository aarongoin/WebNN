#include <stdio.h>
#include <assert.h>
#include <string.h>

#define MINIBATCH 32
#define VALIDATIONBATCH 64
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define X_WIDTH 784
#define Y_WIDTH 10
#define X_HEADERS 16
#define Y_HEADERS 8


void split(int populationSize,int batchSize, FILE * origX, FILE * origY, char dest[]) {

	FILE* miniBatch;
	char path[15];
	strcpy(path, dest);

	// parse out as many batches as possible from data (discard data that won't fit into minibatch)
	int i = (populationSize / batchSize) >> 0;
	float mb = (float) batchSize;
	float n;
	float buffer[X_WIDTH];

	int b = -1;
	while (++b < i) {
		// create new file to hold this training batch
		sprintf(&path[7], "%d", b);
		miniBatch = fopen(path, "wb");
		fwrite(&mb, sizeof mb, 1, miniBatch); // write minibatch size

		printf("Working on batch: %d\n", b);
		// read in X data point
		int m = batchSize;
		int l;
		int w = 28;
		while (m--) {
			// read in minibatch X data
			printf("Reading in sample %d and downsampling...\n", (batchSize - m));
			l = X_WIDTH;
			while (l--) {
				// read in data point and normalize
				n = ((float) fgetc(origX)) / 255.0;
				//buffer[l] = n;
				// write data to batch file
				fwrite(&n, sizeof n, 1, miniBatch);

				// print input ascii art to console
				if (n > 0) printf("#");
				else printf(" ");
				if (--w == 0) {
					printf("\n");
					w = 28;
				}
			}
			// for (int y = 0; y < 27; y += 1) {
			// 	for (int x = 0; x < 27; x += 1) {
			// 		n = (buffer[x + (y * 28)] + buffer[x + (y * 28) + 1] + buffer[x + (y * 28) + 28] + buffer[x + (y * 28) + 29]) / 2;
			// 		if (n > 1.0) n = 1.0;
			// 		else if (n < 0.0) n = 0.0;
			// 		// write data to batch file
			// 		fwrite(&n, sizeof n, 1, miniBatch);

			// 		// print input ascii art to console
			// 		if (n > 0.75) printf("#");
			// 		else printf(" ");
			// 		if (--w == 0) {
			// 			printf("\n");
			// 			w = 14;
			// 		}
			// 	}
			// }

		}

		

		// read in minibatch Y data
		printf("One-hot-encoding output...\n");
		l = batchSize;
		char c;
		while (l--) {
			// read in data point
			c = fgetc(origY);
			// write Y one-hot-encoded to minibatch file
			for (char label = 0; label < Y_WIDTH; label++) {
				n = label == c ? 1 : 0;
				fwrite(&n, sizeof n, 1, miniBatch);
			}
		}
		printf("Closing minibatch... \n");
		fclose(miniBatch);
	}

	fclose(origX);
	fclose(origY);
}


int main(int argc, char **argv) {
	// parse training data into minibatches (length=64)
	FILE* trainX = fopen("./orig/train-images.idx3-ubyte", "rb");
	FILE* trainY = fopen("./orig/train-labels.idx1-ubyte", "rb");
	FILE* testX = fopen("./orig/t10k-images.idx3-ubyte", "rb");
	FILE* testY = fopen("./orig/t10k-labels.idx1-ubyte", "rb");

	// assert(trainX != NULL);
	// assert(trainY != NULL);
	// assert(testX != NULL);
	// assert(testY != NULL);

	int length = X_WIDTH * NUM_TRAIN - X_HEADERS;

	// skip over header data
	printf("Skipping over headers...\n");
	int i;
	i = X_HEADERS;
	while (i--) {
		fgetc(trainX);
		fgetc(testX);
	}
	i = Y_HEADERS;
	while (i--) {
		fgetc(trainY);
		fgetc(testY);
	}

	printf("Splitting training data into minibatches...\n");
	split(NUM_TRAIN, MINIBATCH, trainX, trainY, "./data/\0\0\0\0\0\0\0\0");
	printf("Splitting validation data into minibatches...\n");
	split(NUM_TEST, VALIDATIONBATCH, testX, testY, "./test/\0\0\0\0\0\0\0\0");
}
