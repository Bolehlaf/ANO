#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "backprop.h"
#include <fstream>
#include <string>
#include <iostream>

void train(NN* nn)
{
	double F1[] = { 0.114228, 0.152912, 0.152498 , 0.157233 , 0.650592 , 0.683311, 0.635285, 0.624427, 0.299501, 0.218254, 0.222665, 0.219657 };
	double F2[] = { 0.933485, 0.959695, 0.959695, 0.954513, 0.935107, 0.964361, 0.928736, 0.912165, 0.079112, 0.0810539, 0.0750288, 0.0780312 };
	int classes[] = { 0,0,0,0,1,1,1,1,2,2,2,2 };

    int n = 12;
	double ** trainingSet = new double * [n];
	for ( int i = 0; i < n; i++ ) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
					
		trainingSet[i][0] = F1[i];
		trainingSet[i][1] = F2[i];

		if (classes[i] == 0) {
			trainingSet[i][2] = 0;
			trainingSet[i][3] = 0;
			trainingSet[i][4] = 1;
		}
		else if (classes[i] == 1) {
			trainingSet[i][2] = 0;
			trainingSet[i][3] = 1;
			trainingSet[i][4] = 0;
		}
		else {
			trainingSet[i][2] = 1;
			trainingSet[i][3] = 0;
			trainingSet[i][4] = 0;
		}

	}
    
    double error = 1.0;
    int i = 0;
	while(error > 0.001)
    {
		setInput( nn, trainingSet[i%n] );
		feedforward( nn );
		error = backpropagation( nn, &trainingSet[i%n][nn->n[0]] );
        i++;
		printf( "\rerr=%0.3f", error );
	}
	printf( " (%d iterations)\n", i );

	for ( int i = 0; i < n; i++ ) {
		delete [] trainingSet[i];
	}
	delete [] trainingSet;
}

void test(NN* nn, int num_samples = 5)
{
	double F1[] = { 0.147784 ,0.218978, 0.185920 ,0.132795 ,0.493442 };
	double F2[] = { 0.922365 ,0.164472, 0.164655 ,0.901263 ,0.835079 };
	int classes[] = { 0,2,2,0,1 };
	double ** in = new double *[num_samples];
	for (int i = 0; i < num_samples; i++) {
		in[i] = new double[2];

		in[i][0] = F1[i];
		in[i][1] = F2[i];
	}
    int num_err = 0;
    for(int n = 0; n < num_samples; n++)
    {       
        printf("predicted: %d\n", 2-classes[n]);
        setInput( nn, in[n], true );

        feedforward( nn );
        int output = getOutput( nn, true );
        if(output != 2-classes[n]) num_err++;
        printf( "\n" );
    }
    double err = (double)num_err / num_samples;
    printf("test error: %.2f\n", err);
}

int main(int argc, char** argv)
{
    NN * nn = createNN(2, 4, 3);

    train(nn);
    
    getchar();
    
    test(nn);

	getchar();

	releaseNN( nn );
    
	return 0;
}
