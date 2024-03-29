#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "backprop.h"

#define LAMBDA 1.0
#define ETA 0.1

#define SQR( x ) ( ( x ) * ( x ) )

void randomize( double * p, int n ) 
{
	for ( int i = 0; i < n; i++ ) {
		p[i] = ( double )rand() / ( RAND_MAX );
	}
}

NN * createNN( int n, int h, int o ) 
{
	srand(time(NULL));
	NN * nn = new NN;
	
    nn->n = new int[3];
	nn->n[0] = n;
	nn->n[1] = h;
	nn->n[2] = o;
	nn->l = 3;

	nn->w = new double ** [nn->l - 1];
    

	for ( int k = 0; k < nn->l - 1; k++ ) 
    {
		nn->w[k] = new double * [nn->n[k + 1]];
		for ( int j = 0; j < nn->n[k + 1]; j++ ) 
        {
			nn->w[k][j] = new double[nn->n[k]];			
			randomize( nn->w[k][j], nn->n[k]);
			// BIAS
			//nn->w[k][j] = new double[nn->n[k] + 1];			
			//randomize( nn->w[k][j], nn->n[k] + 1 );
		}		
	}

	nn->y = new double * [nn->l];
	for ( int k = 0; k < nn->l; k++ ) {
		nn->y[k] = new double[nn->n[k]];
		memset( nn->y[k], 0, sizeof( double ) * nn->n[k] );
	}

	nn->in = nn->y[0];
	nn->out = nn->y[nn->l - 1];

	nn->d = new double * [nn->l];
	for ( int k = 0; k < nn->l; k++ ) {
		nn->d[k] = new double[nn->n[k]];
		memset( nn->d[k], 0, sizeof( double ) * nn->n[k] );
	}

	return nn;
}

void releaseNN( NN *& nn ) 
{
	for ( int k = 0; k < nn->l - 1; k++ ) {
		for ( int j = 0; j < nn->n[k + 1]; j++ ) {
			delete [] nn->w[k][j];
		}
		delete [] nn->w[k];
	}
	delete [] nn->w;
		
	for ( int k = 0; k < nn->l; k++ ) {
		delete [] nn->y[k];
	}
	delete [] nn->y;
	
	for ( int k = 0; k < nn->l; k++ ) {
		delete [] nn->d[k];
		
	}
	delete [] nn->d;

	delete [] nn->n;

	delete nn;
	nn = NULL;
}

double sigmoid(double s)
{
	double y = 1 / (1 + exp(-s));
	return y;
}

double sigmoid_deriv(double y)
{
	double d = y * (1 - y);
	return d;
}

void feedforward( NN * nn ) 
{ 
	double s = 0;

	for (int k = 1; k < nn->l; k++) {
		for (int i = 0; i < nn->n[k]; i++) {
			s = 0;
			for (int j = 0; j < nn->n[k - 1]; j++) {
				s += nn->w[k-1][i][j] * nn->y[k - 1][j];
			}
			nn->y[k][i] = sigmoid(s);
		}
	}
}

double backpropagation( NN * nn, double * t ) 
{
	for (int i = 0; i < nn->n[nn->l - 1]; i++) {
		nn->d[nn->l-1][i] = (t[i] - nn->y[nn->l-1][i])*sigmoid_deriv(nn->y[nn->l-1][i]) * LAMBDA;
	}

	for (int k = nn->l - 2; k > 0; k--) {
		for (int i = 0; i < nn->n[k]; i++) {
			for (int j = 0; j < nn->n[k + 1]; j++) {
				nn->d[k][i] += nn->d[k + 1][j] * nn->w[k][j][i];
			}
			nn->d[k][i] *= sigmoid_deriv(nn->y[k][i]) * LAMBDA;
		}
	}

	for (int k = 0; k < nn->l - 1; k++) {
		for (int i = 0; i < nn->n[k + 1]; i++) {
			for (int j = 0; j < nn->n[k]; j++) {
				nn->w[k][i][j] += nn->d[k + 1][i] * nn->y[k][j] * ETA;
			}
		}
	}

	double error = 0.0;
	for (int i = 0; i < nn->n[nn->l-1]; i++) {
		error += pow(t[i] - nn->y[2][i], 2);
	}
	error /= 2.0;

	return error;
}

void setInput( NN * nn, double * in, bool verbose ) 
{
	memcpy( nn->in, in, sizeof( double ) * nn->n[0] );

	if ( verbose ) {
		printf( "input=(" );
		for ( int i = 0; i < nn->n[0]; i++ ) {
			printf( "%0.3f", nn->in[i] );
			if ( i < nn->n[0] - 1 ) {
				printf( ", " );
			}
		}
		printf( ")\n" );
	}
}

int getOutput( NN * nn, bool verbose ) 
{	
    double max = 0.0;
    int max_i = 0;
    if(verbose) printf( " output=" );
	for ( int i = 0; i < nn->n[nn->l - 1]; i++ ) 
    {
		if(verbose) printf( "%0.3f ", nn->out[i] );
        if(nn->out[i] > max) {
            max = nn->out[i];
            max_i = i;
        }
	}
	if(verbose) printf( " -> %d\n" , max_i);
    if(nn->out[0] > nn->out[1] && nn->out[0] - nn->out[1] < 0.1) return 2;
    return max_i;
}
