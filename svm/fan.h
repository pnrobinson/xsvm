/**
 * fan.h
 * There are several variations upon the Sequential Minimal Optimization 
 * (SMO) originally proposed by John Platt. In essence, the methods differ
 * in the way they pick pairs of Lagrange multipliers to be optimized.
 * This module implements the method of  working-set selection using second order
 * information as described in Fan R, Chen P, Lin C (2005).
 * Sebastian Bauer, Peter N. Robinson, 2006.
 */


#ifndef FAN_H_
#define FAN_H_

#include "svm.h"

void train_model_fan(struct svm *svm);

#endif /*FAN_H_*/
