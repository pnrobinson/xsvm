/**
 * platt.h
 * There are several variations upon the Sequential Minimal Optimization
 * (SMO) originally proposed by John Platt. In essence, the methods differ
 * in the way they pick pairs of Lagrange multipliers to be optimized.
 * This module implements the original algorithm as preoposed in Platt (1998) 
 * and described in the book An Introduction to Support Vector Machines and
 * other kernel-based learning methods (2000) by N. Cristianini and J.
 * Shawe-Taylor.
 * Sebastian Bauer, Peter N. Robinson, 2006.
 */

#ifndef PLATT_H
#define PLATT_H


#include "svm.h"

void train_model_platt(struct svm *svm);


#endif
