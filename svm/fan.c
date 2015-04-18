/**
 * fan.c
 * There are several variations upon the Sequential Minimal Optimization 
 * (SMO) originally proposed by John Platt. In essence, the methods differ
 * in the way they pick pairs of Lagrange multipliers to be optimized.
 * This module implements the method of  working-set selection using second order
 * information as described in Fan R, Chen P, Lin C (2005).
 * Sebastian Bauer, Peter N. Robinson, 2006.
 */
 

#include "svm.h"
#include "fan.h"

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TAU 1e-12
#define EPS 1e-3

#define IS_UPPER_BOUND(i,C) ((i) >= (C))
#define IS_LOWER_BOUND(i)  ((i) <= 0)
#define MIN(x,y)  ((x) < (y) ? (x) : (y) )
#define MAX(x,y)  ((x) > (y) ? (x) : ( y) )

#define VERBOSE 1


/**
 * \brief Select the indices of two Lagrange multipliers to be optimized. 
 *
 * Put the values of the two selected Lagrange multipliers into I and J.
 * @param I pointer to index of first Lagrange multiplier
 * @param J pointer to index of second selected Lagrange multiplier
 * @param svm The SVM model
 * @param G todo ?
 */
static void selectB(int *I, int *J,struct svm *svm, double *G){
  int i,j;
  int t;
  int N;
  double G_max, G_min,obj_min,b,a;
  double k11,k22,k12;
  signed char *y;
  double *alpha;
  
  G_max = -FLT_MAX;
  G_min = FLT_MAX;
  
  N = svm->training_count;
  y = svm->data_class;
  alpha = svm->alpha;
  
  /* ----   Select i  ----  */
  i = -1;
  for (t=0;t<N;++t) {
    /* Is exemplar t in I_up ? */
    /*printf("Example %d: class %d penalty %f\n",t,svm->data_class[t],GET_C(svm,t));*/
    if ( (y[t] == 1 && alpha[t] < GET_C(svm,t) ) ||  
	 (y[t] == -1 && alpha[t] > 0) ) { 
      if (-y[t] * G[t] >= G_max){
	G_max = -y[t] * G[t];
	i = t;
      }
    }
  }
  
  /* ----   Select j  ----  */
  j = -1;	
  obj_min = FLT_MAX;
  for (t=0;t<N;++t) {
    /* is exemplar t in I_low? */
    if ( (y[t] == 1 && alpha[t] > 0) ||
	 (y[t] == -1 && alpha[t] < GET_C(svm,t) ) ) {
      b = G_max + y[t] * G[t];
      if (-y[t] * G[t] <= G_min) {
	G_min = -y[t] * G[t];
      }
      if (b > 0) {
	k11 = svm->kernel(i, i, svm);
	k12 = svm->kernel(i, t, svm);
	k22 = svm->kernel(t, t, svm);
	/* The following is equivalent to
	 * a = Q[i][i] + Q[t][t] - 2*y[i]y[t]*Q[i][t];
	 * recall that Q[i][j] = y[i]y[j]K(i,j) and
	 * y[i]*y[i] = 1 forall i
	 */
	a = k11 + k22 - 2 * k12;
	if (a <= 0){ a = TAU; }
	if ( -(b*b)/a <= obj_min ) {
	  j = t;
	  obj_min = -(b*b)/a;
	}
      }
    }
  }
  if (G_max - G_min < EPS) { /* This terminates training */
    *I = -1; 
    *J = -1; 
  } else {
    *I = i;
    *J = j;
  }		
}

/** \brief Calculate the bias (b) term of the SVM.
 *
 * @param svm The SVM model
 * @param G todo explain
 */
double calculate_bias(struct svm *svm, double *G)
{
  double r1, r2;
  int i,N;
  int nrfree;
  int nneg,npos;
  double ub,lb,sum_free;
  signed char *y;
  double *alpha;
  
  nrfree = 0;
  ub  = FLT_MAX;
  lb  = -FLT_MAX;
  sum_free = 0.0;	
  
  N = svm->training_count;
  y = svm->data_class;
  alpha = svm->alpha;
  
  nneg = 0; npos = 0;
  
  for (i=0;i<N;++i) {
    if (y[i] == 1) {
      double yG = y[i] * G[i];
      npos++;
      if (IS_LOWER_BOUND(alpha[i]) ) {
	ub = MIN(ub,yG);
      } else if (IS_UPPER_BOUND(alpha[i], GET_C(svm,i) ) ) {	
	lb = MAX(lb,yG);
      } else {
	nrfree++;
	sum_free += yG;
      }
    } 
  }
  if (nrfree > 0)
    r1 = sum_free / (double) nrfree;
  else
    r1 = (ub + lb)/2.0;
  
  nrfree = 0;
  ub  = FLT_MAX;
  lb  = -FLT_MAX;
  sum_free = 0.0;	
  
  for (i=0;i<N;++i) {
    if (y[i] == -1) {
      double yG = y[i] * G[i];
      nneg++;
      if (IS_LOWER_BOUND(alpha[i]) ) {
	lb = MAX(lb,yG);
      } else if (IS_UPPER_BOUND(alpha[i], GET_C(svm,i) ) ) {
	ub = MIN(ub,yG);
      } else {
	nrfree++;
	sum_free += yG;
      }	
    } 
  }
  
  if (nrfree > 0)
    r2 = sum_free / (double) nrfree;
  else
    r2 = (ub + lb)/2.0;
  
  if (nneg + npos != N) {
    fprintf(stderr,"Error, in bias: nneg = %d + nnpos =%d != N = %d\n",nneg,npos,N);
  }	
  
  /*printf("Got bias: %f (r1 = %f, r2 = %f \n", (r2-r1)/2.0, r1, r2);*/
  return (r2 + r1)/2.0;  /* -b = 1/2*(r_1 - r_2) */	
}

/** \brief This function corresponds to algorithm 2 in Fan et al. */
void train_model_fan(struct svm *svm)
{
  int N;
  int i,j;
  int k,t;
  double *G;
  int maxiter, iter;
  double a, k11,k12,k22;
  double sum;
  double b;
  signed char *y; /* The class of the exemplar, +1 or -1 */
  double *alpha;
  double old_alpha_i, old_alpha_j, new_alpha_i, new_alpha_j;
  double delta_alpha_i,delta_alpha_j;
  
  N = svm->training_count;
  maxiter = svm->max_iter;
  y = svm->data_class;
  alpha = svm->alpha;
  iter = 0;
  
  if ((G = malloc(sizeof(double)*N)) == NULL) {
    fprintf(stderr,"Could not allocate memory for G array in fan.c\n");
    exit(1);
  }
  
  /* Initialize alpha Lagrange multiplier array to all zero. 
   * Also initialize the Gradient array G to all -1*/
  for (k=0;k<N;k++){
    svm->alpha[k] = 0.0;	
    G[k] = -1;
  }
  
  while (1)  {
    if (iter++ > maxiter) break;
    selectB(&i,&j,svm,G);
    if (j == -1) break;
    k11 = svm->kernel(i, i, svm);
    k12 = svm->kernel(i, j, svm);
    k22 = svm->kernel(j, j, svm);
    /* The following is equivalent to
     * a = Q[i][i] + Q[t][t] - 2*y[i]y[t]*Q[i][t];
     * recall that Q[i][j] = y[i]y[j]K(i,j) and
     * y[i]*y[i] = 1 forall i
     */
    a = k11 + k22 - 2 * k12;
    if (a <= 0) a = TAU;
    b = -y[i]*G[i] + y[j]*G[j]; /* equation 14 of Fan et al */
    
    /* Update alpha */
    old_alpha_i = alpha[i];
    old_alpha_j = alpha[j];
    new_alpha_i = alpha[i] + y[i]*(b/a);
    new_alpha_j = alpha[j] - y[j]*(b/a);
    
    /* project alpha back to feasible region */
    sum = y[i] * old_alpha_i + y[j] * old_alpha_j;
    if (new_alpha_i  > GET_C(svm,i) )
      new_alpha_i = GET_C(svm,i);
    if (new_alpha_i < 0) 
      new_alpha_i = 0;
    new_alpha_j = y[j] * (sum - y[i]*new_alpha_i);
    if (new_alpha_j  > GET_C(svm,j) )
      new_alpha_j = GET_C(svm,j) ;
    if (new_alpha_j < 0) 
      new_alpha_j = 0;
    new_alpha_i = y[i] * (sum - y[j]*new_alpha_j);
    
    alpha[i] = new_alpha_i;
    alpha[j] = new_alpha_j;
    
    /* Update Gradient */
    delta_alpha_i = new_alpha_i - old_alpha_i;
    delta_alpha_j = new_alpha_j - old_alpha_j;
    for (t=0;t<N;++t) {
      double delta_Gt = 
	y[i]*y[t]*svm->kernel(i,t,svm)*delta_alpha_i +
	y[j]*y[t]*svm->kernel(j,t,svm)*delta_alpha_j ;
      G[t] += delta_Gt;	
    }
#if VERBOSE
    if (iter % MIN(100,svm->training_count) == 0){
      svm->b = calculate_bias(svm, G);
      double obj = objective_function(svm);
      
      fprintf(stderr,"Obj = %.3f; iter=%dn",obj, iter);
      calculate_diagnostics(svm);
      fprintf(stderr,"\ttrain_err: %d/%d (TP:%d, TN:%d, FP:%d,FN:%d) test_err: %d/%d (TP:%d, TN:%d, FP:%d,FN:%d)\n",
	      svm->training_err_count,svm->training_count,svm->train_TP,svm->train_TN,svm->train_FP,svm->train_FN,
	      svm->test_err_count,svm->test_count,svm->test_TP,svm->test_TN,svm->test_FP,svm->test_FN);
      fprintf(stderr,"\t");
      output_bound_vs_unbound_supports(svm, stderr);
      fprintf(stderr,"\n");
      
    }
#endif
  }
  svm->b = calculate_bias(svm, G);
  free(G);	
}

#undef IS_UPPER_BOUND
#undef IS_LOWER_BOUND
#undef MIN
#undef MAX

/* eof */
