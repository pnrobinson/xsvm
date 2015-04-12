/************************************************************************/
/*                                                                      */
/*   svm.c                                                              */
/*                                                                      */
/*   Functions for kernels and general manipulation of SVMs             */
/*                                                                      */
/*   Author: Peter N Robinson                                           */
/*   Date: 12.04.15                                                     */
/*                                                                      */
/*   Copyright (c) 2015  Peter Robinson - All rights reserved           */
/*                                                                      */
/*   This software is available on a BSD2 license.                      */
/*                                                                      */
/************************************************************************/

#include "svm.h"





GRAM_MATRIX * initialize_gram_matrix(unsigned int n){
GRAM_MATRIX *gm = (GRAM_MATRIX*)xmalloc(sizeof(GRAM_MATRIX));
gm->n = n;
gm->matrix = (double**) xmalloc(n*sizeof(double*));
  for (unsigned i=0;i<n;++i) {
    gm->matrix[i] = (double*)xmalloc(n*sizeof(double));
  }
  for (unsigned i=0;i<n;++i)
    for (unsigned j=0;j<n;++j)
      gm->matrix[i][j]=0.0d;

return gm;
}

GRAM_MATRIX * calculate_gram_matrix(unsigned int n,
				      FVECTOR **feature_vector_list,
				      KERNEL_PARAM *kernel_parameters) 
{
GRAM_MATRIX *gm =initialize_gram_matrix(n);
if(verbosity>=1) {
switch (kernel_parameters->kernel_type) {
 case LINEAR:
printf("Calculating gram matrix [size=%u, kernel type=LINEAR]...",n); break;
 default:
printf("Calculating gram matrix [size=%u, kernel type=??]...",n); break;
}
fflush(stdout);
  }
for (unsigned int i=0;i<n;++i) {
FVECTOR *a=feature_vector_list[i];
for (unsigned int j=0;j<i;++j) {
FVECTOR *b=feature_vector_list[j];
double d = kernel_function(kernel_parameters,a,b);
gm->matrix[i][j]=d;
gm->matrix[j][i]=d;
}
double d = kernel_function(kernel_parameters,a,a);
gm->matrix[i][i]=d;
}
if(verbosity>=1) {
printf("done\n"); fflush(stdout);
  }
return gm;
}


/** \brief Calculate the kernel function between two feature vectors.
 * This function is used to calculate the kernel function between
 * two vectors. It implements linear, polynomial, RBF, and sigmoid
 * kernels.*/
double kernel_function(KERNEL_PARAM *k_params, FVECTOR *a, FVECTOR *b)  
{
  switch(k_params->kernel_type) {
    case 0: /* linear */ 
      return(sparse_dotproduct(a,b)); 
    case 1: /* polynomial */
      return(pow(k_params->coef_lin*sparse_dotproduct(a,b)+k_params->coef_const,
		 (double)k_params->poly_degree)); 
    case 2: /* radial basis function */
      return(exp(-k_params->rbf_gamma*(a->twonorm_sq-2*sparse_dotproduct(a,b)+b->twonorm_sq)));
    case 3: /* sigmoid neural net */
            return(tanh(k_params->coef_lin*sparse_dotproduct(a,b)+k_params->coef_const)); 
    default: printf("Error: Unknown kernel function\n"); exit(1);
  }
}

