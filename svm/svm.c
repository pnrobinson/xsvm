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
#include "svm_util.h"
#include "platt.h"
//#include "fan.h"



GRAM_MATRIX * initialize_gram_matrix(unsigned int n){
  GRAM_MATRIX *gm = (GRAM_MATRIX*)xmalloc(sizeof(GRAM_MATRIX));
  gm->n = n;
  gm->matrix = (double**) xmalloc(n*sizeof(double*));
  for (unsigned i=0;i<n;++i) {
    gm->matrix[i] = (double*)xmalloc(n*sizeof(double));
  }
  for (unsigned i=0;i<n;++i)
    for (unsigned j=0;j<n;++j)
      gm->matrix[i][j]=0.0;
  return gm;
}

GRAM_MATRIX * calculate_gram_matrix(unsigned int n,
				    FVECTOR **feature_vector_list,
				    KERNEL_PARAM *kernel_parameters) {
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


/**
 * This function calculates the kernel evaluation for
 * example k.In this code, we assume that the training
 and test data are entered into one large Gram matrix.
 Therefore, we can evaluate the kernel for example k
 against all training exemplars (or the support vectors
 resulting from training) for test exemplars or training
 exemplars using this function.
 */
double learned_func_nonlinear(struct svm *svm, int k, double b)
{
  double s = 0;
  double * alph;

  int N;
  int i;

  alph = svm->alpha;
  N = svm->training_count;

  for (i = 0; i < N; i++)
    {
      if (alph[i] > 0)
	{
	  s += alph[i] * svm->data_class[i] * svm->kernel(i, k, svm);
	}
    }
  s -= b;
  return s;
}


/** \brief This function calculated the objective of the dual. */
double objective_function(struct svm *svm)
{
  int i,j;
  double obj = 0.0;
  int end_support_i;
  unsigned char y1,y2;
  double a1,a2;
  double *alph;

  end_support_i = svm->training_count;
  alph = svm->alpha;

  for (i = 0; i < end_support_i; ++i) {
    for (j=0;j<end_support_i; ++j) {
      y1 = svm->data_class[i];
      y2 = svm->data_class[j];
      a1 = alph[i];
      a2 = alph[j];
      obj += y1 * y2 * a1 * a2 * svm->kernel(i, j, svm);
    }
  }
  obj = (-1.0 * obj )/ 2.0;
  for (i = 0; i < end_support_i; ++i) {
    obj += alph[i];
  }


  return obj;
}


/** \brief This function counts the number of lower bound (alpha = 0), upper
 * bound (alpha = c) and unbound (0<alpha<C) support vectors.      
 */
void calculate_bound_vs_unbound_supports(struct svm *svm, int *lb, int *ub, int *unb_sv)
{
  int lower_bound = 0; /* alpha = 0, non-support vector */
  int upper_bound = 0; /* alpha = C, misclassified */
  int unbound_sv = 0; /* 0 < alpha < C */
  
  int i;
  double C;
  
  C = svm->C;
  
  for (i = 0; i < svm->training_count; i++)
  {
    if (svm->alpha[i] > 0){
      if (svm->alpha[i] < C)
	unbound_sv++;
      else
	upper_bound++;
    } else {
      lower_bound++;
    }
  }
  *lb = lower_bound;
  *ub = upper_bound;
  *unb_sv = unbound_sv;
  
  /*
    fprintf(stderr,"bound_support=%d, non_bound_support=%d, non_sv=%d",
    upper_bound,unbound_sv,lower_bound);
  */
}

/* *********************************************************************
 * This function can be used to check that the user-supplied arguments *
 * are OK. Returns -1 if there is a problem.                           *
 * *********************************************************************/

int plausibility_check(struct svm *svm)
{
  int plausible = 1;
  int N;
  int i;
  int pos;
  
  if (svm->data == NULL) {
    fprintf(stderr,"Error: SVM data not initialized\n");
    plausible = -1;
  }
  if (svm->training_count < 2) {
    fprintf(stderr,"Error: Not enough training exemplars (%d found).\n",
    svm->training_count);
    plausible = -1;
  }
  if (svm->test_count < 0) {
    fprintf(stderr,"Error: Bad initiailization of test_count (%d).\n",
    svm->test_count);
    plausible = -1;
  }
  N = svm->training_count + svm->test_count;
  if (svm->end_support_i != N)
  {
    fprintf(stderr,"Error: end_support_i not initiailized correctly: n_support_i: %d.\n",
    svm->end_support_i);
    plausible = -1;
  }
  for (i= 0; i<N; i++) {
    if (svm->data_class[i] != 1 && svm->data_class[i] != -1) {
      fprintf(stderr,"Error: Bad format for data class for item %d (of %d total items): class \"%d\"\n",
      i,N,svm->data_class[i]);
      plausible = -1;
    }
  }
  
  pos = 0;
  for (i = 0; i<svm->training_count;++i) {
    if (svm->data_class[i] == 1) pos++;
  }
  if (pos < 1) {
    fprintf(stderr,"Error: No positive training examples found\n");
    plausible = -1;
  }
  if (pos == svm->training_count) {
    fprintf(stderr,"Error: No negative training examples found\n");
    plausible = -1;
  }
  
  if (svm->C < 0.0) {
    fprintf(stderr,"Error: penalty parameter C must be positive (%f)\n",
    svm->C);
    plausible = -1;
  }
  
  if (svm->kernel == NULL) {
    fprintf(stderr,"Error: The kernel callback function was not initialized\n");
    plausible = -1;
  }
  return plausible;
}



/*************************************************************/
#define VERBOSE 1
#define SIGNT(x)  ( (x)>(0)  ?   (1):(-1) )

/*************************************************************/




/** \brief Count the number of errors on the training data.               
 * This function makes ues of the function <b>learned_func_nonlinear</b>
 * to predict the class of an example using the SVM, and compares
 * the sign of the answer (positive/negative) to the label of the example.
 * It returns the total number of errors.
 * @param svm The trained SVM
 */
static double training_errors(struct svm *svm)
{
  int n_error = 0;
  int i;
  int N; /* training count */
  double *alpha;
  alpha = svm->alpha;
  N = svm->training_count;
  
  for (i=0; i<svm->training_count; i++){
    if (SIGNT(learned_func_nonlinear(svm,i,svm->b) ) != SIGNT(svm->data_class[i]) )
      n_error++;
  }
  return n_error;
}

/**
 * \brief Similar to the function training_errors, but for test data.
 */
static double test_errors(struct svm *svm)
{
  int n_error, max, i;
  int N; /* training count */
  double *alpha;
  
  n_error = 0;
  alpha = svm->alpha;
  max = svm->training_count + svm->test_count;
  N = svm->training_count;
  
  for (i = svm->training_count; i < max; i++)
  {
    if (SIGNT(learned_func_nonlinear(svm,i,svm->b) ) != SIGNT(svm->data_class[i]) )
    n_error++;
  }
  return n_error;
}

/** \brief Monitor progress of training classification accuracy for debugging purposes etc. 
* The number of true/false positive and negative predictions are 
* calculated for train and test data.
*/
void calculate_diagnostics(struct svm *svm)
{
  int i,max,end_support_i;
  int ntrain_err,ntrain_FP,ntrain_TP, ntrain_FN, ntrain_TN;
  int ntest_err,ntest_FP,ntest_TP, ntest_FN, ntest_TN;
  double *alpha;
  double b;
  
  alpha = svm->alpha;
  b = svm->b;
  end_support_i = svm->training_count;
  
  ntrain_err = ntrain_FP = ntrain_TP =ntrain_FN = ntrain_TN = 0;
  ntest_err = ntest_FP = ntest_TP = ntest_FN = ntest_TN = 0;
  
  /* CALCULATE TRAINING ERROR */
  for (i=0; i<svm->training_count; i++)
  {
    if (SIGNT(learned_func_nonlinear(svm,i,b)) != SIGNT(svm->data_class[i])) {
      ntrain_err++;
      if (SIGNT(svm->data_class[i]) == 1)
      ntrain_FN++;
      else
      ntrain_FP++;
    } else { /* i.e., correct prediction */
      if (SIGNT(svm->data_class[i]) == 1)
      ntrain_TP++;
      else
      ntrain_TN++;
    }
  }
  /* CALCULATE TEST ERROR */
  max = svm->training_count + svm->test_count;
  
  for (i = svm->training_count; i < max; i++)
  {
    if ((learned_func_nonlinear(svm,i,b) > 0) != (svm->data_class[i] > 0)) {
      ntest_err++;
      if (SIGNT(svm->data_class[i]) == 1)
      ntest_FN++;
      else
      ntest_FP++;
    } else { /* i.e., correct prediction */
      if (SIGNT(svm->data_class[i]) == 1)
      ntest_TP++;
      else
      ntest_TN++;
    }
  }
  svm->train_FP = ntrain_FP;
  svm->train_FN = ntrain_FN;
  svm->train_TP = ntrain_TP;
  svm->train_TN = ntrain_TN;
  svm->test_FN = ntest_FN;
  svm->test_FP = ntest_FP;
  svm->test_TP = ntest_TP;
  svm->test_TN = ntest_TN;
  svm->training_err_count = ntrain_err;
  svm->test_err_count = ntest_err;
}

/* ******************************************************************
 * This function can be used to monitor progress of training for    *
 * debugging purposes etc.                                          *
 * Non-support vectors have alpha = 0. Support vectors have alpha>0 *
 * If alpha = C then there is a misclassification of exemplar i.    *
 * ******************************************************************/
void output_bound_vs_unbound_supports(struct svm *svm, FILE* fp)
{
  int non_bound_support = 0;
  int bound_support = 0;
  int non_support_vector = 0;
  /* call function from svm.c, maybe refactor this all... */
  calculate_bound_vs_unbound_supports(svm, &non_support_vector,
  &bound_support,&non_bound_support);
  fprintf(fp,"bound_support=%d, non_bound_support=%d, non_sv=%d",bound_support,non_bound_support,non_support_vector);
}

void smo_print_results_to_log(struct svm *svm, FILE *fp)
{
  calculate_diagnostics(svm);
  fprintf(fp,"train_err: %d/%d (TP:%d, TN:%d, FP:%d,FN:%d) test_err: %d/%d (TP:%d, TN:%d, FP:%d,FN:%d)",
  svm->training_err_count,svm->training_count,svm->train_TP,svm->train_TN,svm->train_FP,svm->train_FN,
  svm->test_err_count,svm->test_count,svm->test_TP,svm->test_TN,svm->test_FP,svm->test_FN);
  fprintf(fp,"[");
    output_bound_vs_unbound_supports(svm, fp);
    fprintf(fp,"]\n");
}

void debug_svm_struct(struct svm *svm)
{
  int i,k, npostrain,npostest,ntotal;
  int nnegtrain = 0;
  int nnegtest = 0;
  npostrain = npostest = ntotal = 0;
  for (k=0;k<svm->training_count;++k) {
    printf("TRAIN: %d: class = %d\n",k,svm->data_class[k]);
  }
  for ( ; k < svm->training_count + svm->test_count;  ++k) {
    printf("TEST: %d: class = %d\n",k,svm->data_class[k]);
  }
  for (i = 0; i < svm->training_count;  ++i) {
    ntotal++;
    if (svm->data_class[i] > 0) npostrain++; else nnegtrain++;
  }
  for ( ; i < svm->training_count + svm->test_count;  ++i) {
    ntotal++;
    if (svm->data_class[i] > 0) npostest++; else nnegtest++;
  }
  printf("Total: %d, pos train %d, neg train %d pos test %d neg test %d \n",ntotal,npostrain,nnegtrain,
  nnegtest,npostest);
  printf("Ntrain %d, ntest %d\n",svm->training_count,svm->test_count);
  
}


void free_svm(struct svm *svm){
  free(svm->alpha);
  free(svm->error_cache);
}




/** \brief The main entry point for training the SVM.
*/
void svm_train(struct svm *svm, enum optimization opt)
{
  int k;
  int verbose=2;
  if (verbose>=1) {
    printf("Training SVM...[kernel:%s]\n",opt==PLATT?"platt":"feng");
  }
  
  /* Allocate memory for alphas and error cache */
  if (!(svm->alpha = malloc(sizeof(svm->alpha[0])*svm->training_count))) {
    fprintf(stderr,"Could not allocate memory for svm->alpha (%s, %d)\n",
	    __FILE__,__LINE__);
    exit(1);
  }
  for (k=0;k<svm->training_count;k++)
    svm->alpha[k] = 0.0;
  
  if (!(svm->error_cache = malloc(sizeof(svm->error_cache[0])*svm->training_count))){
    fprintf(stderr,"Could not allocate memory for svm->error_cache (%s, %d)\n",
	    __FILE__,__LINE__);
    exit(1);
  }
  
  /*	debug_svm_struct(svm); */
  
  
  switch(opt) {
  case PLATT:
    train_model_platt(svm);
    break;
  case FAN:
    //train_model_fan(svm);
    printf("FAN needs to be imported"); exit(1);
    break;
  default:
    fprintf(stderr,"Optimization method %d not recognized.\n",opt);
    fprintf(stderr,"terminating program...\n");
    exit(1);
  }
  
  calculate_diagnostics(svm);
  
  svm->training_err_count = training_errors(svm);
  if (svm->test_count)
    svm->test_err_count = test_errors(svm);
  
  
  if (svm->output_file)
  {
    int i;
    int N; /* training count */
    int NN; /* Total count of training and test exemplars */
    FILE *out = fopen(svm->output_file,"w");
    if (!out)
    {
      fprintf(stderr,"Could not open svm outputfile for writing\n");
      exit(-1);
    };
    N = svm->training_count;
    NN = svm->training_count + svm->test_count;
    for (i=0; i<NN; i++)
    {
      double prediction = learned_func_nonlinear(svm,i, svm->b);
      fprintf(out,"%e\t%e\n",prediction,(double)svm->data_class[i]);
    }
    
    fclose(out);
  }
  return;
}

