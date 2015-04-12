/**
 * svm.h
 *
 * Created on: 20.12.2008
 * 
 * @author Peter Robinson
 * @version 0.01
 */

#ifndef SVM_H_
#define SVM_H_

#include "stdio.h"
#include "svm_util.h"

typedef struct gram_mat {
  unsigned int n; /**< Number of data points */
  double **matrix;
} GRAM_MATRIX;

struct svm
{
  /* In general, data represents an nxn matrix of scores of kernel evalutions for the
   * (i,j) pair of training exemplars, i.e., the Gram matrix. In the
   * case of the spectrum and mismatch kernels, the score is a dot product of count vectors
   * and is represented as integers. For many of kernels, the score is a float/double.
   * If desired, this can be a pointer to some data structure with which the kernel
   * evaluation can be performed each time dynamically, i.e., without storing the
   * entire Gram matrix in memory.
   */

  void *data;
  /* +1 or -1 class for each data exemplar */
  signed char *data_class;
  int training_count;
  int test_count;
  int end_support_i;
  int positive_test_exemplar_count;
  int positive_training_exemplar_count;

  /* C is the penalty term for misclassification
     of a training exemplar. The following allows
     differing penalties to be defined for the
     positive and negative classes. If only one
     penalty is indicated, they are set to be the same. */
  double C_pos;
  double C_neg;
  /* At the moment, the two-penalty training is implemented
   * only for the Fan algorithm. The Platt 
   * algorithms is less easily adaptable to this and use only
   * a single penalty.
   */
   double C;


  int max_iter;

  void *userdata;

  /* The kernel function */
  double (*kernel)(int i1, int i2, struct svm *svm);

  /* Learned Model Parameters */
  double *alpha; /* Lagrange multipliers */
  double b;  /* the bias */
  double *error_cache; /* E_i or F_i depending on optimization method */

  char *output_file;

  /* Out parameters */
  int training_err_count;
  int test_err_count;
  /* training false positive */
  int train_FP;
  /* training false negative */
  int train_FN;
  /* training true positive */
  int train_TP;
  /* training true negative */
  int train_TN;
  int test_FP;
  int test_FN;
  int test_TP;
  int test_TN;
};

#define GET_C(svm,idx) ( (svm->data_class[idx] > 0 ? svm->C_pos : svm->C_neg) )

double learned_func_nonlinear(struct svm *svm, int k, double b);
double objective_function(struct svm *svm);
void calculate_bound_vs_unbound_supports(struct svm *svm, int *lb, int *ub, int *unb_sv);
int plausibility_check(struct svm *svm);
void free_svm(struct svm *svm);

enum optimization { PLATT, FAN};

void smo_train(struct svm *svm, enum optimization opt);

void smo_print_results_to_log(struct svm *svm, FILE *fp);

/* The following is used by platt.h and for debugging/verbose */
void output_bound_vs_unbound_supports(struct svm *svm, FILE* fp);
void calculate_diagnostics(struct svm *svm);


GRAM_MATRIX * calculate_gram_matrix(unsigned int n,
				    FVECTOR **feature_vector_list,
				    KERNEL_PARAM *kernel_parameters);
double kernel_function(KERNEL_PARAM *k_params, FVECTOR *a, FVECTOR *b);

#endif /* SVM_H_ */
