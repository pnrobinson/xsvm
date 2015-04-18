/**
 * platt.c
 * There are several variations upon the Sequential Minimal Optimization 
 * (SMO) originally proposed by John Platt. In essence, the methods differ
 * in the way they pick pairs of Lagrange multipliers to be optimized.
 * This module implements the original algorithm as in Platt (1998) and 
 * described in the book An Introduction to Support Vector Machines and
 * other kernel-based learning methods (2000) by N. Cristianini and J.
 * Shawe-Taylor.
 * Sebastian Bauer, Peter N. Robinson, 2006.
 */

#include "platt.h"


#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int smo_examine_example(struct svm *svm,  int i1);
static int takeStep(struct svm *svm, int i1, int i2);

/*************************************************************/
#define SIGNT(x)  ( (x)>(0)  ?   (1):(-1) )
#define MAX(x,y)  ( (x)>(y)  ?   (x):(y) )
#define MIN(x,y)  ( (x)<(y)  ?   (x):(y) )

#define VERBOSE 1  /* For debugging, to follow progress of training */

/*************************************************************/
/** C is the penalty for misclassifying an example during training. */
static double C = 1;
static double tolerance = 0.001;
/** bias of the SVM */
static double b = 0.0;
static double eps=0.001;
static double delta_b;
static int end_support_i = -1;


/**
 * \brief Optimize the SVM using the Sequential Minimal Optimization scheme.
 * This is the only public function in this module. THere is a loop over all 
 * examples (0..[training_count-1]) to start the loop. Then, only those
 * examples are examine that are unbound, i.e., 0 < alpha < C. If we do not
 * change any example in a loop like this, then we again go through all examples.
 * Only if then, no further examples are changed do we terminate. We can also terminate
 * the loop if we reach max_iter.
 * @param svm The svn data structure containg the Gram matrix and various parameters.
 */
void train_model_platt(struct svm *svm)
{
  int k, num_changed, examine_all;
  int iter,max_iter;
  
  /* initialize some file-scope variables */	
  C = svm->C;
  end_support_i = svm->training_count;
  
  examine_all = 1;
  iter = 0;
  max_iter = svm->max_iter;
  if (max_iter < 1) max_iter = 0x7fffffff;
  do
  {
    num_changed = 0;
    
    if (examine_all){
      for (k = 0; k < svm->training_count; k++)
	num_changed += smo_examine_example(svm,k);
      examine_all = 0;
    } else {
      for (k = 0; k < svm->training_count; k++){
	if (svm->alpha[k] != 0 && svm->alpha[k] != C)
	  num_changed += smo_examine_example(svm, k);
      }
      if (num_changed == 0) examine_all = 1;
    }
    
#if VERBOSE
    if (iter % MIN(100,svm->training_count) == 0){
      fprintf(stderr,"iter=%d; number changed=%d\n",iter,num_changed);
      svm->b = b;
      calculate_diagnostics(svm);
      fprintf(stderr,"train_err: %d/%d (TP:%d, TN:%d, FP:%d,FN:%d) test_err: %d/%d (TP:%d, TN:%d, FP:%d,FN:%d)\n",
	      svm->training_err_count,svm->training_count,svm->train_TP,svm->train_TN,svm->train_FP,svm->train_FN,
	      svm->test_err_count,svm->test_count,svm->test_TP,svm->test_TN,svm->test_FP,svm->test_FN);
      output_bound_vs_unbound_supports(svm, stderr);
      fprintf(stderr,"\n");
    }
#endif
    
    iter++;
  } while ((num_changed > 0 || examine_all) && (iter<max_iter));
  fprintf(stderr,"\n ***\nDONEDONE SMO-Platt Training iter=%d; number changed=%d\n",iter,num_changed);
  svm->b = b;
}

/**
 * \brief Examine a single example.
 * TODO better documentaiton
 * @param svm the SVM model with the Gram matrix and other parameters
 * @param i1 The index of the example to be examined (index in the Gram matrix)
 **/
static int smo_examine_example(struct svm *svm, int i1)
{
  double y1, alph1, E1, r1;
  double *alph = svm->alpha;
  double *error_cache = svm->error_cache;
  
  y1 = svm->data_class[i1];
  alph1 = alph[i1];
  
  if (alph1 > 0 && alph1 < C)
    E1 = error_cache[i1];/* unbound SV */
  else
    E1 = learned_func_nonlinear(svm,i1,b) - y1;
  
  r1 = y1 * E1;
  if ((r1 < -tolerance && alph1 < C) || (r1 > tolerance && alph1 > 0))
  {
    /* Try i2 by three ways; if successful, then immediately return 1; */
    
    /* 1) Try the pair with maximum |E1 - E2| */
    {
      int k, i2;
      double tmax;
      
      for (i2 = (-1), tmax = 0, k = 0; k < end_support_i; k++)
      {
	if (alph[k] > 0 && alph[k] < C)
	{
	  double E2, temp;
	  
	  E2 = error_cache[k];
	  temp = fabs(E1 - E2);
	  if (temp > tmax)
	  {
	    tmax = temp;
	    i2 = k;
	  }
	}
      }
      
      if (i2 >= 0)
      {
	if (takeStep(svm,i1, i2))
	  return 1;
      }
    }
    
    /* 2) try any other unbound example */
    {
      int k, k0;
      int i2;
      
      for (k0 = (int)(drand48() * end_support_i), k = k0;
	   k < end_support_i + k0; k++)
      {
	i2 = k % end_support_i;
	if (alph[i2] > 0 && alph[i2] < C)
	{
	  if (takeStep(svm,i1, i2))
	    return 1;
	}
      }
    }
    
    
    /* 3) Try any other example */
    
    {
      int k0, k, i2;
      
      for (k0 = (int)(drand48() * end_support_i), k = k0;
	   k < end_support_i + k0; k++)
      {
	i2 = k % end_support_i;
	if (takeStep(svm,i1, i2))
	  return 1;
      }
    }
  }
  return 0;
}


/**
 * \brief TODO.
 */
static int takeStep(struct svm *svm, int i1, int i2)
{
  int y1, y2, s;
  double alph1, alph2;		/* old_values of alpha_1, alpha_2 */
  double a1, a2;			    /* new values of alpha_1, alpha_2 */
  double E1, E2, L, H, k11, k22, k12, eta, Lobj, Hobj;
  
  double *alph = svm->alpha;
  double *error_cache = svm->error_cache;
  
  if (i1 == i2)
    return 0;
  
  alph1 = alph[i1];
  y1 = svm->data_class[i1];
  if (alph1 > 0 && alph1 < C)
    E1 = error_cache[i1];
  else
    E1 = learned_func_nonlinear(svm, i1,b) - y1;
  
  alph2 = alph[i2];
  y2 = svm->data_class[i2];
  if (alph2 > 0 && alph2 < C)
    E2 = error_cache[i2];
  else
    E2 = learned_func_nonlinear(svm, i2,b) - y2;
  
  s = y1 * y2;
  
  if (y1 == y2)
    {
      double gamma = alph1 + alph2;
      if (gamma > C)
	{
	  L = gamma - C;
	  H = C;
	}
      else
	{
	  L = 0;
	  H = gamma;
	}
    }
  else
    {
      double gamma = alph1 - alph2;
      if (gamma > 0)
	{
	  L = 0;
	  H = C - gamma;
	}
      else
	{
	  L = -gamma;
	  H = C;
	}
    }
  
  if (L == H)
    return 0;
  
  k11 = svm->kernel(i1, i1, svm);
  k12 = svm->kernel(i1, i2, svm);
  k22 = svm->kernel(i2, i2, svm);
  eta = 2 * k12 - k11 - k22;
  
  if (eta < 0)
    {
      a2 = alph2 + y2 * (E2 - E1) / eta;
      if (a2 < L)
	a2 = L;
      else if (a2 > H)
	a2 = H;
    }
  else
    {
      double c1 = eta / 2;
      double c2 = y2 * (E1 - E2) - eta * alph2;
      Lobj = c1 * L * L + c2 * L;
      Hobj = c1 * H * H + c2 * H;
      
      if (Lobj > Hobj + eps)
	a2 = L;
      else if (Lobj < Hobj - eps)
	a2 = H;
      else
	a2 = alph2;
    }
  
  if (fabs(a2 - alph2) < eps * (a2 + alph2 + eps))
    return 0;
  
  a1 = alph1 - s * (a2 - alph2);
  if (a1 < 0)
    {
      a2 += s * a1;
      a1 = 0;
    }
  else if (a1 > C)
    {
      double t = a1 - C;
      a2 += s * t;
      a1 = C;
    }
  
  {
    double b1, b2, bnew;
    
    if (a1 > 0 && a1 < C)
      bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
    else
      {
	if (a2 > 0 && a2 < C)
	  bnew =
	    b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 -
						     alph2) * k22;
	else
	  {
	    b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 -
							  alph2) * k12;
	    b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 -
							  alph2) * k22;
	    bnew = (b1 + b2) / 2;
	  }
      }
    
    delta_b = bnew - b;
    b = bnew;
  }
  
  {
		/* Update error cache using new Lagrange multipliers */
    int i;
    double t1 = y1 * (a1 - alph1);
    double t2 = y2 * (a2 - alph2);
    
    for (i = 0; i < end_support_i; i++)
      {
				if (0 < alph[i] && alph[i] < C)
					error_cache[i] +=   t1 * svm->kernel(i1, i, svm) + 
						t2 * svm->kernel(i2, i, svm) - delta_b;
      }
    error_cache[i1] = 0.;
    error_cache[i2] = 0.;
  }
  
  alph[i1] = a1;				/* Store a1 in the alpha array. */
  alph[i2] = a2;				/* Store a2 in the alpha array. */
  
  return 1;
}


