/************************************************************************/
/*                                                                      */
/*   svm_util.h                                                         */
/*                                                                      */
/*   Definitions and I/O functionality.                                 */
/*                                                                      */
/*   Author: Peter N Robinson                                           */
/*   Date: 12.04.15                                                     */
/*                                                                      */
/*   Copyright (c) 2015  Peter Robinson - All rights reserved           */
/*                                                                      */
/*   This software is available on a BSD2 license.                      */
/*                                                                      */
/************************************************************************/

#ifndef SVM_UTIL
#define SVM_UTIL

# include <stdio.h>
# include <ctype.h>
# include <math.h>
# include <string.h>
# include <stdlib.h>
# include <time.h> 
# include <float.h>

#define VERSION      "v0.1"
#define VERSION_DATE "12-04-2015"


# define LINEAR  0           /** linear kernel type */
# define POLY    1           /** polynomial kernel type */
# define RBF     2           /** rbf kernel type */
# define SIGMOID 3           /** sigmoid kernel type */

# define MAXFEATNUM 99999999 /** maximum feature number (must be in
			  	valid range of long int) */

/** \brief An individual feature of input data.
 *
 * This struct represents an individual feature. Note that
 * we are storing the data as a sparse vector, and data features
 * that are zero are not explicitly represented.
 */
typedef struct feature {
  unsigned long   fnum;	/**< Feature number */
  float           fval; /**< Value of the feature */
} FEATURE;

/** \brief a data point.
 *
 * This struct represents an individual data point (with all
 * of its features) in the original feature space. Note that
 * we represent the features as a sparse version, i.e., feature
 * numbers that are not represented are treated as having the
 * value of zero. 
 */
typedef struct fvector {
  unsigned long id; /**< The position of this feature vector in the training data array. */
  FEATURE  *features; /**< An array of N features has length N+1, and
			 the final slot is NULL. */
  double  twonorm_sq; /**< The squared euclidian length of the
                                  feature vector (Used for the RBF kernel). */
  double  factor;   /**< Factor by which this feature vector
				  is multiplied in the sum. */
  double data_class; /**< +1 or -1 */
} FVECTOR;


/** \brief The data and classifications to be used for training the SVM.
 *
 * This struct contains the feature vectors and classifications of the
 * training data. It will be used to create a Gram matrix using one of 
 * the implemented kernels.
 */
typedef struct training_data {
  int n; /**< Number of data points */
  FVECTOR *data; /**< The feature vectors */
} TRAINING;

/** verbosity level (0-4) */
extern int verbosity;             

extern void input_training_data(const char *path, FVECTOR ***fvec_list, unsigned long *total_features, long int *total_fvecs);

extern void scan_n_lines_and_features(char *file, 
		   unsigned *n_lines, 
		   unsigned *wol, 
		   unsigned *ll);
extern int space_or_null(int c);
extern void *xmalloc(size_t size);
extern FVECTOR *create_feature_vector(FEATURE *features,double label,double factor);

//extern *tranin


#endif
