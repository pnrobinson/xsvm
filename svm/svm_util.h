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
#define VERSION_DATA "12-04-2015"


# define LINEAR  0           /* linear kernel type */
# define POLY    1           /* polynomial kernel type */
# define RBF     2           /* rbf kernel type */
# define SIGMOID 3           /* sigmoid kernel type */

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
  FEATURE  *features; /**< An array of N features has length N+1, and
			 the final slot is NULL. */
  double  twonorm_sq; /**< The squared euclidian length of the
                                  feature vector (Used for the RBF kernel). */
  double  factor;   /**< Factor by which this feature vector
				  is multiplied in the sum. */
} SVECTOR;


//extern *tranin


#endif
