/************************************************************************/
/*                                                                      */
/*   svm_util.c                                                         */
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


#include "svm_util.h"


int verbosity;

/* prototypes */


void input_training_data(const char *path, FVECTOR ***fvec_list, unsigned long *total_features, long int *total_fvecs) {

}

/** \brief Scan file to prepare for data input.
 *
 * Grep through file and count number of lines, maximum number of
 * spaces per line, and longest line. 
 */
void scan_n_lines_and_features(char *file, 
	    unsigned *n_lines, 
	    unsigned *wol, 
	    unsigned *ll) 
{
  FILE *fl;
  int ic;
  char c;
  long current_length,current_wol;

  if ((fl = fopen (file, "r")) == NULL) { 
    perror (file); 
    exit (1); 
  }
  current_length=0;
  current_wol=0;
  (*ll)=0;
  (*n_lines)=1;
  (*wol)=0;
  while((ic=getc(fl)) != EOF) {
    c=(char)ic;
    current_length++;
    if(space_or_null((int)c)) {
      current_wol++;
    }
    if(c == '\n') {
      (*n_lines)++;
      if(current_length>(*ll)) {
	(*ll)=current_length;
      }
      if(current_wol>(*wol)) {
	(*wol)=current_wol;
      }
      current_length=0;
      current_wol=0;
    }
  }
  fclose(fl);
}

/** \brief malloc wrapper. */
void *xmalloc(size_t size)
{
  void *ptr;
  if(size<=0) size=1; 
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}


/** \brief Return true (1) if character is a whitespace or NULL.
 * @param c Character to be checked
 */
int space_or_null(int c) {
  if (c==0)
    return 1;
  return isspace((unsigned char)c);
}




/** \brief Initialize a feature vector during parsing of input file.
 * 
 * This function is called during parsing of training file lines and is parsed a vector
 * of features, a label, and a factor. Note that the end of the feature vector is
 * signaled by FEATURE.fnum==0, which otherwise should never occur.
 */
FVECTOR *create_feature_vector(FEATURE *features,double label,double factor)
{
  FVECTOR *vec;
  long    fnum,i;

  fnum=0;
  while(features[fnum].fnum) {
    fnum++;
  }
  fnum++;
  vec = (FVECTOR *)xmalloc(sizeof(FVECTOR));
  vec->features = (FEATURE *)xmalloc(sizeof(FEATURE)*(fnum));
  for(i=0;i<fnum;i++) { 
      vec->features[i]=features[i];
  }
  //vec->twonorm_sq=sprod_ss(vec,vec);
  vec->data_class=label;
  vec->factor=factor;
  return(vec);
}
