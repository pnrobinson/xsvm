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

/** Verbosity level for output */
int verbosity;


/** \brief Scan file to prepare for data input.
 *
 * Grep through file and count number of lines, maximum number of
 * spaces per line, and longest line. 
 * @param file name (full path) to file with training data
 * @param n_lines Pointer to variable that will receive the total number of lines of training data
 * @param wol ?
 * @param ll pointer to var that will get the longest line length (ll)
 */
void scan_n_lines_and_features(char *file, 
			       unsigned *n_lines, 
			       unsigned *wol, 
			       unsigned *ll) 
{
  FILE *fl;
  int ic; /* int value returned from getc representing one char */
  char c; /* current char value derived from 'ic' */
  long current_length; /* length of the current line */
  long current_wol; /* one more than the number of word on the line */
  
  if ((fl = fopen (file, "r")) == NULL) { 
    perror (file); 
    exit (1); 
  }
  current_length=0;
  current_wol=0;
  (*ll)=0;
  (*n_lines)=1;
  (*wol)=0;
  /* Get char's one by one. If the char is
     a '\n', check if the number of white-spaces or the total
     length of the czurrent line is longer than wol or ll,
     and reset. */
  while((ic=getc(fl)) != EOF) {
    c=(char)ic;
    current_length++;
    if(space_or_null((int)c)) {
      current_wol++;
    }
    putc(ic,stdout);
    if(c == '\n') {
      (*n_lines)++;
      if(current_length>(*ll)) {
	(*ll)=current_length;
      }
      if(current_wol>(*wol)) {
	(*wol)=current_wol;
      }
      //printf("currentlen=%lu, currentWOL=%lu\n",current_length,current_wol);
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


/** \brief Compute the inner product of two sparse vectors.
 * Compute the dot product of two sparse vectors. Note that 
 * the indices of the vectors must be in ascending order. This
 * allows us to increment the features of a or b until we get to features
 * that match.
 */
double sparse_dotproduct(FVECTOR *a, FVECTOR *b) 
{
    register double sum=0;
    register FEATURE *ai,*bj;
    ai=a->features;
    bj=b->features;
    while (ai->fnum && bj->fnum) {
      if(ai->fnum > bj->fnum) {
	bj++;
      }
      else if (ai->fnum < bj->fnum) {
	ai++;
      }
      else {
	sum+=(ai->fval) * (bj->fval);
	ai++;
	bj++;
      }
    }
    return((double)sum);
}


/* eof */
