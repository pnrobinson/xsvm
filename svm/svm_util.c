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


/** \brief Read in one line of the training data
 * Note that we use the format of libSVM.
 * For example, <b>1 1:2 2:1 # your comments</b>
 * Note that for now comments are ignored
 *
 * @param line The input line
 * @param features This is a vector that will be filled with the results of parsing
 * @param label one of +1.0 or -1.0
 * @param n_features This will be filled with the number of features
 * @param max_features The maximum features a line can have
 * @return 0 on failure, 1 if we successfully parse a FEATURE vector
 */
int parse_line(char *line, FEATURE *features, double *label,
	       long int *n_features, long int max_features)
{
  register int pos; /* keep track of position in line */
  register long wpos; /* 'word position, the index of the current feature (word) */
  long wnum;
  double weight;
  int numread;
  char featurepair[1000],junk[1000];
  pos=0;
  while(line[pos] ) {      /* cut off comments */
    if((line[pos] == '#')) {
      line[pos]=0;
      break; 
    }
    if(line[pos] == '\n') { /* strip the CR */
      line[pos]=0;
    }
    pos++;
  }
  wpos=0;
  /* check, that line starts with target value or zero, but not with
   * feature pair. The following reads characters until a whitespace is found, and puts
   * the result in a null-terminated string in 'featurepar'*/
  if(sscanf(line,"%s",featurepair) == EOF) return(0);
  pos=0;
  while((featurepair[pos] != ':') && featurepair[pos]) pos++;
  if(featurepair[pos] == ':') {
    printf("[%s:%d] Line must start with label or 0 (training data file)\n",__FILE__,__LINE__); 
    printf("LINE: %s\n",line);
    exit (1); 
  }
  /* read the target value. The following puts the label into 'label' */
  if(sscanf(line,"%lf",label) == EOF) return(0);
  pos=0;
  while(space_or_null((int)line[pos])) pos++;
  while((!space_or_null((int)line[pos])) && line[pos]) pos++;
  /** The following reads the text up to the next whitespace into 'featurepair' */
  while(((numread=sscanf(line+pos,"%s",featurepair)) != EOF) && 
	(numread > 0) && 
	(wpos<max_features)) {
    /* printf("%s\n",featurepair); */
    /* The following advances pos until after the current 'featurepair' */
    while(space_or_null((int)line[pos])) pos++;
    while((!space_or_null((int)line[pos])) && line[pos]) pos++;
    
    if(sscanf(featurepair,"%ld:%lf%s",&wnum,&weight,junk)==2) {
      /* it is a regular feature */
      if(wnum<=0) { 
	printf("[%s:%d] Feature numbers must be larger or equal to 1!!!\n",__FILE__,__LINE__); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
      if((wpos>0) && ((features[wpos-1]).fnum >= wnum)) { 
	perror ("Features must be in increasing order!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
      (features[wpos]).fnum=wnum;
      (features[wpos]).fval=(float)weight; 
      wpos++;
    }
    else {
      perror ("Cannot parse feature/value pair!!!\n"); 
      printf("'%s' in LINE: %s\n",featurepair,line);
      exit (1); 
    }
  }
  (features[wpos]).fnum=0;
  (*n_features)=wpos;
  return(1);
}


/**
 * Output the contents of a feature vector to stdout for debugging purposes.
 */
extern void print_fvector(FVECTOR *fv){
  printf("feature vector %u\n",fv->id);
  FEATURE *f=fv->features;
  while(f!=NULL && f->fnum != 0) {
    printf("%u:%.1f ",f->fnum,f->fval);
    f++;
  }
  printf("\n");
  printf("class: %.1f\n",fv->data_class);
}




/* eof */
