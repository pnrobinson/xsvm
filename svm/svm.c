
#include "svm.h"
#include "svm_util.h"




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
