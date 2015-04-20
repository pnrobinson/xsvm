/**
 * test_xsvm.c
 *
 * Unit testing of some of the functions of xsvm
 * 
 * @author Peter Robinson
 * @version 0.01
 */


#include <glib.h>
#include "svm.h"


typedef struct {
  GRAM_MATRIX *gram;
} gram_fixture;

void gram_setup_A(gram_fixture *gf,gconstpointer test_data) {
  gf->gram =  initialize_gram_matrix(42);
}

void gram_teardown(gram_fixture *gf,gconstpointer test_data) {
  //free(gf->gram);
}

void test_gram_init_A(gram_fixture *gf,gconstpointer ignored) {
  g_assert(42 == gf->gram->n);
}

/** Note that isspace returns a nonzero value for true (not necessarily 1).*/
void test_space_or_null(gram_fixture *gf,gconstpointer ignored){
  //int result = space_or_null((unsigned char)' ');
  int result = space_or_null(0);
  g_assert(0 != result);
  int space = ' ';
  result = space_or_null((int) space);
  g_assert(0 != result);
}


void test_feature_vector(gram_fixture *gf,gconstpointer ignored){
  unsigned long   fnum;	/**< Feature number */
  float           fval; /**< Value of the feature */
  int n=3;
  unsigned int i;
  FEATURE *featuresA,*featuresB;  
  featuresA = (FEATURE *)xmalloc(sizeof(FEATURE)*n);
  featuresB = (FEATURE *)xmalloc(sizeof(FEATURE)*n);
  for (i=0;i<n;++i) {
    featuresA[i].fnum=i;
    featuresA[i].fval=i*2;
    featuresB[i].fnum=i;
    featuresB[i].fval=i;
  }
}


void test_parse_line_A(gram_fixture *gf,gconstpointer ignored){
  char *line="+1	3:1	4:1	6:1	8:1";
  // parse_line(char *line, FEATURE *features, double *label,
  //		      long int *n_features, long int max_features);
  FEATURE *features = (FEATURE *)xmalloc(sizeof(FEATURE)*4);
  double *labels = (double *)xmalloc(sizeof(double));
  int n_features;
  long int max_features=7;
  printf("%s",line);
  printf("pos 0: %c\n",line[0]);
  parse_line(line,features,labels,&n_features,max_features);
  printf("n features=%d\n",n_features);
  g_assert(4==n_features);
}

int main(int argc,char**argv) {
  g_test_init(&argc, &argv, NULL);
  //g_test_add ("/set1/new gram test", gram_fixture, NULL,
  //	      gram_setup_A, test_gram_init_A, gram_teardown);
  g_test_add("/set2/utilities",gram_fixture,NULL,NULL,test_space_or_null,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_A,NULL);
  return g_test_run();
}
