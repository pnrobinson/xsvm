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
  long int n_features;
  long int max_features=7;
  //printf("%s",line);
  //printf("pos 0: %c\n",line[0]);
  parse_line(line,features,labels,&n_features,max_features);
  //printf("n features=%d\n",n_features);
  g_assert(4==n_features);
}

void test_parse_line_B(gram_fixture *gf,gconstpointer ignored){
  char *line="+1	3:1	4:1 ";
  FEATURE *features = (FEATURE *)xmalloc(sizeof(FEATURE)*4);
  double *labels = (double *)xmalloc(sizeof(double));
  long int n_features;
  long int max_features=7;
  parse_line(line,features,labels,&n_features,max_features);
  g_assert(2==n_features);
}

void test_parse_line_C(gram_fixture *gf,gconstpointer ignored){
  char *line="+1  3:1 4:1 5:1 6:1 7:1 8:1  9:1  10:1  13:1  14:1  15:1   16:1";
  int n=12;
  FEATURE *features = (FEATURE *)xmalloc(sizeof(FEATURE)*n);
  double *labels = (double *)xmalloc(sizeof(double));
  long int n_features;
  long int max_features=n;
  parse_line(line,features,labels,&n_features,max_features);
  g_assert(12==n_features);
}

void test_parse_line_D(gram_fixture *gf,gconstpointer ignored){
  char *line="+1	3:1	4:1 ";
  FEATURE *features = (FEATURE *)xmalloc(sizeof(FEATURE)*4);
  double *labels = (double *)xmalloc(sizeof(double));
  long int n_features;
  long int max_features=7;
  parse_line(line,features,labels,&n_features,max_features);
  g_assert(2==n_features);
  g_assert_cmpfloat(labels[0],==,1.0);
}

void test_parse_line_E(gram_fixture *gf,gconstpointer ignored){
  char *line="-1	3:1	4:1 ";
  FEATURE *features = (FEATURE *)xmalloc(sizeof(FEATURE)*4);
  double *labels = (double *)xmalloc(sizeof(double));
  long int n_features;
  long int max_features=7;
  parse_line(line,features,labels,&n_features,max_features);
  g_assert(2==n_features);
  g_assert_cmpfloat(labels[0],==,-1.0);
}


void test_parse_line_F(gram_fixture *gf,gconstpointer ignored){
  char *line="-1	3:111.88	4:1 ";
  double DELTA=0.0001;
  FEATURE *features = (FEATURE *)xmalloc(sizeof(FEATURE)*4);
  double *labels = (double *)xmalloc(sizeof(double));
  long int n_features;
  long int max_features=7;
  parse_line(line,features,labels,&n_features,max_features);
  g_assert_cmpfloat((features[0].fval-111.88),<,DELTA);
  //g_assert_cmpint(1,==,2);
}

void test_sparse_dotproductA(gram_fixture *gf,gconstpointer ignored){
  char *line1="+1	3:1	4:1 ";
  char *line2="+1	3:1	4:1 ";
  FEATURE feat1;
  FEATURE feat2;
  double *labels;
sparse_dotproduct(FVECTOR *a, FVECTOR *b) 

  }


int main(int argc,char**argv) {
  g_test_init(&argc, &argv, NULL);
  g_test_set_nonfatal_assertions ();
  //g_test_add ("/set1/new gram test", gram_fixture, NULL,
  //	      gram_setup_A, test_gram_init_A, gram_teardown);
  g_test_add("/set2/utilities",gram_fixture,NULL,NULL,test_space_or_null,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_A,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_B,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_C,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_D,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_E,NULL);
  g_test_add("/set2/parseline",gram_fixture,NULL,NULL,test_parse_line_F,NULL);
  return g_test_run();
}
