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


int main(int argc,char**argv) {
  g_test_init(&argc, &argv, NULL);
  g_test_add ("/set1/new gram test", gram_fixture, NULL,
	      gram_setup_A, test_gram_init_A, gram_teardown);
  return g_test_run();
}
