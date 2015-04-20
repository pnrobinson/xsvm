#include <stdlib.h>
#include <stdio.h>

/** \mainpage xsvm
 *
 * This code is an implementation  of the support vector machine in C. It can use two different algorithms
 * for optimization (Platt and Fan). We intend it to be used especially for custom string kernels for
 * immunogenetics. Therefore, in order to streamline the design of the code, we explicitly calculate
 * a Gram matrix from input data (rather than calculating it on the fly or using a cache), since this
 * will simplify the design of the custom kernel code. In order to test the code, we provide linear, RBF,
 * and polynomial kernels.
 * @author Peter Robinson
 * @version 0.01 (April 12, 2015)
 */

#include "svm_util.h"
#include "svm.h"

/** Path to the file with training data */
char training_data_file[200];
/** Path to the SVM model file */
char model_file[200];

void input_arguments(int argc,char *argv[],char *docfile,char *modelfile,
		     int *verbosity, KERNEL_PARAM *kernel_parameters);
void print_help();
void read_training_data(char *trainfile, FVECTOR ***fvecs, 
		    unsigned long *n_features, unsigned long *n_fvecs);
int parse_line(char *line, FEATURE *features, double *label,
	       long int *n_features, long int max_words_doc);
void initialize_svm(SVM *svm, GRAM_MATRIX *gram, FVECTOR **fv_list);

/** Determined wheter the optimization will be performed using the Fan algorithm (default)
 * or the SMO algorithm of Platt).
 */
enum optimization opt_type=FAN;

int main(int argc,char ** argv) {
  FVECTOR **feature_vector_list; /* the training data */
  unsigned long total_features;
  unsigned long total_feature_vectors;
  KERNEL_PARAM kernel_parameters;
  GRAM_MATRIX *gram;
  SVM svm;
 
  printf("xsvm\n");
  input_arguments(argc,argv,training_data_file,model_file,&verbosity, &kernel_parameters);
  read_training_data(training_data_file,&feature_vector_list,&total_features,
		     &total_feature_vectors);
  // For now, let us use a Euclidean kernel
  gram = calculate_gram_matrix(total_feature_vectors,feature_vector_list,&kernel_parameters);
  
  initialize_svm(&svm, gram, feature_vector_list);
  
  svm_train(&svm,opt_type);

  svm_output_message(&svm);

  return 0;
}


double dumbkernelfxn(int i1, int i2, SVM *svm) {
  double** mat = (double**)svm->data;
  return mat[i1][i2];
}

void initialize_svm(SVM *svm, GRAM_MATRIX *gram, FVECTOR **fv_list)
{
  svm->data = gram->matrix;
  unsigned int N = gram->n;
  signed char *labels = xmalloc(N*sizeof(signed char));
  for (unsigned int i=0;i<N;++i) {
    if (fv_list[i]->data_class < 0)
      labels[i] = (signed char)-1;
    else if (fv_list[i]->data_class > 0)
      labels[i] = (signed char)1;
    else {
      printf("Error, did not recognize data class for item %d: %f\n",i,fv_list[i]->data_class);
      exit(1);
    }
  }
  svm->data_class = labels;
  svm->training_count = N;
  svm->test_count = 0;
  svm->end_support_i = N;
  svm->kernel = dumbkernelfxn;
  double C=1.0;
  svm->C=C;
  svm->C_neg = C;
  svm->C_pos = C;
  svm->output_file = "xsvm.out";
  int maxIter=100;
  svm->max_iter = maxIter;
  
  if ( plausibility_check(svm) < 0 ) {
    fprintf(stderr,"Terminating program because of errors in SVM initialization\n");
    exit(1);
  }
}



/** \brief Input the data from the training data file.
 *
 * @param trainfile A file with training data (sparse format)
 * @param fvecs The data will be put here
 * @param n_features The total number of features will be put here
 * @param n_fvecs The total number of training data points will be put here
 */
void read_training_data(char *trainfile, FVECTOR ***fvecs, 
			unsigned long *n_features, unsigned long *n_fvecs){
  char *line;
  FEATURE *features;
  long dnum=0,wpos,dpos=0,dneg=0,dunlab=0;
  unsigned n_featvecs; /* number of feature vectores in the training data */
  unsigned max_features; /* maximum number of features in a feature vector */
  unsigned ll; /* maximum line length */
  double doc_label;
  FILE *FH;

  if(verbosity>=1) {
    printf("Scanning examples..."); fflush(stdout);
  }
  scan_n_lines_and_features(trainfile,&n_featvecs,&max_features,&ll); /* scan size of input file */
  max_features+=2;
  ll+=2;
  n_featvecs+=2;
  if(verbosity>=1) {
    printf("done [lines=%u, max features=%u, max line len=%u]\n",
	   n_featvecs,max_features,ll); 
    fflush(stdout);
  }

  (*fvecs) = (FVECTOR **)xmalloc(sizeof(FVECTOR *)*n_featvecs);    /* feature vectors */
  //(*label) = (double *)xmalloc(sizeof(double)*max_docs); /* target values */
  line = (char *)xmalloc(sizeof(char)*ll);
  
  if ((FH = fopen (trainfile, "r")) == NULL){ 
    perror (trainfile); 
    exit (1); 
  }

  features = (FEATURE *)xmalloc(sizeof(FEATURE)*(max_features+10));
  if(verbosity>=1) {
    printf("Reading training data into memory..."); fflush(stdout);
  }
  dnum=0;
  (*n_features)=0;
  while((!feof(FH)) && fgets(line,(int)ll,FH)) {
    if(line[0] == '#') continue;  /* line contains comments */
  
    if(!parse_line(line,features,&doc_label,&wpos,max_features)){
      printf("\nParsing error in line %ld!\n%s",dnum,line);
      exit(1);
    }
    //printf("docnum=%ld: Class=%f,wpos=%ld,max_features=%ud\n",dnum,doc_label,wpos,max_features); 
    if(doc_label > 0) dpos++;
    if (doc_label < 0) dneg++;
    if (doc_label == 0) dunlab++;

    if((wpos>1) && ((features[wpos-2]).fnum>(*n_features))) 
      (*n_features)=(features[wpos-2]).fnum;
    if((*n_features) > MAXFEATNUM) {
      printf("\nMaximum feature number exceeds limit defined in MAXFEATNUM!\n");
      printf("LINE: %s\n",line);
      exit(1);
    }

    double factor=1.0; /* todo: implement this or remove it */
     (*fvecs)[dnum] = create_feature_vector(features,doc_label,factor);
     //printf("\nNorm=%f\n",((*docs)[dnum]->fvec)->twonorm_sq);  
     dnum++;  
     if(verbosity>=1) {
       if((dnum % 100) == 0) {
	 printf("%ld..",dnum); fflush(stdout);
       }
     }
  } 

  fclose(FH);
  free(line);
  free(features);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%ld examples read)\n", dnum);
  }
  (*n_fvecs)=dnum;
  
}


/**
 *\brief Read command line arguments.
 * 
 * Input the arguments from the command line.
 * TODO: Need to extend for testing and custom kernel as well
 */
void input_arguments(int argc,char *argv[],
		     char *trainingfile,
		     char *modelfile,
		     int *verbosity,
		     KERNEL_PARAM *kernel_parameters)
{
  unsigned int i;
  /* default values for model file and verbosity level */
  strcpy (modelfile, "svm_model");
  (*verbosity)=1;
  kernel_parameters->kernel_type=LINEAR;
  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) {
    case '?': print_help(); exit(0);
    case 'v': i++; (*verbosity)=atol(argv[i]); break;
    case 'o': /* default is Fan, so only change if user enters Platt */
      i++;
      const char *opt=argv[i];
      if (!strcmp(opt,"Platt"))
	opt_type=PLATT;
      break;
    default: printf("did not recognize flag %s\n",argv[i]); 
      print_help(); 
      exit(0);
    }
  }
  /* When we get here, we should be receiving two more file names.
  * One of them is obligatory (the training data), the next (model file name)
  * is optional*/
  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy (trainingfile, argv[i]);
  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }

}


void print_help(void) {
 printf("\nxsvm %s: Support Vector Machine learning and classification     %s\n\n",VERSION,VERSION_DATE);
 printf("Author: Peter N Robinson, peter.robinson@charite.de\n\n");
 printf("License: BSD2\n\n");
 printf("\tusage: xsvm [options] file\n\n");
 printf("Argument: file contains the training data or the test data \n");
 printf("\tif the flag -m (model) is set, the argument will be interpreted as test data\n");
 printf("General options:\n");
 printf("\t-?\t->Show help message\n");
 printf("\t-v [0..3]\t-> verbosity level (default 1)\n");
 printf("Learning options:\n");
 printf("\t-o [Fan|Platt]\t->Optimization  (default: Fan)\n");

  
}
