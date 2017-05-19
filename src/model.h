#ifndef	_MODEL_H
#define	_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include "dataset.h"
#include "document.h"
#include "map_type.h"
#include "utils.h"
#include "math_func.h"
#include "polya_fit_simple.h"
#include "strtokenizer.h"

using namespace std;
/// <summary>
/// Offers parameters and functions to train a single
/// time slice JST model.
/// </summary>
class model {

public:
	model(void);
	~model(void);

	/// <summary>
	/// word2atr holds the global vocabulary map (string, Word_atr)
	/// </summary>
	mapword2atr word2atr;

	/// <summary>
	/// id2word holds the global vocabulary map (int, string)
	/// Every word (string) holds a unique Word-ID (int)
	/// </summary>
	mapid2word id2word;
	mapword2prior sentiLex; // <word, [senti lab, word prior distribution]>
	
	string data_dir;
	string datasetFile;
	string result_dir;
	string sentiLexFile;
	string wordmapfile;
	string tassign_suffix;
	string pi_suffix;
	string theta_suffix;
	string phi_suffix;
	string others_suffix;
	string twords_suffix;

	int numTopics;
	int numSentiLabs; 
	int niters;
	int liter;
	int twords;
	int savestep;
	int updateParaStep;
	double _alpha;
	double _beta;
	double _gamma;



	/// <summary>
	/// This function loads the parameters specified in .properties file.
	/// Internally it calls <see cref="T:utils"/>-><see cref="M:utils.parse_args_est"/>.
	/// </summary>
	/// <returns></returns>
	int init(int argc, char ** argv);

	/// <summary>
	/// Trains the JST model on specified data set in .properties
	/// </summary>
	/// <returns></returns>
	int execute_model();

	/// <summary>
	/// Initializes the first model for the dJST model.
	/// The model will be trained on data collected for the first time step.
	/// </summary>
	/// <returns></returns>
	int initFirstModel();


	/// <summary>
	/// Initializes a new model for the specified time slot <paramref name="epoch" />.
	/// </summary>
	/// <param name="epoch">The epoch on which the model will be trained.</param>
	/// <returns></returns>
	int initNewModel(int epoch, string model_dir);

	// added
	// Declaration of counts
	vector<vector<double> > new_p; // for posterior
	vector<vector<int> > new_z;
	vector<vector<int> > new_l;
	vector<int> new_nd; // number of words in a document d
	vector<vector<int> > new_ndl; // counter for sentiment-state per document (wieviele counts an Senti-Labels hat also ein gewisses Doc)
	vector<vector<vector<int> > > new_ndlz; // counter for sentiment-topic-state per document
	vector<vector<vector<int> > > new_nlzw; // counter for individual word-to-topic-sentiment assignment (wie oft taucht Wort w mit diesem topic&sentiment auf)
	vector<vector<int> > new_nlz; // counter for sentilabels per topic
	vector<vector<double> > newpi_dl; // size: (numDocs x L)
	vector<vector<vector<double> > > newtheta_dlz; // size: (numDocs x L x T) 
	vector<vector<vector<double> > > newphi_lzw; // size: (L x T x V)
	

private:
	/// <summary>
	/// Refer to <see cref="F:dataset.numDocs"/>.
	/// </summary>
	int numDocs;
	/// <summary>
	/// Refer to <see cref="F:dataset.vocabSize"/>.
	/// </summary>
	int vocabSize;
	/// <summary>
	/// Refer to <see cref="F:dataset.corpusSize"/>.
	/// </summary>
	int corpusSize;
	/// <summary>
	/// Refer to <see cref="F:dataset.corpusSize"/>.
	/// </summary>
	int aveDocLength;
	
	ifstream fin;	
	dataset * pdataset;
	utils * putils;

	// model counts
	vector<int> nd; // number of words in a document d
	vector<vector<int> > ndl; // counter for sentiment-state per document
	vector<vector<vector<int> > > ndlz; // counter for sentiment-topic-state per document
	vector<vector<vector<int> > > nlzw; // counter for individual word-to-topic-sentiment assignment
	vector<vector<int> > nlz; // counter for sentilabels per topic
	
	// topic and label assignments
	vector<vector<double> > p;
	vector<vector<int> > z;
	vector<vector<int> > l;
	
	// model parameters
	vector<vector<double> > pi_dl; // size: (numDocs x L)
	vector<vector<vector<double> > > theta_dlz; // size: (numDocs x L x T) 
	vector<vector<vector<double> > > phi_lzw; // size: (L x T x V)
	
	// hyperparameters 
	vector<vector<double> > alpha_lz; // \alpha_tlz size: (L x T)
	vector<double> alphaSum_l; 
	vector<vector<vector<double> > > beta_lzw; // size: (L x T x V)
	vector<vector<double> > betaSum_lz;
	vector<vector<double> > gamma_dl; // size: (numDocs x L)
	vector<double> gammaSum_d; 
	vector<vector<double> > lambda_lw; // size: (L x V) -- for encoding prior sentiment information 
		
	vector<vector<double> > opt_alpha_lz;  //optimal value, size:(L x T) -- for storing the optimal value of alpha_lz after fix point iteration
	
	/************************* Functions ***************************/
	int set_gamma();

	/// <summary>
	/// Initializes model parameters like <see cref="F:model.numDocs" />.
	/// Also the counts will be resized according to the current data.
	/// </summary>
	/// <returns></returns>
	int init_model_parameters();

	/// <summary>
	/// Initializes the model parameters like <see cref="F:model.numDocs" />.
	/// Also the counts will be resized according to the current data.
	/// In contrast to <see cref="M:model.init_model_parameters" /> this method is designed
	/// to work for continious data.
	/// </summary>
	/// <returns></returns>
	int init_model_parameters1();

	inline int delete_model_parameters() {
		numDocs = 0;
		vocabSize = 0;
		corpusSize = 0;
		aveDocLength = 0;
		
		if (pdataset != NULL) {
			delete pdataset;
			pdataset = NULL;
		}
		
		return 0;
	}

	/// <summary>
	/// The model gets prepared for the training phase.
	/// By this, the counts necessary for the Gibbs Sampler will be initialized.
	/// </summary>
	/// <returns></returns>
	int init_estimate();
	int init_estimate1();

	/// <summary>
	/// In this function the model is trained.
	/// This includes the Gibbs Sampling procedure and the
	/// re-estimation of the parameters (phi, pi,...) based
	/// on the new sampled counts.
	/// </summary>
	/// <returns></returns>
	int estimate();

	/// <summary>
	/// A single model is trained like in <see cref="M:model.estimate" />.
	/// The difference is that we only train on the data restricted at that
	/// time <paramref name="epoch" />
	/// </summary>
	/// <returns></returns>
	int estimate(int epoch);
	int estimate1(int epoch);
	int prior2beta();
	int prior2beta1(); // added

	/// <summary>
	/// The Gibbs Sampling procedure is specified in this function.
	/// New Topic- and Senti-Labels have been sampled after running this method.
	/// </summary>
	/// <returns></returns>
	int sampling(int m, int n, int& sentiLab, int& topic);
	int sampling1(int m, int n, int& sentiLab, int& topic);
	
	// compute parameter functions
	/// <summary>
	/// Estimates the pi parameter based on the new samples.
	/// </summary>
	void compute_pi_dl();

	/// <summary>
	/// Estimates the theta parameter based on the new samples.
	/// </summary>
	void compute_theta_dlz(); 

	/// <summary>
	/// Estimates the phi parameter based on the new samples.
	/// </summary>
	void compute_phi_lzw(); 

	void compute_phi_lzw1(); // added
	
	// update parameter functions
	void init_parameters();
	int update_Parameters();

	// save model parameter funtions 
	int save_model(string model_name);
	int save_model1(string model_name);
	int save_model(string model_name, int epoch); // added
	int save_model1(string model_name, int epoch);
	int save_model_tassign(string filename);
	int save_model_pi_dl(string filename);
	int save_model_theta_dlz(string filename);
	int save_model_phi_lzw(string filename);
	int save_model_phi_lzw1(string filename);
	int save_model_others(string filename);
	int save_model_twords(string filename);
	int save_model_twords1(string filename);
};

#endif
