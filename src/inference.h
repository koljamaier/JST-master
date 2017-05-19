/**********************************************************************
		        Joint Sentiment-Topic (JST) Model
***********************************************************************

(C) Copyright 2013, Chenghua Lin and Yulan He

Written by: Chenghua Lin, University of Aberdeen, chenghua.lin@abdn.ac.uk.
Part of code is from http://gibbslda.sourceforge.net/.

This file is part of JST implementation.

JST is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.

JST is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

***********************************************************************/

#ifndef _INFERENCE_H
#define _INFERENCE_H

#include <sys/stat.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "constants.h"
#include "document.h"
#include "dataset.h"
#include "utils.h"
#include "strtokenizer.h"
#include "model.h"

using namespace std; 

/// <summary>
/// This class implements the dJST model. Several JST models (<see cref="T:model" />)
/// are trained and combined over time.
/// </summary>
class Inference {

public:
	Inference(void);
	~Inference(void);

    int numSentiLabs; 
	int numTopics;
	int numDocs;      // for trained model
	int vocabSize;    // for trained model
	int newNumDocs;   // for test set
	int newVocabSize; // for test set
	int time_slices; // added

	vector<vector<vector<int> > > nlzw; // for trained model
	vector<vector<int> > nlz;  // for trained model
	mapid2word id2word; // glob. Vokabular "2984 access"
	mapword2id word2id; // glob. Vokabular "access 2984"
    mapword2atr word2atr; // lok.(!!!) Vokabular "access {2984, sentiLabel}" Beachte: word2atr ist nicht mit dem globalen Mapping zu verwechseln! Hier gilt es nur für das lokale Vokabular
    map<int, int> id2_id; // lok. Vok. (glob_id, loc_id) Mapping zwischen glob und loc Word-IDs (für neue docs)
	map<int, int> _id2id; // lok. Vok. Dieses Mapping bildet dagegen loc auf glob Word-IDs ab
	mapword2prior sentiLex; // <string, int> => <word, polarity>
	vector<string> newWords;

	string data_dir;
	string datasetFile;
	string result_dir;
	string model_dir;
	string model_name;
	string sentiLexFile;
	string wordmapfile;
	string betaFile;

	string tassign_suffix;
    string pi_suffix;
    string theta_suffix;
    string phi_suffix;
    string others_suffix;
    string twords_suffix;

	dataset * pmodelData;	// pointer to trained model object
    dataset * pnewData; // pointer to new/test dataset object
	utils * putils;

	// added
	model * firstModel;

    int niters;
	int liter;
    int twords;
    int savestep;
	int updateParaStep;

	double _alpha;
	double _beta;
	double _gamma;
	
	vector<vector<double> > new_p; // for posterior
	vector<vector<int> > new_z;
    vector<vector<int> > new_l;
	vector<vector<int> > z;  // for trained model
    vector<vector<int> > l;  // for trained model


	// from NEW/test documents
	vector<int> new_nd; // number of words in a document d
	vector<vector<int> > new_ndl; // counter for sentiment-state per document (wieviele counts an Senti-Labels hat also ein gewisses Doc)
	vector<vector<vector<int> > > new_ndlz; // counter for sentiment-topic-state per document
	vector<vector<vector<int> > > new_nlzw; // counter for individual word-to-topic-sentiment assignment (wie oft taucht Wort w mit diesem topic&sentiment auf)
	vector<vector<int> > new_nlz; // counter for sentilabels per topic

	// hyperparameters 
    vector<vector<double> > alpha_lz; // size: (L x T)
	vector<double> alphaSum_l; 
	vector<vector<vector<double> > > beta_lzw; // size: (L x T x V)
	vector<vector<double> > betaSum_lz;
	vector<double> gamma_l; // size: (L)
	double gammaSum; 
	vector<vector<double> > lambda_lw; // size: (L x V) -- for encoding prior sentiment information 
	
	// model parameters
	vector<vector<double> > newpi_dl; // size: (numDocs x L)
	vector<vector<vector<double> > > newtheta_dlz; // size: (numDocs x L x T) 
	vector<vector<vector<double> > > newphi_lzw; // size: (L x T x V)

	// functions 
	int init(int argc, char ** argv);
    int init_inf();
    int inference(); // inference for new (unseen) data based on previously trained model
    int inf_sampling(int m, int n, int& sentiLab, int& topic);
	int init_parameters();
    
	//added
	/// <summary>
	/// Initializes the first model. For this, data is read at
	/// the first epoch. The corpus is analyzed and model parammeters
	/// get initialized. For further detail how the data is analyzed
	/// refer to <see cref="T:dataset" />
	/// </summary>
	/// <returns></returns>
	int initFirstModel();

	/// <summary>
	/// Initializes a new JST model for <paramref name="epoch" /> based on the
	/// global vocabulary known from previous epochs. The global vocabulary is extended by new words.
	/// The models initiated in this function form the base models for the dJST model.
	/// Note: The model parameters from previous don't influence each other in this function!
	///		  The models are trained separately in this function!
	/// </summary>
	/// <param name="epoch">The epoch.</param>
	/// <returns></returns>
	int initNewModel(int epoch);

	int read_newData(string filename); // filename = data_dir + datasetFile
	int read_model_setting(string filename);
	int load_model(string model_name);
	int prior2beta(); // for incorporating prior information

	// compute model parameters
	void compute_newpi();
	void compute_newtheta();
	int compute_newphi();

	// save new data models
	int save_model(string model_name);
    int save_model_newtassign(string filename);
    int save_model_newpi_dl(string filename);
    int save_model_newtheta_dlz(string filename);
    int save_model_newphi_lzw(string filename);
    int save_model_newothers(string filename);
    int save_model_newtwords(string filename);

	private:
		ifstream fin;
};

#endif
