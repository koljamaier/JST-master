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


#ifndef	_DATASET_H
#define	_DATASET_H

#include "constants.h"
#include "document.h"
#include "map_type.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
using namespace std; 


/// <summary>
/// This class manages several <see cref="T:document" /> instances.
/// </summary>
class dataset {

public:
	/// <summary>
	/// Mapping that saves words an their corresponding attributes (string, Word_atr)
	/// </summary>
	mapword2atr word2atr;

	/// <summary>
	/// Mapping between Word-ID and the word itself e.g. "2984 access"
	/// </summary>
	mapid2word id2word;

	mapword2id word2id; // glob. Vokabular "access 2984"
	mapword2prior sentiLex; // <word, polarity>
	// added

	/// <summary>
	/// Mapping between local and global word id's.
	/// This is needed to include new unseen words.
	/// </summary>
	map<int, int> id2_id; // lok. Vok. (glob_id, loc_id) Mapping zwischen glob und loc Word-IDs (für neue docs)
	map<int, int> _id2id; // lok. Vok. Dieses Mapping bildet dagegen loc auf glob Word-IDs ab
	vector<string> newWords;
	
	document ** pdocs; // store training data vocab ID (in pdocs sind alle alten Trainings-Dokumente als ID-Wort-Sentiment-Paare gespeichert)
	document ** _pdocs; // only use for inference, i.e., for storing the new/test vocab ID (Hier werden also die Word-IDs nur bezüglich des neuen Docs gespeichert; unabhängig davon, ob ein Wort schon in den Trainingsdaten gesehen wurde (es bekommt also dennoch eine "neue" Word-ID))
    ifstream fin;

	string data_dir; // path to data
	string result_dir; // result model path
	string wordmapfile;

	// added for new data
	string model_dir; // path to old model
	string datasetFile; // new data file
	//string model_name;
	//string data_dir;


	/// <summary>
	/// The number of docs found for the epoch
	/// </summary>
	int numDocs;
	int aveDocLength; // average document length
	/// <summary>
	/// The count of unique words over all documents for the epoch
	/// </summary>
	int vocabSize;
	int newVocabSize; // added; Zählt die Anzahl an unterschiedlichen Worten/Vokabeln über das neue Dokument
	/// <summary>
	/// The count of all words in the corpus (documents). 
	/// Also duplicates will be counted.
	/// </summary>
	int corpusSize;
	
	vector<string> docs; // for buffering dataset
		
	// functions 
	dataset();
	dataset(string result_dir);
	dataset(string result_dir, string model_dir); // added 
	dataset::dataset(string result_dir, mapword2atr word2atr); // added
	~dataset(void);
	
	int read_dataStream(ifstream& fin);
	int read_dataStream1(ifstream& fin);
	/// <summary>
	/// This function analyzes the new corpus of documents.
	/// It will process each word and include it into the mappings.
	/// </summary>
	/// <returns></returns>
	int read_newData(vector<string>& docs);
	int read_senti_lexicon(string sentiLexiconFileDir);
	int analyzeCorpus(vector<string>& docs);
	int analyzeNewCorpus(vector<string>& docs);

	static int write_wordmap(string wordmapfile, mapword2atr& pword2atr);
	static int write_wordmap1(string wordmapfile, mapword2id &pword2id);
	static int read_wordmap(string wordmapfile, mapid2word& pid2word);
	static int read_wordmap(string wordmapfile, mapword2id& pword2id); 

	int init_parameter();
	void deallocate();  
	void add_doc(document * doc, int idx);
	void _add_doc(document * doc, int idx);

};

#endif
