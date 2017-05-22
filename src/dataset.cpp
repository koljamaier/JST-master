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

#include "dataset.h"
#include "document.h"
#include "model.h"
#include "map_type.h"
#include "strtokenizer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace std; 


dataset::dataset() {
	pdocs = NULL;
	_pdocs = NULL;
	word2atr.clear();
	result_dir = ".";
	wordmapfile = "wordmap.txt";

	numDocs = 0;
	aveDocLength = 0;
	vocabSize = 0;
	corpusSize = 0;
}

dataset::dataset(string result_dir) {
	pdocs = NULL;
	_pdocs = NULL;
	word2atr.clear();
	this->result_dir = result_dir;
	wordmapfile = "wordmap.txt";

	numDocs = 0; 
	aveDocLength = 0;
	vocabSize = 0; 
	corpusSize = 0;
}

dataset::dataset(string result_dir, mapword2atr word2atr) {
	pdocs = NULL;
	_pdocs = NULL;
	this->word2atr = word2atr;
	this->result_dir = result_dir;
	wordmapfile = "wordmap.txt";

	numDocs = 0;
	aveDocLength = 0;
	vocabSize = 0;
	corpusSize = 0;
}

dataset::dataset(string result_dir, string model_dir) {
	pdocs = NULL;
	_pdocs = NULL;
	word2atr.clear();
	this->result_dir = result_dir;
	this->model_dir = model_dir;
	wordmapfile = "wordmap.txt";

	numDocs = 0;
	aveDocLength = 0;
	vocabSize = 0;
	corpusSize = 0;
}

dataset::~dataset(void) {
	deallocate();
}


int dataset::read_dataStream(ifstream& fin) {
	string line;
	char buff[BUFF_SIZE_LONG];
	docs.clear();
	numDocs = 0;
	
	while (fin.getline(buff, BUFF_SIZE_LONG)) {
		line = buff;
		if(!line.empty()) {
			// docs ist ein vector und kann damit dynamisch erweitert werden.
			// "getline" liest jedoch nur bis zu einem Zeilenumbruch (\n). Dies entspricht bei unserem Datenformat also genau einem Dokument
			// Es werden hier also alle Trainingsdokumente geladen
			docs.push_back(line);
			numDocs++;
		}
	}
	
	if (numDocs > 0) {
		this->analyzeCorpus(docs);
	}
	
	return 0;
}

int dataset::read_dataStream1(ifstream& fin) {
	string line;
	char buff[BUFF_SIZE_LONG];
	docs.clear();
	numDocs = 0;

	while (fin.getline(buff, BUFF_SIZE_LONG)) {
		line = buff;
		if (!line.empty()) {
			// docs ist ein vector und kann damit dynamisch erweitert werden.
			// "getline" liest jedoch nur bis zu einem Zeilenumbruch (\n). Dies entspricht bei unserem Datenformat also genau einem Dokument
			// Es werden hier also alle Trainingsdokumente geladen
			docs.push_back(line);
			numDocs++;
		}
	}
	
	if (numDocs > 0) {
		this->read_newData(docs);
	}

	return 0;
}

// Diese Methode liest alle Dokumente von der .txt ein und legt für jedes Wort eine ID + prior-sentilabel an und schreibt dies in die Datenstruktur document (pdocs)
// Diese Fkt. wird nur einmal am Anfang aufgerufen
int dataset::analyzeCorpus(vector<string>& docs) {

	mapword2atr::iterator it;
	mapword2id::iterator vocabIt;   
	mapword2prior::iterator sentiIt;
	map<int,int>::iterator idIt;
		
	string line;
	numDocs = docs.size();
	// Zählt die Anzahl an unterschiedlichen Worten/Vokabeln über alle Dokumente
	vocabSize = 0;
	// Gibt die Anzahl an allen Wörter in allen Dokumenten an (auch Duplikate)
	corpusSize = 0;
	aveDocLength = 0; 

  // allocate memory for corpus/dataset pdocs
	if (pdocs) {
		deallocate();
		pdocs = new document*[numDocs];
    } 
	else {
		pdocs = new document*[numDocs];
	}
	
	for (int i = 0; i < (int)docs.size(); ++i) {
		line = docs.at(i); // aktuelles Dokument
		//strtokenizer ist eine eigene Klasse des Projekts. Darüber können wir immer wieder auf der aktuellen line (einzelnes Dokument) arbeiten
		strtokenizer strtok(line, " \t\r\n");  // \t\r\n are the separators
		int docLength = strtok.count_tokens();
	
		if (docLength <= 0) {
			printf("Invalid (empty) document!\n");
			deallocate();
			numDocs = vocabSize = 0;
			return 1;
		}
	
		corpusSize += docLength - 1; // the first word is document name/id (z.B. "d0"; deshalb subtrahieren wir 1)
		
		// allocate memory for the new document_i 
		document * pdoc = new document(docLength-1);
		pdoc->docID = strtok.token(0).c_str(); // z.B. "d0"

		// generate ID for the tokens in the corpus, and assign each word token with the corresponding vocabulary ID.
		for (int k = 0; k < docLength-1; k++) {
			int priorSenti = -1;
			// Der Map-Iterator springt an die Stelle, wo dieser Token (Wort) vorkommt
			it = word2atr.find(strtok.token(k+1).c_str());
		
			if (it == word2atr.end()) { //  i.e., new word; denn .find liefert nur .end() zurück, wenn der Token nicht gefunden wurde
				pdoc->words[k] = word2atr.size(); //(weil es ein neues Wort ist wissen wir, dass es ganz ans Ende muss)
				sentiIt = sentiLex.find(strtok.token(k+1).c_str()); // check whether the word token can be found in the sentiment lexicon
				// incorporate sentiment lexicon
				if (sentiIt != sentiLex.end()) {
					// Wenn das Wort also im Sentilexicon (mpqa o.ä.) vorkommt, dann setze für das Wort den entspr. Senti-Prior-Label aus diesem Lexicon
				    priorSenti = sentiIt->second.id;
				}
					
				// insert sentiment info into word2atr
				Word_atr temp = {word2atr.size(), priorSenti};  // vocabulary index (weil es ein neues Wort ist wissen wir, dass es ganz ans Ende muss); word polarity
				word2atr.insert(pair<string, Word_atr>(strtok.token(k+1), temp));
				pdoc->priorSentiLabels[k] = priorSenti;
				
			} 
			else { // word seen before
				// Beachte: Dabei wird diese Funktion (analyzeCorpus) nur einmal am Anfang aufgerufen (in read_dataStream). Später können gleiche Worte auch unterschiedliche Labels bekommen
				pdoc->words[k] = it->second.id; // Das Wort ist also schon mit Index bekannt (it->second.id)
				pdoc->priorSentiLabels[k] = it->second.polarity; // Auch die Polarität/Sentiment wurde bereits ermittelt
			}
		}
		// Während die Variable "docs" also immer noch sehr nah bei dem rohen Input war, übertragen wir dies nun in die statische Datenstruktur document (pdoc)
		add_doc(pdoc, i);
	} 
	    
	    
	// update number of words
	vocabSize = word2atr.size();
	aveDocLength = corpusSize/numDocs;

	if (write_wordmap(result_dir + wordmapfile, word2atr)) {
		printf("ERROR! Can not write wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}
	if (read_wordmap(result_dir + wordmapfile, id2word)) {
		printf("ERROR! Can not read wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}
	/*if (write_wordmap(result_dir +"1"+ wordmapfile, word2atr)) {
		printf("ERROR! Can not write wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}*/

	docs.clear();
	return 0;
}

int dataset::analyzeNewCorpus(vector<string>& docs) {

	mapword2atr::iterator it;
	mapword2id::iterator vocabIt;
	mapword2prior::iterator sentiIt;
	map<int, int>::iterator idIt;

	string line;
	numDocs = docs.size();
	// Zählt die Anzahl an unterschiedlichen Worten/Vokabeln über alle Dokumente
	vocabSize = 0;
	// Gibt die Anzahl an allen Wörter in allen Dokumenten an (auch Duplikate)
	corpusSize = 0;
	aveDocLength = 0;

	// allocate memory for corpus/dataset pdocs
	if (pdocs) {
		deallocate();
		pdocs = new document*[numDocs];
	}
	else {
		pdocs = new document*[numDocs];
	}

	for (int i = 0; i < (int)docs.size(); ++i) {
		line = docs.at(i); // aktuelles Dokument
						   //strtokenizer ist eine eigene Klasse des Projekts. Darüber können wir immer wieder auf der aktuellen line (einzelnes Dokument) arbeiten
		strtokenizer strtok(line, " \t\r\n");  // \t\r\n are the separators
		int docLength = strtok.count_tokens();

		if (docLength <= 0) {
			printf("Invalid (empty) document!\n");
			deallocate();
			numDocs = vocabSize = 0;
			return 1;
		}

		corpusSize += docLength - 1; // the first word is document name/id (z.B. "d0"; deshalb subtrahieren wir 1)

									 // allocate memory for the new document_i 
		document * pdoc = new document(docLength - 1);
		pdoc->docID = strtok.token(0).c_str(); // z.B. "d0"

											   // generate ID for the tokens in the corpus, and assign each word token with the corresponding vocabulary ID.
		for (int k = 0; k < docLength - 1; k++) {
			int priorSenti = -1;
			// Der Map-Iterator springt an die Stelle, wo dieser Token (Wort) vorkommt
			it = word2atr.find(strtok.token(k + 1).c_str());

			if (it == word2atr.end()) { //  i.e., new word; denn .find liefert nur .end() zurück, wenn der Token nicht gefunden wurde
				pdoc->words[k] = word2atr.size(); //(weil es ein neues Wort ist wissen wir, dass es ganz ans Ende muss)
				sentiIt = sentiLex.find(strtok.token(k + 1).c_str()); // check whether the word token can be found in the sentiment lexicon
																	  // incorporate sentiment lexicon
				if (sentiIt != sentiLex.end()) {
					// Wenn das Wort also im Sentilexicon (mpqa o.ä.) vorkommt, dann setze für das Wort den entspr. Senti-Prior-Label aus diesem Lexicon
					priorSenti = sentiIt->second.id;
				}

				// insert sentiment info into word2atr
				Word_atr temp = { word2atr.size(), priorSenti };  // vocabulary index (weil es ein neues Wort ist wissen wir, dass es ganz ans Ende muss); word polarity
				word2atr.insert(pair<string, Word_atr>(strtok.token(k + 1), temp));
				pdoc->priorSentiLabels[k] = priorSenti;

			}
			else { // word seen before
				   // Beachte: Dabei wird diese Funktion (analyzeCorpus) nur einmal am Anfang aufgerufen (in read_dataStream). Später können gleiche Worte auch unterschiedliche Labels bekommen
				pdoc->words[k] = it->second.id; // Das Wort ist also schon mit Index bekannt (it->second.id)
				pdoc->priorSentiLabels[k] = it->second.polarity; // Auch die Polarität/Sentiment wurde bereits ermittelt
			}
		}
		// Während die Variable "docs" also immer noch sehr nah bei dem rohen Input war, übertragen wir dies nun in die statische Datenstruktur document (pdoc)
		add_doc(pdoc, i);
	}


	// update number of words
	vocabSize = word2atr.size();
	aveDocLength = corpusSize / numDocs;

	if (write_wordmap(result_dir + wordmapfile, word2atr)) {
		printf("ERROR! Can not write wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}
	if (read_wordmap(result_dir + wordmapfile, id2word)) {
		printf("ERROR! Can not read wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}

	docs.clear();
	return 0;
}

int dataset::read_newData(vector<string>& docs) {
	mapword2id::iterator it;
	map<int, int>::iterator _it;
	mapword2atr::iterator itatr;
	mapword2prior::iterator sentiIt;
	string line;
	//char buff[BUFF_SIZE_LONG];
	
	// Liest die Vokabeln der alten Trainingsdokumente ein und bildet daraus Maps
	read_wordmap(model_dir + "wordmap.txt", word2id);  // map word2id
	read_wordmap(model_dir + "wordmap.txt", id2word);  // map id2word

	if (word2id.size() <= 0) {
		printf("Invalid wordmap!\n");
		return 1;
	}
	
	if (numDocs <= 0) {
		printf("Error! No documents found in test data %s.\n", (data_dir + datasetFile).c_str());
		return 1;
	}

	// allocate memory
	if (pdocs) {
		deallocate();
	}
	else {
		pdocs = new document*[numDocs];
	}
	_pdocs = new document*[numDocs];
	vocabSize = 0;
	corpusSize = 0;

	// process each document in the new data
	for (int i = 0; i < numDocs; i++) {
		line = docs.at(i);
		strtokenizer strtok(line, " \t\r\n"); // \t\r\n are separators
		int docLength = strtok.count_tokens();
		if (docLength <= 0) {
			printf("Invalid (empty) document!\n");
			deallocate();
			numDocs = 0;
			vocabSize = 0;
			return 1;
		}

		corpusSize += docLength - 1;
		// Hiermit modellieren wir das neue Doc aufgrund bekannter Word-IDs aus den Trainingsdaten
		vector<int> doc;
		// Hier modellieren wir das neue Doc mittels Word-IDs, aber nur bezüglich der neuen Daten (loc. Voc)! Eine Wort-ID kann also z.B. schon in den Trainingsdaten vorgekommen sein. Dennoch benutzen wir hier eine neue dafür
		// Entspricht also im Grunde dem Wort Index der Test-Daten
		vector<int> _doc;

		vector<int> priorSentiLabels;

		// process each token in the document
		for (int k = 1; k < docLength; k++) {
			int priorSenti = -1;
			it = word2id.find(strtok.token(k).c_str());
			if (it == word2id.end()) { // neues Wort in den Testdaten (unbekannt im glob. Voc)
				newWords.push_back(strtok.token(k).c_str());
				// neue Einträge sollten damit für word2id, id2word (glob. Voc) und id2_id, _id2id, word2atr (loc. Voc) entstehen
				// Die korrespondierenden counts dazu werden später in anderen Methoden gebildet
				// Beachte: word2atr ist nicht mit dem globalen Mapping zu verwechseln! Hier gilt es nur für das lokale Vokabular
				int new_glob_id = word2id.size();
				newWords1.push_back(new_glob_id);
				sentiIt = sentiLex.find(strtok.token(k).c_str());
				if (sentiIt != sentiLex.end()) {
					priorSenti = sentiIt->second.id;
				}

				// insert into glob. voc.
				word2id.insert(pair<string, int>(strtok.token(k).c_str(), new_glob_id));
				id2word.insert(pair<int, string>(new_glob_id, strtok.token(k).c_str()));


				// insert sentiment info into loc. word2atr
				Word_atr temp = { word2atr.size(), priorSenti };  // vocabulary index; word polarity
				word2atr.insert(pair<string, Word_atr>(strtok.token(k).c_str(), temp));
				priorSentiLabels.push_back(priorSenti);

				// insert into loc. voc.
				int _id;
				_id = id2_id.size(); // Die letzte Stelle der lok. Map wo die Word-ID eingepflegt wird
				id2_id.insert(pair<int, int>(new_glob_id, _id)); // Ein Paar bestehend aus glob. Wort-ID und lok. Wort-ID der Map wird eingefügt
				_id2id.insert(pair<int, int>(_id, new_glob_id)); // Ein Paar bestehend aus lok. Wort-ID und glob. Wort-ID der Map wird eingefügt

				//int new_loc_id = id2_id.size();
				doc.push_back(new_glob_id);
				_doc.push_back(_id);

				// TODO: Die neue wordmap rausschreiben!
			}
			else { // Ansonsten ist das Wort schon im glob. Voc bekannt und wir suchen das Vorkommen
				int _id;
				_it = id2_id.find(it->second); // Wir suchen nach der glob. Word-ID
				if (_it == id2_id.end()) { // Das Wort ist zwar im glob. Voc. bekannt, aber die entsprechende Word-ID wurde noch nicht in die lokale id2_id Map eingepflegt
					_id = id2_id.size(); // Die letzte Stelle der lokalen Map wo die Word-ID eingepflegt wird
					id2_id.insert(pair<int, int>(it->second, _id)); // Ein Paar bestehend aus glob. Wort-ID und loc. Wort-ID der Map wird eingefügt
					_id2id.insert(pair<int, int>(_id, it->second));
				}
				else {	// Die Wort-ID wurde bereits in id2_id eingepflegt
					_id = _it->second; // Die Stelle in id2_id unter der die bekannte Word-ID gespeichert wurde
				}

				doc.push_back(it->second); // Hier wird die glob. Word-ID gepusht
				_doc.push_back(_id); // Hier wird der Index/Stelle (in der Map id2_id) der Word-ID gepusht. Dies entspricht wiederum der Word-ID für nur die neuen Dokumente

				// 'word2atr' is specific to new/test dataset (es gilt also nur für loc. Voc.!!!)
				itatr = word2atr.find(strtok.token(k).c_str());
				int priorSenti = -1;
				if (itatr == word2atr.end()) {
					sentiIt = sentiLex.find(strtok.token(k).c_str()); // check whether the word token can be found in the sentiment lexicon
					if (sentiIt != sentiLex.end()) {
						priorSenti = sentiIt->second.id; // Gibt zu diesem Wort das Senti-Label
					}
					// encode sentiment info into word2atr
					// Falls zu dem Wort also ein Eintrag in mpqa vorliegt, so wird dieser hier benutzt
					Word_atr temp = { _id, priorSenti };  // vocabulary index; word polarity (Beachte: Falls im Lexicon nichts gefunden wurde, so wird -1 übernommen)
					word2atr.insert(pair<string, Word_atr>(strtok.token(k), temp));
					priorSentiLabels.push_back(priorSenti);
				}
				else { // Falls das Wort direkt gefunden wurde, so müssen wir nicht nochmal ins Lexicon schauen, sondern übernehmen direkt die Polarity
					priorSentiLabels.push_back(itatr->second.polarity);
				}

			} // end else: Für Worte/Token die bereits bekannt sind
		} // end for: Alle Tokens/Worte des neuen Dokuments sind bearbeitet

		  // allocate memory for new doc
		document * pdoc = new document(doc, priorSentiLabels, "inference");
		document * _pdoc = new document(_doc, priorSentiLabels, "inference");
		pdoc->docID = strtok.token(0).c_str();
		_pdoc->docID = strtok.token(0).c_str();

		// add new doc
		add_doc(pdoc, i);
		_add_doc(_pdoc, i);
	} // end for: Alle Dokumente wurden eingelesen & bearbeitet

	  // update number of new words
	vocabSize = word2id.size();
	newVocabSize = newWords.size();
	aveDocLength = corpusSize / numDocs;

	// Neue Wordmap speichern
	if (write_wordmap1(result_dir +wordmapfile, word2id)) {
		printf("ERROR! Can not write wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}

	if (read_wordmap(result_dir + wordmapfile, id2word)) {
		printf("ERROR! Can not read wordmap file %s!\n", wordmapfile.c_str());
		return 1;
	}

	docs.clear();
	return 0;
}




void dataset::deallocate() 
{
	if (pdocs) {
		for (int i = 0; i < numDocs; i++) 
			delete pdocs[i];		
		delete [] pdocs;
		pdocs = NULL;
	}
	
	if (_pdocs) {
		for (int i = 0; i < numDocs; i++) 
			delete _pdocs[i];
		delete [] _pdocs;
		_pdocs = NULL;
	}
}
    

void dataset::add_doc(document * doc, int idx) {
    if (0 <= idx && idx < numDocs)
        pdocs[idx] = doc;
}   

void dataset::_add_doc(document * doc, int idx) {
    if (0 <= idx && idx < numDocs) {
	    _pdocs[idx] = doc;
    }
}

// Hier liest man im Grunde nur das Textfile der Sentiments ein (mpqa)
int dataset::read_senti_lexicon(string sentiLexiconFile) {
	sentiLex.clear();
	char buff[BUFF_SIZE_SHORT];
    string line;
    vector<double> wordPrior;
    int labID;
    double tmp, val;
    int numSentiLabs;
    
    FILE * fin = fopen(sentiLexiconFile.c_str(), "r");
    if (!fin) {
		printf("Cannot read file %s!\n", sentiLexiconFile.c_str());
		return 1;
    }    
    
	// Die while Schleife läuft über das ganze mpqa. Zeile für Zeile wird eingelesen
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin) != NULL) {
		line = buff;
		strtokenizer strtok(line, " \t\r\n");
			
		if (strtok.count_tokens() < 1)  {
			printf("Warning! The strtok count in the lexicon line [%s] is smaller than 2!\n", line.c_str());
		}
		else {	
			tmp = 0.0;
			labID = 0;
			wordPrior.clear();
			numSentiLabs = strtok.count_tokens();
			for (int k = 1; k < strtok.count_tokens(); k++) { // k=1, weil wir das eigentliche Wort auslassen und in der for-Schleife nur die Sentiment-Werte dafür betrachten
				val = atof(strtok.token(k).c_str()); // Der String wird in ein double "val" konvertiert
				if (tmp < val) {
					tmp = val;
					labID = k-1; // Die labID gibt also das Sentiment mit dem höchsten Wert an. Bei [0.05 0.9 0.05] würde somit labID=1 sein
				}
				wordPrior.push_back(val);
			}
			Word_Prior_Attr temp = {labID, wordPrior};  // sentiment label ID, sentiment label distribution
			sentiLex.insert(pair<string, Word_Prior_Attr >(strtok.token(0), temp));
		}
    }
    
	if (sentiLex.size() <= 0) {
		printf("Can not find any sentiment lexicon in file %s!\n", sentiLexiconFile.c_str());
		return 1;
	}
	
    fclose(fin);
    return 0;
}


int dataset::write_wordmap(string wordmapfile, mapword2atr &pword2atr) {

    FILE * fout = fopen(wordmapfile.c_str(), "w");
    if (!fout) {
		printf("Cannot open file %s to write!\n", wordmapfile.c_str());
		return 1;
    }    
    
    mapword2atr::iterator it;
	// Die erste Zeile in wordmap.txt ist also die Anzahl an Wörtern
    fprintf(fout, "%d\n", (int)(pword2atr.size())); // wieviele Worte gibt es insgesamt (vocabSize)
    for (it = pword2atr.begin(); it != pword2atr.end(); it++) {
		// wordmap.txt wird dann gefüllt mit dem Wort (it->first) und dem Wortindex (it->second.id) (an welcher Stelle kommt es zum ersten Mal vor)
	    fprintf(fout, "%s %d\n", (it->first).c_str(), it->second.id);
    }
    
    fclose(fout);
    return 0;
}

int dataset::write_wordmap1(string wordmapfile, mapword2id &pword2id) {

	FILE * fout = fopen(wordmapfile.c_str(), "w");
	if (!fout) {
		printf("Cannot open file %s to write!\n", wordmapfile.c_str());
		return 1;
	}

	mapword2id::iterator it;
	// Die erste Zeile in wordmap.txt ist also die Anzahl an Wörtern
	fprintf(fout, "%d\n", (int)(pword2id.size())); // wieviele Worte gibt es insgesamt (vocabSize)
	for (it = pword2id.begin(); it != pword2id.end(); it++) {
		// wordmap.txt wird dann gefüllt mit dem Wort (it->first) und dem Wortindex (it->second.id) (an welcher Stelle kommt es zum ersten Mal vor)
		fprintf(fout, "%s %d\n", (it->first).c_str(), it->second);
	}

	fclose(fout);
	return 0;
}

// Liest wordmap.txt ein. Die erste Zeile beschreibt numVocSize (Anzahl aller unterschiedlicher Vokabeln)
// Die folgenden Zeilen beschreiben Paare von Word-String und korrespondierender Word-ID für den gescannten Corpus
// z.B. "brutal 35"
// Hier wird eine mapid2word pid2word beschrieben
int dataset::read_wordmap(string wordmapfile, mapid2word &pid2word) {
    pid2word.clear(); 
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
		printf("Cannot open file %s to read!\n", wordmapfile.c_str());
		return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
		fgets(buff, BUFF_SIZE_SHORT - 1, fin);
		line = buff;
		strtokenizer strtok(line, " \t\r\n");
		if (strtok.count_tokens() != 2) {
			printf("Warning! Line %d in %s contains less than 2 words!\n", i+1, wordmapfile.c_str());
			continue;
		}
		
		pid2word.insert(pair<int, string>(atoi(strtok.token(1).c_str()), strtok.token(0))); // Wort-ID (strtok.token(1)) und Wort (strtok.token(0)) werden gespeichert
    }
    
    fclose(fin);
    return 0;
}

// Hier wird dagegen eine mapword2id beschrieben
int dataset::read_wordmap(string wordmapfile, mapword2id& pword2id) {
    pword2id.clear();
    char buff[BUFF_SIZE_SHORT];
    string line;

    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
		printf("Cannot read file %s!\n", wordmapfile.c_str());
		return 1;
    }    
        
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
		fgets(buff, BUFF_SIZE_SHORT - 1, fin);
		line = buff;
		strtokenizer strtok(line, " \t\r\n");
		if (strtok.count_tokens() != 2) {
			continue;
		}
		pword2id.insert(pair<string, int>(strtok.token(0), atoi(strtok.token(1).c_str())));
    }
    
    fclose(fin);
    return 0;
}

