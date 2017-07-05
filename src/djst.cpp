/**********************************************************************
		       dynamic Joint Sentiment-Topic (dJST) Model
***********************************************************************/

   
#include "djst.h"
using namespace std;

djst::djst(void) {

    numSentiLabs = 0; 
	numTopics = 0;
	numDocs = 0; 
	vocabSize = 0;
	newNumDocs = 0;
	newVocabSize = 0;
	_beta = -1.0;
	
	wordmapfile = "wordmap.txt";
    tassign_suffix = ".tassign";
    pi_suffix = ".pi";
    theta_suffix = ".theta";
    phi_suffix = ".phi";
    others_suffix = ".others";
    twords_suffix = ".twords";
	model_name = "";
	data_dir = "";
	datasetFile = "";
	result_dir = "";
	sentiLexFile = "";

	updateParaStep = -1;
	savestep = 20;
	twords = 20;
	niters = 40;
	
	putils = new utils();
	pmodelData = NULL;
	pnewData = NULL;
}


djst::~djst(void) {

	if (putils)
		delete putils;
		
	if (pmodelData)
		delete pmodelData;
	
	if (pnewData)
		delete pnewData;

	if (firstModel)	// added
		delete firstModel;
}


int djst::init(int argc, char ** argv) {
	numSentiLabs = 0;
	numTopics = 0;

	firstModel = new model();
	// Die ganzen Argumente in test.properties werden eingelesen (wie Pfade für result-directory (result_dir) und data-directory (data_dir))
	// Diese Werte werden dann in dieses (this) Modell geschrieben
    if (putils->parse_args_djst(argc, argv, this)) {
	    return 1;
    }

	if (putils->parse_args_est(argc, argv, firstModel)) {
		return 1;
	}
	
	// Das initialisiert das erste Modell auf Daten der ersten Epoche
	// Labels werden gesampled und counts gebildet
	if (firstModel->initFirstModel()) {
		printf("Throw exception in initFirstModel()!\n");
		return 1;
	}
	sliding_window_phi.push_back(firstModel->phi_lzw);


	// Die ersten drei Modelle werden unabhängig voneinander trainiert
	// (auf den ersten 3 Zeitschlitzen)
	for (size_t epoch = 2; epoch < time_slices+1; epoch++) {
		if (firstModel->initNewModel(epoch, model_dir)) {
			printf("Throw exception in initNewModel(), NO %d!\n", epoch);
			return 1;
		}
	sliding_window_phi.push_back(firstModel->phi_lzw);
	}
	//delete firstModel;

	// train djst model (taking old models into account) as long as new data is available
	for (size_t epoch = time_slices+1; epoch < 101; epoch++) {
		newsigma_lzw.clear();
		trainNextModel(epoch);
		// update the sliding window of word distributions
		std::rotate(sliding_window_phi.begin(), sliding_window_phi.begin() + 1, sliding_window_phi.end());
		sliding_window_phi.pop_back();
		sliding_window_phi.push_back(newsigma_lzw);
	}

    return 0;
}

// Gleicht quasi init_estimate wie bei model
int djst::init_djstestimate2() {
	// Hier initialisieren (zufällig) wir die ersten Sentiment-/Topic-Labels. Somit kann dann das "richtige" Sampling starten
	srand(1234);
	int sentiLab, topic;
	new_z.resize(pnewData->numDocs);
	new_l.resize(pnewData->numDocs);

	for (int m = 0; m < pnewData->numDocs; m++) {
		int docLength = pnewData->pdocs[m]->length;
		new_z[m].resize(docLength);
		new_l[m].resize(docLength);
		for (int t = 0; t < docLength; t++) {
			if (pnewData->pdocs[m]->words[t] < 0) { // z.B. wenn t größer als die docLength ist ;)
				printf("ERROR! word token %d has index smaller than 0 in doc[%d][%d]\n", pnewData->pdocs[m]->words[t], m, t);
				return 1;
			}

			// sample sentiment label
			if ((pnewData->pdocs[m]->priorSentiLabels[t] > -1) && (pnewData->pdocs[m]->priorSentiLabels[t] < numSentiLabs)) {
				sentiLab = pnewData->pdocs[m]->priorSentiLabels[t]; // incorporate prior information into the model  
			}
			else { // Wenn keine Prior Information (über das Lexicon) vorliegt, so samplen wir zufällig ein Label
				sentiLab = (int)(((double)rand() / RAND_MAX) * numSentiLabs);
				if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1;
			}
			new_l[m][t] = sentiLab;

			// sample topic label
			topic = (int)(((double)rand() / RAND_MAX) * numTopics);
			if (topic == numTopics)  topic = numTopics - 1;
			new_z[m][t] = topic;
			new_nd[m]++;
			new_ndl[m][sentiLab]++;
			new_ndlz[m][sentiLab][topic]++;
			new_nlzw[sentiLab][topic][pnewData->pdocs[m]->words[t]]++;
			new_nlz[sentiLab][topic]++;
		}
	}
	return 0;
}

int djst::trainNextModel(int epoch) {
	newsigma_lzw.clear();
	pnewData = new dataset(result_dir, model_dir);

	if (sentiLexFile != "") {
		if (pnewData->read_senti_lexicon((sentiLexFile).c_str())) {
			printf("Error! Cannot read sentiFile %s!\n", (sentiLexFile).c_str());
			delete pnewData;
			return 1;
		}
		this->sentiLex = pnewData->sentiLex;
	}
	fin.open((data_dir + std::to_string(epoch) + ".dat").c_str(), ifstream::in);
	if (!fin) {
		printf("Error! Cannot read dataset %s!\n", (data_dir + std::to_string(epoch) + ".dat").c_str());
		return 1;
	}

	if (pnewData->read_dataStream1(fin)) {
		printf("Throw exception in function read_dataStream1()! \n");
		delete pnewData;
		return 1;
	}

	word2atr = pnewData->word2atr; // "access {2984, sentiLabel}" glob. Voc
	id2word = pnewData->id2word; // "2984 access" glob. Voc
	numDocs = pnewData->numDocs;

	// init_parameters -> set up counts and incorporate prior senti-information
	// Schwierigkeit: Setze Prior über Lambda in Beta NUR DANN (!), wenn das Wort neu ist.
	// Denn ansonsten sollten wir das  neue Beta nach der Vorschrift beta=µ*sigma errechnen

	// Hier werden sämtliche new_count-parameter initialisiert (z.B. new_nlzw=0)
	if (init_parameters2()) {
		printf("Throw exception in init_parameters!\n");
		return 1;
	}

	if (init_djstestimate2()) {
		printf("Throw exception in init_djstestimate()!\n");
		return 1;
	};

	if (djst_estimate(epoch)) {
		printf("Throw exception in djst_estimate(epoch)!\n");
		return 1;
	};
	fin.close();

	// added; create data for pyLDAvis
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			for (int r = 0; r < vocabSize; r++) {
				term_senti_frequency[r] += new_nlzw[l][z][r];
			}
		}
	}

	int fixed_senti = 1;
	for (int z = 0; z < numTopics; z++) {
		for (int r = 0; r < vocabSize; r++) {
			term_senti_frequency_pos[r] += new_nlzw[fixed_senti][z][r];
		}
	}

	fixed_senti = 2;
	for (int z = 0; z < numTopics; z++) {
		for (int r = 0; r < vocabSize; r++) {
			term_senti_frequency_neg[r] += new_nlzw[fixed_senti][z][r];
		}
	}

	save_data_pyldavis(epoch);
	delete_model_parameters();

	return 0;
}


// Hier geschieht viel bzgl. der Berechnung (bzw. hier werden alle wichtigen Funktionen dafür gecallt)
// Das Modell wird auf die Daten trainiert (Parameter Phi,... werden estimated)
int djst::djst_estimate(int epoch) {
	int sentiLab, topic;
	printf("Sampling %d iterations!\n", niters);

	for (liter = 1; liter <= niters; liter++) {
		for (int m = 0; m < numDocs; m++) {
			for (int n = 0; n < pnewData->pdocs[m]->length; n++) {
				// Hier werden auch die counts geupdatet (wie z.B. nlzw)
				// Auf diesen neuen counts können dann die Parameter estimated werden (z.B. compute_phi_lzw())
				djst_sampling(m, n, sentiLab, topic);
				new_l[m][n] = sentiLab;
				new_z[m][n] = topic;
			}
		}
		
		if (updateParaStep > 0 && liter % updateParaStep == 0) {
			this->update_Parameters();
		}

		if (savestep > 0 && liter % savestep == 0) {
			if (liter == niters) break;
			printf("Iteration %d ...\n", liter); // added

		//	compute_newpi1();
		//	compute_newtheta();
		//	compute_newphi1();
			//save_model1(putils->generate_model_name(liter));
		}
	}

	printf("Gibbs sampling completed!\n");
	printf("Saving the final model! for %d \n", epoch);
	compute_newpi1();
	compute_newtheta();
	compute_newphi1();


	vocabSize = pnewData->vocabSize;
	expected_counts_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		expected_counts_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			expected_counts_lzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++) {
				expected_counts_lzw[l][z][r] = newphi_lzw[l][z][r]*new_nlzw[l][z][r];
				expected_counts_sum_lz[l][z] += expected_counts_lzw[l][z][r];
			}
		}
	}


	for (int l = 0; l < numSentiLabs; l++) {
		newsigma_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			newsigma_lzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++) {
				if (expected_counts_sum_lz[l][z] != 0) {
					newsigma_lzw[l][z][r] = (expected_counts_lzw[l][z][r] / expected_counts_sum_lz[l][z]);
				}
			}
		}
	}

	save_model(putils->generate_model_name(-1), epoch);
	return 0;
}


// Hier samplen wir für die bekannten Worte/Vokabeln aus den neuen Dokumenten jeweils neue Topic&Sentiment Labels
// Dabei betrachten wir nur Wörter des aktuellen Zeitschlitzes! 
// (Später wird auch das globale Vokabular betrachtet; fürs Sampling selbst werden aber nur die lokalen betrachtet)
// Achtung: Wir gehen hier mit dem lokalen Vokabular rein. Deshalb failt es!
int djst::djst_sampling(int m, int n, int& sentiLab, int& topic) {
	srand(1234);
	sentiLab = new_l[m][n];
	topic = new_z[m][n];

	int w = pnewData->pdocs[m]->words[n];
	double u;

	new_nd[m]--;
	new_ndl[m][sentiLab]--;
	new_ndlz[m][sentiLab][topic]--;
	new_nlzw[sentiLab][topic][w]--; // Der count welcher hier durch die vorausgegangene sampling-iteration ermittelt wurde. nlzw[l][k][w] bezieht sich dagegen auf den bereits gelernten Count des trainierten Modells
	new_nlz[sentiLab][topic]--;

	// do multinomial sampling via cumulative method
	for (int l = 0; l < numSentiLabs; l++) {
		for (int k = 0; k < numTopics; k++) {
			// TODO exchange betaSum_lz for µ
			new_p[l][k] = (new_nlzw[l][k][w] + beta_lzw[l][k][w]) / (new_nlz[l][k] + betaSum_lz[l][k]) *
				(new_ndlz[m][l][k] + alpha_lz[l][k]) / (new_ndl[m][l] + alphaSum_l[l]) *
				(new_ndl[m][l] + gamma_dl[m][l]) / (new_nd[m] + gammaSum_d[m]);
		}
	}

	// accumulate multinomial parameters
	// hier werden letztlich "sentiLab" und "topic" gesampled
	for (int l = 0; l < numSentiLabs; l++) {
		for (int k = 0; k < numTopics; k++) {
			if (k == 0) {
				if (l == 0) continue;
				else new_p[l][k] += new_p[l - 1][numTopics - 1];
			}
			else new_p[l][k] += new_p[l][k - 1];
		}
	}
	// probability normalization
	u = ((double)rand() / RAND_MAX) * new_p[numSentiLabs - 1][numTopics - 1];

	for (sentiLab = 0; sentiLab < numSentiLabs; sentiLab++) {
		for (topic = 0; topic < numTopics; topic++) {
			if (new_p[sentiLab][topic] > u) {
				goto stop;
			}
		}
	}

stop:
	if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1; // the max value of label is (S - 1)
	if (topic == numTopics) topic = numTopics - 1;

	// add estimated 'z' and 'l' to counts
	new_nd[m]++;
	new_ndl[m][sentiLab]++;
	new_ndlz[m][sentiLab][topic]++;
	new_nlzw[sentiLab][topic][w]++;
	new_nlz[sentiLab][topic]++;

	return 0;
}

int djst::initFirstModel() {
	pnewData = new dataset(result_dir);

	if (sentiLexFile != "") {
		if (pnewData->read_senti_lexicon((sentiLexFile).c_str())) {
			printf("Error! Cannot read sentiFile %s!\n", (sentiLexFile).c_str());
			delete pnewData;
			return 1;
		}
		this->sentiLex = pnewData->sentiLex;
	}

	// read first training set
	fin.open((data_dir + "1.dat").c_str(), ifstream::in);
	if (!fin) {
		printf("Error! Cannot read dataset %s!\n", (data_dir + "1.dat").c_str());
		return 1;
	}
	fin.close();
	// Dort wird analyzeCorpus aufgerufen, um die Trainingsdaten zu verarbeiten
	if (pnewData->read_dataStream(fin)) {
		printf("Throw exception in function read_dataStream()! \n");
		delete pnewData;
		return 1;
	}

	word2atr = pnewData->word2atr; // "access {2984, sentiLabel}" glob. Voc
	id2word = pnewData->id2word; // "2984 access"

	// Hier werden sämtliche count-parameter initialisiert (z.B. nlzw)
	if (init_parameters()) {
		printf("Throw exception in init_parameters!\n");
		return 1;
	}

	printf("Testset statistics: \n");
	printf("numDocs = %d\n", pnewData->numDocs);
	printf("vocabSize = %d\n", pnewData->vocabSize);
	printf("numNew_word = %d\n", (int)(pnewData->newWords.size()));


	// Erweitere die Größe von nlzw um die neuen Wörter
	nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			nlzw[l][z].resize(vocabSize + pnewData->newWords.size());
			for (int r = vocabSize; r < vocabSize + pnewData->newWords.size(); r++) {
				nlzw[l][z][r] = 0;
			}
		}
	}

	// init inf
	// Hier initialisieren (zufällig) wir die ersten Sentiment-/Topic-Labels. Somit kann dann das "richtige" Sampling starten
	int sentiLab, topic;
	new_z.resize(pnewData->numDocs);
	new_l.resize(pnewData->numDocs);

	for (int m = 0; m < pnewData->numDocs; m++) {
		int docLength = pnewData->_pdocs[m]->length;
		new_z[m].resize(docLength);
		new_l[m].resize(docLength);
		for (int t = 0; t < docLength; t++) {
			if (pnewData->_pdocs[m]->words[t] < 0) { // z.B. wenn t größer als die docLength ist ;)
				printf("ERROR! word token %d has index smaller than 0 in doc[%d][%d]\n", pnewData->_pdocs[m]->words[t], m, t);
				return 1;
			}

			// sample sentiment label
			if ((pnewData->pdocs[m]->priorSentiLabels[t] > -1) && (pnewData->pdocs[m]->priorSentiLabels[t] < numSentiLabs)) {
				sentiLab = pnewData->pdocs[m]->priorSentiLabels[t]; // incorporate prior information into the model  
			}
			else { // Wenn keine Prior Information (über das Lexicon) vorliegt, so samplen wir zufällig ein Label
				sentiLab = (int)(((double)rand() / RAND_MAX) * numSentiLabs);
				if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1;
			}
			new_l[m][t] = sentiLab;

			// sample topic label
			topic = (int)(((double)rand() / RAND_MAX) * numTopics);
			if (topic == numTopics)  topic = numTopics - 1;
			new_z[m][t] = topic;

			new_nd[m]++;
			new_ndl[m][sentiLab]++;
			new_ndlz[m][sentiLab][topic]++;
			new_nlzw[sentiLab][topic][pnewData->_pdocs[m]->words[t]]++;
			new_nlz[sentiLab][topic]++;
		}
	}

	return 0;
}

int djst::initNewModel(int epoch) {

	pmodelData = new dataset();
	pnewData = new dataset(result_dir, model_dir);

	// Das liest die Parameter (.others) von einem alten/trainierten Modell ein (wie z.B. numTopics, numDocs,...)
	if (read_model_setting(model_dir + model_name + ".others")) {
		printf("Throw exception in read_para_setting()!\n");
		return 1;
	}

	// load model old model in pmodelData
	/*if (load_model(model_dir + model_name + ".tassign")) {
		printf("Throw exception in load_model()!\n");
		return 1;
	}*/

	if (sentiLexFile != "") {
		if (pnewData->read_senti_lexicon((sentiLexFile).c_str())) {
			printf("Error! Cannot read sentiFile %s!\n", (sentiLexFile).c_str());
			delete pnewData;
			return 1;
		}
		this->sentiLex = pnewData->sentiLex;
	}

	// read first training set
	fin.open((data_dir + std::to_string(epoch) + ".dat").c_str(), ifstream::in);
	if (!fin) {
		printf("Error! Cannot read dataset %s!\n", (data_dir + std::to_string(epoch) + ".dat").c_str());
		return 1;
	}
	fin.close();
	// Trainingsdaten verarbeiten
	if (pnewData->read_dataStream1(fin)) {
		printf("Throw exception in function read_dataStream()! \n");
		delete pnewData;
		return 1;
	}

	this->newNumDocs = pnewData->numDocs;
	this->newVocabSize = pnewData->vocabSize;

	if (newVocabSize == 0) { // Falls nur neue Worte in den Trainingsdaten auftreten
		printf("ERROR! Vocabulary size of test set after removing unseen words is 0! \n");
		return 1;
	}

	word2atr = pnewData->word2atr; // "access {2984, sentiLabel}" glob. Voc
	id2word = pnewData->id2word; // "2984 access"
								 // delete_model_parameters();

								 // Hier werden sämtliche count-parameter initialisiert (z.B. nlzw)
	if (init_parameters()) {
		printf("Throw exception in init_parameters!\n");
		return 1;
	}

	printf("Testset statistics: \n");
	printf("numDocs = %d\n", pnewData->numDocs);
	printf("vocabSize = %d\n", pnewData->vocabSize);
	printf("numNew_word = %d\n", (int)(pnewData->newWords.size()));


	// Erweitere die Größe von nlzw um die neuen Wörter
	nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			nlzw[l][z].resize(vocabSize + pnewData->newWords.size());
			for (int r = vocabSize; r < vocabSize + pnewData->newWords.size(); r++) {
				nlzw[l][z][r] = 0;
			}
		}
	}

	// init inf
	// Hier initialisieren (zufällig) wir die ersten Sentiment-/Topic-Labels. Somit kann dann das "richtige" Sampling starten
	int sentiLab, topic;
	new_z.resize(pnewData->numDocs);
	new_l.resize(pnewData->numDocs);

	for (int m = 0; m < pnewData->numDocs; m++) {
		int docLength = pnewData->_pdocs[m]->length;
		new_z[m].resize(docLength);
		new_l[m].resize(docLength);
		for (int t = 0; t < docLength; t++) {
			if (pnewData->_pdocs[m]->words[t] < 0) { // z.B. wenn t größer als die docLength ist ;)
				printf("ERROR! word token %d has index smaller than 0 in doc[%d][%d]\n", pnewData->_pdocs[m]->words[t], m, t);
				return 1;
			}

			// sample sentiment label
			if ((pnewData->pdocs[m]->priorSentiLabels[t] > -1) && (pnewData->pdocs[m]->priorSentiLabels[t] < numSentiLabs)) {
				sentiLab = pnewData->pdocs[m]->priorSentiLabels[t]; // incorporate prior information into the model  
			}
			else { // Wenn keine Prior Information (über das Lexicon) vorliegt, so samplen wir zufällig ein Label
				sentiLab = (int)(((double)rand() / RAND_MAX) * numSentiLabs);
				if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1;
			}
			new_l[m][t] = sentiLab;

			// sample topic label
			topic = (int)(((double)rand() / RAND_MAX) * numTopics);
			if (topic == numTopics)  topic = numTopics - 1;
			new_z[m][t] = topic;

			new_nd[m]++;
			new_ndl[m][sentiLab]++;
			new_ndlz[m][sentiLab][topic]++;
			new_nlzw[sentiLab][topic][pnewData->_pdocs[m]->words[t]]++;
			new_nlz[sentiLab][topic]++;
		}
	}

	return 0;
}


// read '.others' file
// (wie z.B. numTopics, numDocs,...)
int djst::read_model_setting(string filename) {

	char buff[BUFF_SIZE_LONG];
	string line;
	numSentiLabs = 0;
	numTopics = 0;
	numDocs = 0;
	vocabSize = 0;

	FILE * fin = fopen(filename.c_str(), "r");
	if (!fin) {
        printf("Cannot read file %s!\n", filename.c_str());
        return 1;
	}
    
	while (fgets(buff, BUFF_SIZE_LONG - 1, fin) != NULL) {
		line = buff; 
		strtokenizer values(line, ": \t\r\n={}[]"); // \t\r\n are separators

		if (values.token(0) == "numSentiLabs") {
			numSentiLabs = atoi(values.token(1).c_str());
		}
		else if (values.token(0) == "numTopics") {
			numTopics = atoi(values.token(1).c_str());
		}
		else if (values.token(0) == "numDocs") {
			numDocs = atoi(values.token(1).c_str());
		}
		else if (values.token(0) == "vocabSize") {
			vocabSize = atoi(values.token(1).c_str());
		}
		if (numSentiLabs > 0 && numTopics > 0 && numDocs > 0 && vocabSize > 0) {
			break;
		}
	}

	fclose(fin);
	
	if (numSentiLabs == 0 || numTopics == 0 || numDocs == 0 || vocabSize == 0) {
		cout << "Throw exception in reading model parameter settings!\n" << filename << endl;
		return 1;
	}
	else {
		cout<<"data_dir = "<<data_dir<<endl;
		cout<<"datasetFile = "<<datasetFile<<endl;
		cout<<"result_dir = "<<result_dir<<endl;
		cout<<"sentiLexFile = "<<sentiLexFile<<endl;
		cout<<"model_dir = "<<model_dir<<endl;
		cout<<"model_name = "<<model_name<<endl;
		cout<<"wordmapfile = "<<wordmapfile<<endl;
		cout<<"numTopics = "<<numTopics<<endl;
		cout<<"numSentiLabs = "<<numSentiLabs<<endl;
		cout<<"niters = "<<niters<<endl;
		cout<<"savestep = "<<savestep<<endl;
		cout<<"twords = "<<twords<<endl;
		cout<<"updateParaStep = "<<updateParaStep<<endl;
	}


	return 0;
}


// read '.tassign' file of previously trained model
// Nach dieser Methode ist unser bereits trainiertes Modell vollständig geladen (als dataset) 
// mit allen Dokumenten auf die trainiert wurde (mittels id-mapping wie in .tassign)
// und die entsprechenden counts (word,senti,topic) werden wiederhergestellt.
// Darüber könnte man dann auch wieder die Verteilungen phi, theta und pi herstellen
int djst::load_model(string filename) {

    char buff[BUFF_SIZE_LONG];
	string line;
    
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
	    printf("Cannot read file %s!\n", filename.c_str());
	    return 1;
    }

	// Die zuvor ausgelesenen Parameter werden nun in das dataset pmodelData geschrieben
	pmodelData->pdocs = new document*[numDocs]; // Array an Dokumenten
	pmodelData->vocabSize= vocabSize;
	pmodelData->numDocs= numDocs;
	l.resize(pmodelData->numDocs);
	z.resize(pmodelData->numDocs);

    for (int m = 0; m < numDocs; m++) { // lies Zeile für Zeile aus .tassign
		fgets(buff, BUFF_SIZE_LONG - 1, fin);  // first line - ignore the document ID
		fgets(buff, BUFF_SIZE_LONG - 1, fin);  // second line - read the sentiment label / topic assignments
		line = buff; 
	    strtokenizer strtok(line, " \t\r\n");
	    int length = strtok.count_tokens();
	
	    vector<int> words;
		vector<int> sentiLabs;
	    vector<int> topics;

	    for (int j = 0; j < length; j++) {
	        string token = strtok.token(j);
	        strtokenizer tok(token, ":");
	        if (tok.count_tokens() != 3) {
		        printf("Invalid word-sentiment-topic assignment format!\n");
		        return 1;
	        }
	    
			// Für ein triple wie z.B. 36:2:1 werden nacheinander 1.) Die Wort-id 2.) Das Senti-Label und 3.) Das Topic-Label ausgelesen
	        words.push_back(atoi(tok.token(0).c_str()));
			sentiLabs.push_back(atoi(tok.token(1).c_str()));
	        topics.push_back(atoi(tok.token(2).c_str()));
	    }
	
		// allocate and add training document to the corpus
		document * pdoc = new document(words);
		pmodelData->add_doc(pdoc, m);

		l[m].resize(sentiLabs.size());
		for (int j = 0; j < (int)sentiLabs.size(); j++) {
			l[m][j] = sentiLabs[j];
		}

		z[m].resize(topics.size());
		for (int j = 0; j < (int)topics.size(); j++) {
			z[m][j] = topics[j];
		}
	}
    fclose(fin);
    
	// init model counts
	nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			nlzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++) {
			    nlzw[l][z][r] = 0;
			}
		}
	}

	nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
            nlz[l][z] = 0;
		}
	}

	// recover count values from trained model
	for (int m = 0; m < pmodelData->numDocs; m++) {
		int docLength = pmodelData->pdocs[m]->length;
		for (int n = 0; n < docLength; n++) {
			int w = pmodelData->pdocs[m]->words[n];
			int sentiLab = this->l[m][n];
			int topic = this->z[m][n];

			nlzw[sentiLab][topic][w]++;
			nlz[sentiLab][topic]++;
		}
	}
	
    return 0;
}


// Hier werden sämtliche count-parameters für die 
// neuen Daten (wie z.B. new_nlzw=0) initialisiert. 
// Auch der Prior durch Lambda (Sentilex) wird in Beta einmodelliert
int djst::init_parameters() {
	
	// model counts
	// new_p wird zur posterior-Berechnung benutzt p(l | d) (das Label eines Dokumentes soll also estimated werden)
	new_p.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) 	{
		new_p[l].resize(numTopics); // Der Vektor wird hier lediglich "vergrößert"/erweitert
		for (int z = 0; z < numTopics; z++) {
		    new_p[l][z] = 0.0;
		}
	}

	new_nd.resize(pnewData->numDocs); 
	for (int m = 0; m < pnewData->numDocs; m++) {
	    new_nd[m] = 0;
	}

	new_ndl.resize(pnewData->numDocs); 
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
		    new_ndl[m][l] = 0;
		}
	}

	new_ndlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_ndlz[m].resize(numSentiLabs);
	    for (int l = 0; l < numSentiLabs; l++)	{
			new_ndlz[m][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++) {
			    new_ndlz[m][l][z] = 0; 
			}
		}
	}

	new_nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			new_nlzw[l][z].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++) {
			    new_nlzw[l][z][r] = 0;
			}
		}
	}

	new_nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
		    new_nlz[l][z] = 0;
		}
	}

	// model parameters
	newpi_dl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		newpi_dl[m].resize(numSentiLabs);
	}

	newtheta_dlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		newtheta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			newtheta_dlz[m][l].resize(numTopics);
		}
	}

	newphi_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		newphi_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			newphi_lzw[l][z].resize(pnewData->vocabSize);
		}
	}

	// hyperparameters
	_alpha =  (double)pnewData->aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);
	alpha_lz.resize(numSentiLabs);
	alphaSum_l.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		alphaSum_l[l] = 0.0;
		alpha_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			alpha_lz[l][z] = _alpha;
			alphaSum_l[l] += alpha_lz[l][z];
		}
	}

	// gamma
	gamma_l.resize(numSentiLabs);
	gammaSum = 0.0;
	for (int l = 0; l < numSentiLabs; l++) {
		gamma_l[l] = (double)pnewData->aveDocLength * 0.05 / (double)numSentiLabs;
		gammaSum += gamma_l[l];
	}


	//beta
	if (_beta <= 0) {
		_beta = 0.01;
	}
	beta_lzw.resize(numSentiLabs);
	betaSum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		beta_lzw[l].resize(numTopics);
		betaSum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			beta_lzw[l][z].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++) {
				beta_lzw[l][z][r] = _beta; 
				betaSum_lz[l][z] += beta_lzw[l][z][r];
			}
		} 		
	}
	
	// incorporate prior knowledge into beta
	if (sentiLexFile != "") {
		// word prior transformation matrix
		lambda_lw.resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
		  lambda_lw[l].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++)
				lambda_lw[l][r] = 1;
		}
		// MUST init beta_lzw first before incorporating prior information into beta
		this->prior2beta(); // Hier wird die prior-senti-Info aus dem Lexicon erst richtig einmodelliert und in beta_lzw bzw. betaSum_lz gesetzt
	}

	return 0;
}

// Hier werden sämtliche count-parameters für die 
// neuen Daten (wie z.B. new_nlzw=0) initialisiert. 
// Auch der Prior durch Lambda (Sentilex) wird in Beta einmodelliert
int djst::init_parameters1() {

	// model counts
	// new_p wird zur posterior-Berechnung benutzt p(l | d) (das Label eines Dokumentes soll also estimated werden)
	new_p.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_p[l].resize(numTopics); // Der Vektor wird hier lediglich "vergrößert"/erweitert
		for (int z = 0; z < numTopics; z++) {
			new_p[l][z] = 0.0;
		}
	}

	new_nd.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_nd[m] = 0;
	}

	new_ndl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			new_ndl[m][l] = 0;
		}
	}

	new_ndlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_ndlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			new_ndlz[m][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++) {
				new_ndlz[m][l][z] = 0;
			}
		}
	}

	new_nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			new_nlzw[l][z].resize(pnewData->id2_id.size());
			for (int r = 0; r < pnewData->id2_id.size(); r++) {
				new_nlzw[l][z][r] = 0;
			}
		}
	}

	new_nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			new_nlz[l][z] = 0;
		}
	}

	// model parameters
	newpi_dl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		newpi_dl[m].resize(numSentiLabs);
	}

	newtheta_dlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		newtheta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			newtheta_dlz[m][l].resize(numTopics);
		}
	}

	newphi_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		newphi_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			newphi_lzw[l][z].resize(pnewData->id2_id.size());
		}
	}

	// hyperparameters
	_alpha = (double)pnewData->aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);
	alpha_lz.resize(numSentiLabs);
	alphaSum_l.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		alphaSum_l[l] = 0.0;
		alpha_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			alpha_lz[l][z] = _alpha;
			alphaSum_l[l] += alpha_lz[l][z];
		}
	}

	// gamma
	gamma_l.resize(numSentiLabs);
	gammaSum = 0.0;
	for (int l = 0; l < numSentiLabs; l++) {
		gamma_l[l] = (double)pnewData->aveDocLength * 0.05 / (double)numSentiLabs;
		gammaSum += gamma_l[l];
	}

	if (_gamma <= 0) {
		_gamma = (double)pnewData->aveDocLength * 0.05 / (double)numSentiLabs;
	}

	// added: Wie bei model.cpp
	gamma_dl.resize(numDocs);
	gammaSum_d.resize(numDocs);
	for (int d = 0; d < numDocs; d++) {
		gamma_dl[d].resize(numSentiLabs);
		gammaSum_d[d] = 0.0;
		for (int l = 0; l < numSentiLabs; l++) {
			gamma_dl[d][l] = _gamma;
			gammaSum_d[d] += gamma_dl[d][l];
		}
	}

	//beta
	if (_beta <= 0) {
		_beta = 0.01;
	}
	beta_lzw.resize(numSentiLabs);
	betaSum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		beta_lzw[l].resize(numTopics);
		betaSum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			beta_lzw[l][z].resize(pnewData->id2_id.size());
			for (int r = 0; r < pnewData->id2_id.size(); r++) {
				beta_lzw[l][z][r] = _beta;
				betaSum_lz[l][z] += beta_lzw[l][z][r];
			}
		}
	}

	// incorporate prior knowledge into beta
	if (sentiLexFile != "") {
		// word prior transformation matrix
		lambda_lw.resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			lambda_lw[l].resize(pnewData->id2_id.size());
			for (int r = 0; r < pnewData->id2_id.size(); r++)
				lambda_lw[l][r] = 1;
		}
		// MUST init beta_lzw first before incorporating prior information into beta
		this->prior2beta2(); // Hier wird die prior-senti-Info aus dem Lexicon erst richtig einmodelliert und in beta_lzw bzw. betaSum_lz gesetzt
	}

	return 0;
}

int djst::init_parameters2() {

	// model counts
	// new_p wird zur posterior-Berechnung benutzt p(l | d) (das Label eines Dokumentes soll also estimated werden)
	new_p.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_p[l].resize(numTopics); // Der Vektor wird hier lediglich "vergrößert"/erweitert
		for (int z = 0; z < numTopics; z++) {
			new_p[l][z] = 0.0;
		}
	}

	new_nd.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_nd[m] = 0;
	}

	new_ndl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			new_ndl[m][l] = 0;
		}
	}

	new_ndlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		new_ndlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			new_ndlz[m][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++) {
				new_ndlz[m][l][z] = 0;
			}
		}
	}

	new_nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			new_nlzw[l][z].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++) {
				new_nlzw[l][z][r] = 0;
			}
		}
	}

	term_senti_frequency.resize(pnewData->vocabSize);
	for (int r = 0; r < vocabSize; r++) {
		term_senti_frequency[r] = 0;
	}

	term_senti_frequency_pos.resize(pnewData->vocabSize);
	for (int r = 0; r < vocabSize; r++) {
		term_senti_frequency_pos[r] = 0;
	}

	term_senti_frequency_neg.resize(pnewData->vocabSize);
	for (int r = 0; r < vocabSize; r++) {
		term_senti_frequency_neg[r] = 0;
	}

	new_nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		new_nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			new_nlz[l][z] = 0;
		}
	}

	// model parameters
	newpi_dl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		newpi_dl[m].resize(numSentiLabs);
	}

	newtheta_dlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++) {
		newtheta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			newtheta_dlz[m][l].resize(numTopics);
		}
	}

	newphi_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		newphi_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			newphi_lzw[l][z].resize(pnewData->vocabSize);
		}
	}

	// added
	expected_counts_lzw.resize(numSentiLabs);
	expected_counts_sum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		expected_counts_lzw[l].resize(numTopics);
		expected_counts_sum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			expected_counts_lzw[l][z].resize(pnewData->vocabSize);
		}
	}

	// added
	newsigma_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		newsigma_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			newsigma_lzw[l][z].resize(pnewData->vocabSize);
		}
	}

	// hyperparameters
	_alpha = (double)pnewData->aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);
	alpha_lz.resize(numSentiLabs);
	alphaSum_l.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		alphaSum_l[l] = 0.0;
		alpha_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			alpha_lz[l][z] = _alpha;
			alphaSum_l[l] += alpha_lz[l][z];
		}
	}

	// gamma
	gamma_l.resize(numSentiLabs);
	gammaSum = 0.0;
	for (int l = 0; l < numSentiLabs; l++) {
		gamma_l[l] = (double)pnewData->aveDocLength * 0.05 / (double)numSentiLabs;
		gammaSum += gamma_l[l];
	}

	if (_gamma <= 0) {
		_gamma = (double)pnewData->aveDocLength * 0.05 / (double)numSentiLabs;
	}

	// added: Wie bei model.cpp
	gamma_dl.resize(numDocs);
	gammaSum_d.resize(numDocs);
	for (int d = 0; d < numDocs; d++) {
		gamma_dl[d].resize(numSentiLabs);
		gammaSum_d[d] = 0.0;
		for (int l = 0; l < numSentiLabs; l++) {
			gamma_dl[d][l] = _gamma;
			gammaSum_d[d] += gamma_dl[d][l];
		}
	}

	//beta
	if (_beta <= 0) {
		_beta = 0.01;
	}
	beta_lzw.resize(numSentiLabs);
	betaSum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		beta_lzw[l].resize(numTopics);
		betaSum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			beta_lzw[l][z].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++) {
				beta_lzw[l][z][r] = _beta;
				betaSum_lz[l][z] += beta_lzw[l][z][r];
			}
		}
	}

	// added
	// Resize old phi-distributions (less words) to avoid overflow
	for (size_t i = 0; i < time_slices; i++) {
		for (int l = 0; l < numSentiLabs; l++) {
			for (int z = 0; z < numTopics; z++) {
				sliding_window_phi[i][l][z].resize(pnewData->vocabSize);
			}
		}
	}

	// incorporate prior knowledge into beta
	if (sentiLexFile != "") {
		// word prior transformation matrix
		lambda_lw.resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			lambda_lw[l].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++)
				lambda_lw[l][r] = 1;
		}
		// MUST init beta_lzw first before incorporating prior information into beta
		this->prior2beta2(); // Hier wird die prior-senti-Info aus dem Lexicon erst richtig einmodelliert und in beta_lzw bzw. betaSum_lz gesetzt
	}

	return 0;
}

void djst::compute_newpi() {
	for (int m = 0; m < pnewData->numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++) {
		    newpi_dl[m][l] = (new_ndl[m][l] + gamma_l[l]) / (new_nd[m] + gammaSum);
	    }
	}
}

// Hier nehmen wir das "neue" gamma auf
void djst::compute_newpi1() {
	for (int m = 0; m < pnewData->numDocs; m++) {
		for (int l = 0; l < numSentiLabs; l++) {
			newpi_dl[m][l] = (new_ndl[m][l] + gamma_dl[m][l]) / (new_nd[m] + gammaSum_d[m]);
		}
	}
}


void djst::compute_newtheta() {

	for (int m = 0; m < pnewData->numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++)  {
			for (int z = 0; z < numTopics; z++) {
			    newtheta_dlz[m][l][z] = (new_ndlz[m][l][z] + alpha_lz[l][z]) / (new_ndl[m][l] + alphaSum_l[l]);
			}
		}
	}
}


int djst::compute_newphi() {
	map<int, int>::iterator it;

	for (int l = 0; l < numSentiLabs; l++)  {
	    for (int z = 0; z < numTopics; z++) {
			for(int r = 0; r < pnewData->vocabSize; r++) {
			    it = _id2id.find(r); // Wir schauen an welcher Stelle das Wort in _id2id eingepflegt wurde. Dort bekommen wir über .second die Word-ID für die Trainingsdaten (um die counts nlzw abzurufen)
				if (it != _id2id.end()) {
					// Das neue Phi wird fast(!) gleich berechnet; allerdings beziehen wir auch hier die neuen Counts ein!!!
				    newphi_lzw[l][z][r] = (nlzw[l][z][it->second] + new_nlzw[l][z][r] + beta_lzw[l][z][r]) / (nlz[l][z] + new_nlz[l][z] + betaSum_lz[l][z]);
				}
				else {
				    printf("Error! Cannot find word [%d] !\n", r);
					return 1; 
				}
			}
		}
	}

	return 0;
}

void djst::compute_newphi1() {
	//vocabSize = pnewData->id2_id.size();
	vocabSize = pnewData->word2id.size();
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			for (int r = 0; r < vocabSize; r++) {
				newphi_lzw[l][z][r] = (new_nlzw[l][z][r] + beta_lzw[l][z][r]) / (new_nlz[l][z] + betaSum_lz[l][z]);
			}
		}
	}
}


int djst::save_model_newpi_dl(string filename) {

    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
    }

	for (int m = 0; m < pnewData->numDocs; m++)	{
		fprintf(fout, "d_%d %s ", m, pnewData->pdocs[m]->docID.c_str());
		for (int l = 0; l < numSentiLabs; l++) {
			fprintf(fout, "%f ", newpi_dl[m][l]);
		}
		fprintf(fout, "\n");
    }
   
    fclose(fout);       
	return 0;
}



int djst::save_model_newtheta_dlz(string filename) {

    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
    }
    
    for(int m = 0; m < pnewData->numDocs; m++) {
        fprintf(fout, "Document %d\n", m);
	    for (int l = 0; l < numSentiLabs; l++) {
	        for (int z = 0; z < numTopics; z++) {
		        fprintf(fout, "%f ", newtheta_dlz[m][l][z]);
	        }
		    fprintf(fout, "\n");
		 }
    }

    fclose(fout);
	return 0;
}


int djst::save_model_newphi_lzw(string filename) {

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }
    
	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++) {
		    fprintf(fout, "Label:%d  Topic:%d\n", l, z);
     	    for (int r = 0; r < pnewData->vocabSize; r++) {
     	    	fprintf(fout, "%.15f ", newphi_lzw[l][z][r]);
     	    }
            fprintf(fout, "\n");
	    }
    }
    
    fclose(fout);    
	return 0;
}


int djst::save_data_pyldavis(int epoch) {
	if (save_vocab_term_frequency(epoch))
		return 1;

	if (save_doc_lengths(epoch))
		return 1;

	if (save_topic_term_dists_phi(epoch))
		return 1;

	if (save_doc_topic_dists_theta(epoch))
		return 1;

	return 0;

}

int djst::save_vocab_term_frequency(int epoch) {
	string filename = result_dir + to_string(epoch) + "vocab_term_frequency.txt";
	FILE * fout = fopen(filename.c_str(), "w");
	if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
	}

	//fprintf(fout, "%s %d %.15f\n", id2word[r].c_str(), term_senti_frequency1[r], phi_lzw[l][z][r]);
	for (int r = 0; r < vocabSize; r++) {
		fprintf(fout, "%s %d\n", id2word[r].c_str(), term_senti_frequency[r]);
	}
	fclose(fout);

	filename = result_dir + to_string(epoch) + "vocab_term_frequency1.txt";
	fout = fopen(filename.c_str(), "w");
	if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
	}

	//fprintf(fout, "%s %d %.15f\n", id2word[r].c_str(), term_senti_frequency1[r], phi_lzw[l][z][r]);
	for (int r = 0; r < vocabSize; r++) {
		fprintf(fout, "%s %d\n", id2word[r].c_str(), term_senti_frequency_pos[r]);
	}
	fclose(fout);

	filename = result_dir + to_string(epoch) + "vocab_term_frequency2.txt";
	fout = fopen(filename.c_str(), "w");
	if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
	}

	//fprintf(fout, "%s %d %.15f\n", id2word[r].c_str(), term_senti_frequency1[r], phi_lzw[l][z][r]);
	for (int r = 0; r < vocabSize; r++) {
		fprintf(fout, "%s %d\n", id2word[r].c_str(), term_senti_frequency_neg[r]);
	}
	fclose(fout);

	return 0;
}

int djst::save_topic_term_dists_phi(int epoch) {
	string filename1 = result_dir + to_string(epoch) + "topic_term_dists_phi.txt";
	FILE * fout1 = fopen(filename1.c_str(), "w");
	if (!fout1) {
		printf("Cannot save file %s!\n", filename1.c_str());
		return 1;
	}

	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			//fprintf(fout1, "Label:%d  Topic:%d\n", l, z);
			for (int r = 0; r < vocabSize; r++) {
				fprintf(fout1, "%.15f ", newphi_lzw[l][z][r]);
			}
			fprintf(fout1, "\n");
		}
	}
	fclose(fout1);


	filename1 = result_dir + to_string(epoch) + "topic_term_dists_phi1.txt";
	fout1 = fopen(filename1.c_str(), "w");
	if (!fout1) {
		printf("Cannot save file %s!\n", filename1.c_str());
		return 1;
	}
	int fixed_senti = 1;
	for (int z = 0; z < numTopics; z++) {
		//fprintf(fout1, "Label:%d  Topic:%d\n", fixed_senti, z);
		for (int r = 0; r < vocabSize; r++) {
			fprintf(fout1, "%.15f ", newphi_lzw[fixed_senti][z][r]);
		}
		fprintf(fout1, "\n");
	}
	fclose(fout1);


	filename1 = result_dir + to_string(epoch) + "topic_term_dists_phi2.txt";
	fout1 = fopen(filename1.c_str(), "w");
	if (!fout1) {
		printf("Cannot save file %s!\n", filename1.c_str());
		return 1;
	}
	fixed_senti = 2;
	for (int z = 0; z < numTopics; z++) {
		//fprintf(fout1, "Label:%d  Topic:%d\n", fixed_senti, z);
		for (int r = 0; r < vocabSize; r++) {
			fprintf(fout1, "%.15f ", newphi_lzw[fixed_senti][z][r]);
		}
		fprintf(fout1, "\n");
	}
	fclose(fout1);

	return 0;
}

int djst::save_doc_lengths(int epoch) {
	string filename2 = result_dir + to_string(epoch) + "doc_lengths.txt";
	FILE * fout2 = fopen(filename2.c_str(), "w");
	if (!fout2) {
		printf("Cannot save file %s!\n", filename2.c_str());
		return 1;
	}

	for (int m = 0; m < pnewData->numDocs; m++) {
		fprintf(fout2, "%d ", pnewData->pdocs[m]->length);
	}
	fprintf(fout2, "\n");
	fclose(fout2);

	return 0;
}

int djst::save_doc_topic_dists_theta(int epoch) {
	string filename3 = result_dir + to_string(epoch)+ "doc_topic_dists_theta.txt";
	FILE * fout3 = fopen(filename3.c_str(), "w");
	if (!fout3) {
		printf("Cannot save file %s!\n", filename3.c_str());
		return 1;
	}

	for (int m = 0; m < numDocs; m++) {
		//fprintf(fout3, "Document %d\n", m);
		for (int l = 0; l < numSentiLabs; l++) {
			for (int z = 0; z < numTopics; z++) {
				fprintf(fout3, "%f ", newtheta_dlz[m][l][z]);
			}
		}
		fprintf(fout3, "\n");
	}
	fclose(fout3);

	filename3 = result_dir + to_string(epoch) + "doc_topic_dists_theta1.txt";
	fout3 = fopen(filename3.c_str(), "w");
	if (!fout3) {
		printf("Cannot save file %s!\n", filename3.c_str());
		return 1;
	}
	int fixed_senti = 1;
	for (int m = 0; m < numDocs; m++) {
		//fprintf(fout3, "Document %d\n", m);
		for (int z = 0; z < numTopics; z++) {
			fprintf(fout3, "%f ", newtheta_dlz[m][fixed_senti][z]);
		}
		fprintf(fout3, "\n");
	}

	fclose(fout3);

	filename3 = result_dir + to_string(epoch) + "doc_topic_dists_theta2.txt";
	fout3 = fopen(filename3.c_str(), "w");
	if (!fout3) {
		printf("Cannot save file %s!\n", filename3.c_str());
		return 1;
	}
	fixed_senti = 2;
	for (int m = 0; m < numDocs; m++) {
		//fprintf(fout3, "Document %d\n", m);
		for (int z = 0; z < numTopics; z++) {
			fprintf(fout3, "%f ", newtheta_dlz[m][fixed_senti][z]);
		}
		fprintf(fout3, "\n");
	}

	fclose(fout3);

	return 0;
}


int djst::save_model_newothers(string filename) {
	
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }
    
	fprintf(fout, "model_dir=%s\n", model_dir.c_str());
	fprintf(fout, "model_name=%s\n", model_name.c_str());
	fprintf(fout, "data_dir=%s\n", data_dir.c_str());
	fprintf(fout, "datasetFile=%s\n", datasetFile.c_str());
	fprintf(fout, "result_dir=%s\n", result_dir.c_str());
	fprintf(fout, "niters-djst=%d\n", niters);
	fprintf(fout, "savestep-djst=%d\n", savestep);

	fprintf(fout, "\n------------------ Testset ** %s ** statistics ----------------------\n", datasetFile.c_str());
    fprintf(fout, "newNumDocs=%d\n", pnewData->numDocs);
    fprintf(fout, "newCorpusSize=%d\n", pnewData->corpusSize);
    fprintf(fout, "newVocabSize=%d\n", pnewData->vocabSize);
	fprintf(fout, "numNewWords=%d\n", (int)(pnewData->newWords.size()));
	fprintf(fout, "aveDocLength=%d\n", pnewData->aveDocLength);
	fprintf(fout, "\n------------------ Loaded model settings ----------------------\n");
	fprintf(fout, "numSentiLabs=%d\n", numSentiLabs);
	fprintf(fout, "numTopics=%d\n", numTopics);
	fprintf(fout, "numDocs=%d\n", pmodelData->numDocs);
	fprintf(fout, "corpusSize=%d\n", pmodelData->corpusSize);
	fprintf(fout, "vocabSize=%d\n", pmodelData->vocabSize);

	fclose(fout);
	return 0;
}

int djst::save_model_newothers1(string filename) {

	FILE * fout = fopen(filename.c_str(), "w");
	if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
	}

	fprintf(fout, "model_dir=%s\n", model_dir.c_str());
	fprintf(fout, "model_name=%s\n", model_name.c_str());
	fprintf(fout, "data_dir=%s\n", data_dir.c_str());
	fprintf(fout, "datasetFile=%s\n", datasetFile.c_str());
	fprintf(fout, "result_dir=%s\n", result_dir.c_str());
	fprintf(fout, "niters-djst=%d\n", niters);
	fprintf(fout, "savestep-djst=%d\n", savestep);

	fprintf(fout, "\n------------------ Testset ** %s ** statistics ----------------------\n", datasetFile.c_str());
	fprintf(fout, "numDocs=%d\n", pnewData->numDocs);
	fprintf(fout, "newCorpusSize=%d\n", pnewData->corpusSize);
	fprintf(fout, "newVocabSize=%d\n", pnewData->vocabSize);
	fprintf(fout, "numNewWords=%d\n", (int)(pnewData->newWords.size()));
	fprintf(fout, "aveDocLength=%d\n", pnewData->aveDocLength);
	fprintf(fout, "numSentiLabs=%d\n", numSentiLabs);
	fprintf(fout, "numTopics=%d\n", numTopics);

	fclose(fout);
	return 0;
}


// Beachte: In dieser Methode werden nur die Worte aus den neuen Daten rausgeschrieben, welche bereits aus den Trainingsdaten bekannt waren.
// Für alle anderen Worte (andere Worte aus den Trainingsdaten & neue, unbekannte Worte aus den Testdaten) wird nichts eingetragen
int djst::save_model_newtwords(string filename) {

	mapid2word::iterator it; // typedef map<int, string> mapid2word
	map<int, int>::iterator _it;

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }
    
    if (twords > pnewData->vocabSize) {
	    twords = pnewData->vocabSize;
    }
   
    for (int l = 0; l < numSentiLabs; l++) {
        fprintf(fout, "Label %dth\n", l);
        for (int k = 0; k < numTopics; k++) {
	        vector<pair<int, double> > words_probs;
	        pair<int, double> word_prob;
	        for (int w = 0; w < pnewData->vocabSize; w++) {
		        word_prob.first = w;
	            word_prob.second = newphi_lzw[l][k][w];
	            words_probs.push_back(word_prob);
	        }
    
		    std::sort(words_probs.begin(), words_probs.end(), sort_pred());

	        fprintf(fout, "Topic %dth:\n", k);
	        for (int i = 0; i < twords; i++) {
				_it = _id2id.find(words_probs[i].first);
				if (_it == _id2id.end()) {
		            continue;
	            }
				it = id2word.find(_it->second);
	            if (it != id2word.end()) {
			        fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
	            } 
	        }
	    } // for topic
    } // for label
     
    fclose(fout);      
	return 0;
}

int djst::save_model_newtwords1(string filename) {

	mapid2word::iterator it; // typedef map<int, string> mapid2word
	map<int, int>::iterator _it;

	FILE * fout = fopen(filename.c_str(), "w");
	if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
	}

	if (twords > pnewData->vocabSize) {
		twords = pnewData->vocabSize;
	}

	for (int l = 0; l < numSentiLabs; l++) {
		for (int k = 0; k < numTopics; k++) {
			vector<pair<int, double> > words_probs;
			pair<int, double> word_prob;
			for (int w = 0; w < pnewData->vocabSize; w++) {
				word_prob.first = w;
				word_prob.second = newphi_lzw[l][k][w];
				words_probs.push_back(word_prob);
			}

			std::sort(words_probs.begin(), words_probs.end(), sort_pred());

			fprintf(fout, "Label%d_Topic%d\n", l, k);
			for (int i = 0; i < twords; i++) {
				it = id2word.find(words_probs[i].first);
				if (it != id2word.end())
					fprintf(fout, "%s   %15f\n", (it->second).c_str(), words_probs[i].second);
			}
		} // for topic
	} // for label

	fclose(fout);
	return 0;
}

// Diese Methode speichert die neuen Zuweisungen.
// Beachte: Dabei werden nur für die bereits aus den Trainingsdaten bekannte Worte assignments gemacht.
// Dennoch kann man gerade daran auch sehen, ob sich eventuell (alte/bekannte) Worte aufgrund der (neuen) Daten in ihren Labels verändert haben
int djst::save_model_newtassign(string filename) {

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }

	for (int m = 0; m < pnewData->numDocs; m++) {
		fprintf(fout, "%s \n", pnewData->pdocs[m]->docID.c_str());
		for (int n = 0; n < pnewData->pdocs[m]->length; n++) {
	        fprintf(fout, "%d:%d:%d ", pnewData->pdocs[m]->words[n], new_l[m][n], new_z[m][n]); //  wordID:sentiLab:topic
	    }
	    fprintf(fout, "\n");
    }

    fclose(fout);
	return 0;
}



int djst::prior2beta() {
	mapword2atr::iterator wordIt;
	mapword2prior::iterator sentiIt;

	for (sentiIt = sentiLex.begin(); sentiIt != sentiLex.end(); sentiIt++) {
		wordIt = word2atr.find(sentiIt->first);
		if (wordIt != word2atr.end()) {
			for (int j = 0; j < numSentiLabs; j++)  {
				lambda_lw[j][wordIt->second.id] = sentiIt->second.labDist[j];
			}
		}
	}

	// Note: the 'r' index of lambda[j][r] is corresponding to the vocabulary ID.
	// Therefore the correct prior info can be incorporated to corresponding word count nlzw,
	// as 'w' is also corresponding to the vocabulary ID.
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			betaSum_lz[l][z] = 0.0;
		    for (int r = 0; r < pnewData->vocabSize; r++) {
			    beta_lzw[l][z][r] = beta_lzw[l][z][r] * lambda_lw[l][r];
			    betaSum_lz[l][z] += beta_lzw[l][z][r];
		    }
		}
	}

	return 0;
}

// Berechnet globalen Beta-Vektor anhand der dJST Vorschrift (alte Verteilungen werden mit einbezogen)
int djst::prior2beta2() {
	mapword2atr::iterator wordIt;
	mapword2prior::iterator sentiIt;

	// TODO Performance: diese schleife outsourcen und lambda_lw global setzen. So berechnen wir es nicht jedes mal
	for (sentiIt = sentiLex.begin(); sentiIt != sentiLex.end(); sentiIt++) {
		wordIt = word2atr.find(sentiIt->first);
		if (wordIt != word2atr.end()) {
			for (int j = 0; j < numSentiLabs; j++) {
				lambda_lw[j][wordIt->second.id] = sentiIt->second.labDist[j];
			}
		}
	}

	// Note: the 'r' index of lambda[j][r] is corresponding to the vocabulary ID.
	// Therefore the correct prior info can be incorporated to corresponding word count nlzw,
	// as 'w' is also corresponding to the vocabulary ID.
	map<int, int>::iterator idIt;
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			// TODO was passiert wenn folgende Zeile auskommentiert wird?
			betaSum_lz[l][z] = 0.0;
			for (int r = 0; r < pnewData->vocabSize; r++) {
				// sentiIt = sentiLex.find(id2word[r]); // check wether word r is sentiment bearing (in mpqa)
				if (std::find(pnewData->newWords1.begin(), pnewData->newWords1.end(), r) != pnewData->newWords1.end() /*|| sentiIt != sentiLex.end()*/) { // check if glob. word r was new
					beta_lzw[l][z][r] = beta_lzw[l][z][r] * lambda_lw[l][r]; // if the word is new we only incorporate standard prior (lambda) information from mpqa
				}
				else { // word isn't new and also no sentiment-bearing word
					// Rechenvorschrift für dJST (Update-Regel Sliding Window)
					for (size_t i = 0; i < sliding_window_phi.size(); i++) {
						beta_lzw[l][z][r] += sliding_window_phi[i][l][z][r] * window_weights[i];
					}
				}
				betaSum_lz[l][z] += beta_lzw[l][z][r];
			}
		}
	}

	return 0;
}

int djst::update_Parameters() {

	int ** data; // temp valuable for exporting 3-dimentional array to 2-dimentional
	double * alpha_temp;
	data = new int*[numTopics];
	for (int k = 0; k < numTopics; k++) {
		data[k] = new int[numDocs];
		for (int m = 0; m < numDocs; m++) {
			data[k][m] = 0;
		}
	}

	alpha_temp = new double[numTopics];
	for (int k = 0; k < numTopics; k++) {
		alpha_temp[k] = 0.0;
	}

	// update alpha
	for (int j = 0; j < numSentiLabs; j++) {
		for (int k = 0; k < numTopics; k++) {
			for (int m = 0; m < numDocs; m++) {
				data[k][m] = new_ndlz[m][j][k]; // ntldsum[j][k][m];
			}
		}

		for (int k = 0; k < numTopics; k++) {
			alpha_temp[k] = alpha_lz[j][k]; //alpha[j][k];
		}

		polya_fit_simple(data, alpha_temp, numTopics, numDocs);

		// update alpha
		alphaSum_l[j] = 0.0;
		for (int k = 0; k < numTopics; k++) {
			alpha_lz[j][k] = alpha_temp[k];
			alphaSum_l[j] += alpha_lz[j][k];
		}
	}

	return 0;
}

int djst::save_model(string model_name, int epoch) {

	if (save_model_newtassign(result_dir + std::to_string(epoch) + model_name + tassign_suffix))
		return 1;

	if (save_model_newtwords1(result_dir + std::to_string(epoch) + model_name + twords_suffix))
		return 1;

	if (save_model_newpi_dl(result_dir + std::to_string(epoch) + model_name + pi_suffix))
		return 1;

	if (save_model_newtheta_dlz(result_dir + std::to_string(epoch) + model_name + theta_suffix))
		return 1;

	if (save_model_newphi_lzw(result_dir + std::to_string(epoch) + model_name + phi_suffix))
		return 1;

	if (save_model_newothers1(result_dir + std::to_string(epoch) + model_name + others_suffix))
		return 1;

	return 0;
}
