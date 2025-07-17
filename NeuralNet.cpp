#include "NeuralNet.hpp"
#include <random>
#include "Matrix.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

NeuralNet::NeuralNet(int numHlayers, int numHneurons, int numIneurons, int numOneurons) :
    _numHlayers(numHlayers),
    _numHneurons(numHneurons),
    _numIneurons(numIneurons),
    _numOneurons(numOneurons),
    IHweights(numIneurons, numHneurons),
    HOweights(numHneurons, numOneurons)
{}

void NeuralNet::init() {
    mt19937 gen(random_device{}());
    uniform_real_distribution<> dis(-1000, 1000);

    for (int r = 0; r < IHweights.getRowsSize(); ++r) {
        for (int c = 0; c < IHweights.getColsSize(); ++c) {
            IHweights.write(r, c, dis(gen));
        }
    }

    for (Matrix& hh_mat : HHweights) {
        for (int r = 0; r < hh_mat.getRowsSize(); ++r) {
            for (int c = 0; c < hh_mat.getColsSize(); ++c) {
                hh_mat.write(r, c, dis(gen));
            }
        }
    }

    for (int r = 0; r < HOweights.getRowsSize(); ++r) {
        for (int c = 0; c < HOweights.getColsSize(); ++c) {
            HOweights.write(r, c, dis(gen));
        }
    }
}

void NeuralNet::save(string savePath) const {
    ofstream file(savePath);

    if (!file.is_open())
    {
        cerr << "Didn't open";
        return;
    }

    for (int r = 0; r < IHweights.getRowsSize(); ++r) {
        for (int c = 0; c < IHweights.getColsSize(); ++c) {
            file << IHweights.read(r, c) << " ";
        }
        file << endl; 
    }

    for (const Matrix& hh_mat : HHweights) {
        for (int r = 0; r < hh_mat.getRowsSize(); ++r) {
            for (int c = 0; c < hh_mat.getColsSize(); ++c) {
                file << hh_mat.read(r, c) << " ";
            }
            file << endl;
        }
    }

    for (int r = 0; r < HOweights.getRowsSize(); ++r) {
        for (int c = 0; c < HOweights.getColsSize(); ++c) {
            file << HOweights.read(r, c) << " ";
        }
        file << endl;
    }

    file.close();
}

void NeuralNet::load(string loadPath) {
    ifstream file(loadPath);
    
    if (!file.is_open())
    {
        cerr << "Didn't open";
    }

    float val;
    string curLine;
    for (int r = 0; r < IHweights.getRowsSize(); ++r) {
        getline(file, curLine);
        for (int c = 0; c < IHweights.getColsSize(); ++c) {
            val = curLine[2*r];
            IHweights.write(r, c, val);
        }
    }

    for (Matrix& hh_mat : HHweights) {
        for (int r = 0; r < hh_mat.getRowsSize(); ++r) {
            getline(file, curLine);
            for (int c = 0; c < hh_mat.getColsSize(); ++c) {
                val = curLine[2*r];
                hh_mat.write(r, c, val);
            }
        }
    }

    for (int r = 0; r < HOweights.getRowsSize(); ++r) {
        getline(file, curLine);
        for (int c = 0; c < HOweights.getColsSize(); ++c) {
            val = curLine[2*r];
            HOweights.write(r, c, val);
        }
    }
}

vector<float> softmax (vector<float> x) {
    float max_x = *max_element(x.begin(), x.end());
    float sum = 0.0f;
    vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = exp(x[i] - max_x);
        sum += result[i];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }
    return result;
}

vector<float> relu(vector<float>& const v) {
    vector<float> result;
    for (int i = 0; i < v.size(); ++i)
        result[i] = max(0.0f, v[i]);
    return result;
}

vector<float> NeuralNet::forwardpass(vector<float> input) {
    if (input.size() != _numIneurons)
    {
        cerr << "Input size does not match number of input neurons" << endl;
        return;
    }

    Ilayer = input;
    Matrix inpMat(1,input.size());
    inpMat.fromVect(input);

    for (int i = 0; i < HHweights.size(); ++i) {
        if (i == 0) {
            Hlayers[i] = softmax((IHweights.dot(inpMat)).toVect());
        }
        else if (i != _numHlayers) {
            Matrix prevMat(1, Hlayers[i-1].size());
            prevMat = prevMat.fromVect(Hlayers[i-1]);

            Matrix outMat = HHweights[i].dot(prevMat);
            vector<float> Hraw = outMat.toVect();
            Hlayers[i] = relu(Hraw);
        }
    }

    Matrix finalHidden(1, Hlayers.back().size());
    finalHidden = finalHidden.fromVect(Hlayers.back());

    Matrix outRaw = HOweights.dot(finalHidden);
    Olayer = softmax(outRaw.toVect());

    return Olayer;
}
