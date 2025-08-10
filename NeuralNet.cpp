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

void NeuralNet::BackPropagation(const vector<float> t, float lr) {
    if (t.size() != Olayer.size()) {
        cerr << "Target size does not match output layer size" << endl;
        return;
    }

    // error = (dL/dz for softmax + cross-entropy)
    vector<float> output_error(Olayer.size());
    for (size_t i = 0; i < Olayer.size(); ++i) {
        output_error[i] = Olayer[i] - t[i];
    }

    // Update HOweights
    Matrix finalHidden(1, Hlayers.back().size());
    finalHidden = finalHidden.fromVect(Hlayers.back());
    Matrix output_error_mat(1, output_error.size());
    output_error_mat = output_error_mat.fromVect(output_error);

    Matrix dHO = finalHidden.transpose().dot(output_error_mat);
    for (int r = 0; r < HOweights.getRowsSize(); ++r) {
        for (int c = 0; c < HOweights.getColsSize(); ++c) {
            float updated = HOweights.read(r, c) - lr * dHO.read(r, c);
            HOweights.write(r, c, updated);
        }
    }

    
    vector<vector<float>> deltas(_numHlayers);
    vector<float> next_delta = output_error;
    for (int l = _numHlayers - 1; l >= 0; --l) {
        // derivate of activation (ReLU)
        vector<float> deriv(Hlayers[l].size());
        for (size_t i = 0; i < Hlayers[l].size(); ++i)
            deriv[i] = Hlayers[l][i] > 0 ? 1.0f : 0.0f;
        vector<float> delta(Hlayers[l].size(), 0.0f);
        if (l == _numHlayers - 1) {
            // for OH weights
            for (int i = 0; i < HOweights.getRowsSize(); ++i) {
                float sum = 0.0f;
                for (int j = 0; j < HOweights.getColsSize(); ++j)
                    sum += HOweights.read(i, j) * next_delta[j];
                delta[i] = sum * deriv[i];
            }
        } else {
            // for HH weights
            for (int i = 0; i < HHweights[l+1].getRowsSize(); ++i) {
                float sum = 0.0f;
                for (int j = 0; j < HHweights[l+1].getColsSize(); ++j)
                    sum += HHweights[l+1].read(i, j) * next_delta[j];
                delta[i] = sum * deriv[i];
            }
        }
        deltas[l] = delta;
        next_delta = delta;
    }

    // Update HHweights
    for (int l = 0; l < _numHlayers; ++l) {
        Matrix prev(1, _numIneurons);
        if (l == 0) {
            prev = Matrix(1, Ilayer.size()).fromVect(Ilayer);
        } else {
            prev = Matrix(1, Hlayers[l-1].size()).fromVect(Hlayers[l-1]);
        }
        Matrix delta_mat(1, deltas[l].size());
        delta_mat = delta_mat.fromVect(deltas[l]);
        Matrix dW = prev.transpose().dot(delta_mat);
        for (int r = 0; r < HHweights[l].getRowsSize(); ++r) {
            for (int c = 0; c < HHweights[l].getColsSize(); ++c) {
                float updated = HHweights[l].read(r, c) - lr * dW.read(r, c);
                HHweights[l].write(r, c, updated);
            }
        }
    }

    // Update IHweights
    Matrix inputMat(1, Ilayer.size());
    inputMat = inputMat.fromVect(Ilayer);
    Matrix delta0_mat(1, deltas[0].size());
    delta0_mat = delta0_mat.fromVect(deltas[0]);
    Matrix dIH = inputMat.transpose().dot(delta0_mat);
    for (int r = 0; r < IHweights.getRowsSize(); ++r) {
        for (int c = 0; c < IHweights.getColsSize(); ++c) {
            float updated = IHweights.read(r, c) - lr * dIH.read(r, c);
            IHweights.write(r, c, updated);
        }
    }
}