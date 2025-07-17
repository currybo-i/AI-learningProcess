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
    HHweights(numHneurons, static_cast<int>(pow(numHneurons, numHlayers - 1))),
    HOweights(numHneurons, numOneurons)
{}

void NeuralNet::init() {
    //Intialises every weights to some randomvalue between -1000 and 1000
    mt19937 gen(random_device{}());
    uniform_real_distribution<> dis(-1000, 1000);
    int rc = 0;
    for (auto& r : IHweights.getData())
    {
        int cc = 0;
        for (const float& w : r)
        {
            IHweights.write(rc, cc, dis(gen));
            cc++;
        }
        rc++;
    }
    rc = 0;
    for (auto& r : HHweights.getData())
    {
        int cc = 0;
        for (const float& w : r)
        {
            HHweights.write(rc, cc, dis(gen));
            cc++;
        }
        rc++;
    }
    rc = 0;
    for (auto& r : HOweights.getData())
    {
        int cc = 0;
        for (const float& w : r)
        {
            HOweights.write(rc, cc, dis(gen));
            cc++;
        }
        rc++;
    }
}

void NeuralNet::save(string savePath) const {
    //saves the current weights ; if the savePath doesn't contain the file, it will create one at that position

    ofstream file(savePath);

    if (!file.is_open())
    {
        cerr << "Didn't open";
        return;
    }
    for (const auto& r : IHweights.getData())
    {
        for (const float& w : r)
        {
            file << w << " ";
        }
        file << endl;
    }
    for (const auto& r : HHweights.getData())
    {        
        for (const float& w : r)
        {            
            file << w << " ";
        }
        file << endl;
    }
    for (const auto& r : HOweights.getData())
    {
        for (const float& w : r)
        {
            file << w << " ";
        }
        file << endl;
    }
    file.close();
}

void NeuralNet::load(string loadPath) {
    //loads weight from the given loadPath

    ifstream file(loadPath);
    
    if (!file.is_open()) {
        cerr << "Didn't open";
    }
    int rc = 0;
    for (auto& r : IHweights.getData()) {
        int cc = 0;
        for (const float& w : r) {
            IHweights.write(rc, cc, 0.0f);
            cc++;
        }
        rc++;
    }
    rc = 0;
    for (auto& r : HHweights.getData()) {
        int cc = 0;
        for (const float& w : r) {
            HHweights.write(rc, cc, 0.0f);
            cc++;
        }
        rc++;
    }
    rc = 0;
    for (auto& r : HOweights.getData()) {
        int cc = 0;
        for (const float& w : r) {
            HOweights.write(rc, cc, 0.0f);
            cc++;
        }
        rc++;
    }
    file.close();
}

vector<float> softmax (vector<float> x) {
    //seriously have no idea what it does, but it seems a lot of ppl are telling me to use this, so

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

vector<float> NeuralNet::forwardpass(vector<float> input) {
    /*This is the part where with the given input it generates a probablity of
    different Outputs in a vector type, on the basis of the accuracy of the 
    weights and how much the NeuralNet has been trained on it*/

    if (input.size() != _numIneurons) {
        cerr << "Input size does not match number of input neurons" << endl;
        return {};
    }

    IHweights.resize(1, _numIneurons);
    for (int i = 0; i < _numIneurons; ++i)
        IHweights.write(0, i, input[i]);
    Ilayer = input;

    /*
    Matrix HHMatrix(1, _numHneurons);
    for (int i = 0; i < _numHlayers; ++i) {
        if (i == 0) {
            HHMatrix = IHMatrix.dot(IHweights);
        } else {
            Matrix prev(1, _numHneurons);
            for (int j = 0; j < _numHneurons; ++j)
                prev.write(0, j, Hlayers[i - 1][j]);
            HHMatrix = prev.dot(HHweights);
        }

        for (int r = 0; r < HHMatrix.getRowsSize(); ++r)
            for (int c = 0; c < HHMatrix.getColsSize(); ++c)
                HHMatrix.write(r, c, max(0.0f, HHMatrix.getData()[r][c]));

        Hlayers[i].clear();
        for (int j = 0; j < _numHneurons; ++j)
            Hlayers[i].push_back(HHMatrix.getData()[0][j]);
    }*/
    HHweights.resize(_numHlayers -1,_numHneurons);
    Hlayers = HHMatrix.getData();


    Matrix lastH(1, _numHneurons);
    for (int i = 0; i < _numHneurons; ++i)
        lastH.write(0, i, Hlayers[_numHlayers - 1][i]);
    
    HOweights = lastH.dot(HOweights);
    vector<float> raw = HOweights.getData()[0];
    Olayer = softmax(raw);
    
    return Olayer;
}

void NeuralNet::BackwardPropagation(vector<float> target, float learningRate) {

    if (target.size() != _numOneurons) {
        cerr << "Target size Does not match the number of output neurons" << endl;
    }


}

