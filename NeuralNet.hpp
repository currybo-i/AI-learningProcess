#include <iostream>
#include <vector>
#include "Matrix.hpp"

using namespace std;

class NeuralNet 
{
    private:
        int _numHlayers;
        int _numHneurons;
        int _numIneurons;
        int _numOneurons;

        vector<Matrix> HHweights; // it will store every hh weights, that means even if there are multiple hh layer connections
        Matrix IHweights;
        Matrix HOweights;

        vector<float> Ilayer;
        vector<vector<float>> Hlayers;
        vector<float> Olayer;
    public:
        NeuralNet(int numHlayers, int numHneurons, int numIneurons, int numOneurons);

        void init();
        void save(string savePath) const;
        void load(string loadPath);
        vector<float> forwardpass(vector<float> input);
        void BackPropagation(vector<float> target, float learningRate);
        void print() const;

        int getNumHiddenLayers() const { return _numHlayers; }
        int getNumHiddenNeurons() const { return _numHneurons; }
        int getNumInputNeurons() const { return _numIneurons; }
        int getNumOutputNeurons() const { return _numOneurons; }
};