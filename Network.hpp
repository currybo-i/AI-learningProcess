#include <iostream>
#include <vector>

using namespace std;

class Network 
{
    private:
        int _numHlayers;
        int _numHneurons;
        int _numIneurons;
        int _numOneurons;

        vector<float> HHweights; // it will store every hh weights, that means even if there are multiple hh layer connections
        vector<float> IHweights;
        vector<float> HOweights;
    public:
        Network(int numHlayers, int numHneurons, int numIneurons, int numOneurons);

        void init();
        void save() const;
        void load();
        void print() const;

        int getNumHiddenLayers() const { return _numHlayers; }
        int getNumHiddenNeurons() const { return _numHneurons; }
        int getNumInputNeurons() const { return _numIneurons; }
        int getNumOutputNeurons() const { return _numOneurons; }
};