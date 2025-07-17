#include <vector>
#include <iostream>

using namespace std;

class Matrix {
    private:
        int rows;
        int cols;
        vector<vector<float>> data;
    public:
        
        Matrix(int r, int c);
        void resize(int r, int c);
        void write(int r, int c, float Value);
        void fill(float Value); 
        void print() const;
        float read(int r, int c) const;
        void init(int r, int c);
        
        Matrix minus(const Matrix& other);
        Matrix add(const Matrix& other);
        Matrix fromVect(vector<float> vec);
        Matrix dot(Matrix other);

        vector<float> toVect();

        int getRowsSize() const { return rows; }
        int getColsSize() const { return cols; }
        int getSize() const { return rows * cols; }
        const vector<vector<float>>& getData() const { return data; }

};