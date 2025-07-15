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
        void write(int x, int y, float Value);
        void fill(float Value); 
        void print() const;
        void init(int r, int c);
        
        Matrix dot(const Matrix& other) const;

        int getRowsSize() const { return rows; }
        int getColsSize() const { return cols; }
        int getSize() const { return rows * cols; }
        const vector<vector<float>>& getData() const { return data; }
};