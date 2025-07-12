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
        void write(int x, int y, float Value);
        void fill(float Value);
        void print() const;
        void init() { fill(0.0f); }
        
        Matrix dot(const Matrix& other) const;

        int getRows() const { return rows; }
        int getCols() const { return cols; }
};