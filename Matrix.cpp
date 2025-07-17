#include "Matrix.hpp"

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r, vector<float>(c, 0.0f)) {}

void Matrix::fill(float Value) 
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = Value;
        }
    }
}

void Matrix::write(int x, int y, float Value) 
{
    if (x >= 0 && x < rows && y >= 0 && y < cols) {
        data[x][y] = Value;
    } else {
        cerr << "Index out of bounds" << endl;
    }
}

void Matrix::print() const 
{
    for (const auto& row : data) {
        for (float value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
}

Matrix Matrix::dot(Matrix other)
{
    if (cols != other.rows) {
        cerr << "Matrix dimensions do not match for dot product" << endl;
        return Matrix(0, 0);
    }
    
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols; ++k) {
                sum += data[i][k] * other.data[k][j];
            }
            result.write(i, j, sum);
        }
    }
    return result;
}

void Matrix::init(int r, int c) 
{
    rows = r;
    cols = c;
    data.resize(rows);
    for (auto& row : data) {
        row.resize(cols, 0.0f);
    }
    fill(0.0f);
}

void Matrix::resize(int r, int c) 
{
    rows = r;
    cols = c;
    data.resize(rows);
    for (auto& row : data) {
        row.resize(cols, 0.0f);
    }
}