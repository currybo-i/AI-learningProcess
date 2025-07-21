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

void Matrix::write(int r, int c, float Value) 
{
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
        data[r][c] = Value;
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

Matrix Matrix::fromVect(vector<float> vec) {
    Matrix result(1,vec.size());
    int cc = 0;
    for (float& v : vec) {
        result.write(1, cc, v);
        ++cc;
    }
    return result;
}

vector<float> Matrix::toVect() {
    vector<float> result;
    result = data[0];
    return result;
}

float Matrix::read(int r, int c) const {
    return data[r][c];
}

Matrix Matrix::add(const Matrix& other) {
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.write(r, c, read(r, c) + other.read(r, c));
        }
    }
    return result;
}

Matrix Matrix::minus(const Matrix& other) {
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.write(r, c, read(r, c) - other.read(r, c));
        }
    }
    return result;
}

Matrix Matrix::transpose() const{
    Matrix result(cols, rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.write(r, c, data[c][r]);
        }
    }
    return result;
}

Matrix Matrix::scale(float x) const{
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.write(r, c, data[c][r]*x);
        }
    }
    return result;
}

Matrix Matrix::product(Matrix& other){
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            result.write(i,j,data[i][j]*other.data[i][j]);
        }
    }
    return result;
}