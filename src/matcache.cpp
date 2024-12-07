#include <string>
#include <tuple>
#include <sstream>
#include "matcache.h"
#include "rand.h"
#include "matmul.h"
#include "precision.h"

// Generate a filename based on matmul dimensions, generation schema, precision and operand (A, B, or C)
std::string generateFilename(size_t M, size_t K, size_t N, char operand, const std::string& schema, const std::string& precision) {
    std::ostringstream oss;
    oss << "matcache/matrix_" << operand << "_" << M << "x" << K << "x" << N << "_" << precision << "_" << schema << ".bin";
    return oss.str();
}

// Store a matrix in a binary file
template<class T>
int storeMatrix(T* matrix, size_t M, size_t K, size_t N, char operand, const std::string& schema) {
    std::string precision = std::is_same<T, float>::value ? "float" : "double";
    std::string filename = generateFilename(M, K, N, operand, schema, precision);
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        perror("MatCache: Failed to open file for writing");
        return -1;
    }

    unsigned long int nrElements = (operand == 'A') ? M*K :
                     (operand == 'B') ? K*N : M*N; // default is C

    size_t elementsWritten = fwrite(matrix, sizeof(T), nrElements, file);
    fclose(file);

    return elementsWritten == nrElements ? 0 : -1;
}

bool fileExists(const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "rb");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

const std::string typeToSchema(int input_type) {
    switch (input_type) {
        default:
            return "unknown";
        case 0:
            return "uniform";
        case 1:
            return "ootomoType1";
        case 2:
            return "ootomoType2";
        case 3:
            return "ootomoType3";
        case 4:
            return "ootomoType4";
    }
}

// Loads matrices if they exist, else computes and stores them
template<class T>
std::tuple<T*, T*, T*> getMatrices(size_t M, size_t K, size_t N, int input_type, struct LCG *rng) {
    const std::string schema = typeToSchema(input_type);
    if (schema == "unknown") {
        perror("MatCache: Unknown input type!");
        return {nullptr, nullptr, nullptr};
    }

    std::string precision = std::is_same<T, float>::value ? "float" : "double";
    std::string filenameA = generateFilename(M, K, N, 'A', schema, precision);
    std::string filenameB = generateFilename(M, K, N, 'B', schema, precision);
    std::string filenameC = generateFilename(M, K, N, 'C', schema, precision);

    bool existsA = fileExists(filenameA);
    bool existsB = fileExists(filenameB);
    bool existsC = fileExists(filenameC);

    if (existsA && existsB && existsC) {
        // All matrices exist; load them
        //printf("Matrices already exist; loading them\n");
        T* A = (T*)malloc(M * K * sizeof(T));
        T* B = (T*)malloc(K * N * sizeof(T));
        T* C = (T*)malloc(M * N * sizeof(T));

        FILE* fileA = fopen(filenameA.c_str(), "rb");
        FILE* fileB = fopen(filenameB.c_str(), "rb");
        FILE* fileC = fopen(filenameC.c_str(), "rb");

        size_t nrElementsA = fread(A, sizeof(T), M * K, fileA);
        size_t nrElementsB = fread(B, sizeof(T), K * N, fileB);
        size_t nrElementsC = fread(C, sizeof(T), M * N, fileC);

        fclose(fileA);
        fclose(fileB);
        fclose(fileC);

        if (nrElementsA == M * K && nrElementsB == K * N && nrElementsC == M * N) {
            return {A, B, C};
        } else {
            perror("MatCache: Failed to read matrices, regenerating them");
            free(A);
            free(B);
            free(C);
        }
    }

    if (existsA || existsB || existsC) {
        // Partial matrices exist; log an error
        perror("MatCache: Only some of the matrices exist. This should not happen.\n");
        return {nullptr, nullptr, nullptr};
    }

    // None of the matrices exist; generate them
    T* A = (T*)malloc(M * K * sizeof(T));
    T* B = (T*)malloc(K * N * sizeof(T));
    T* C = (T*)malloc(M * N * sizeof(T));

    if (!A || !B || !C) {
        perror("MatCache: Memory allocation failed");
        free(A);
        free(B);
        free(C);
    }

    //printf("Matrices do not exist; generating them\n");
    fill_matrices(rng, input_type, A, B, M*K, K*N);
    referenceMatmul_full<T>(A, B, C, M, K, N);

    // Store the matrices
    storeMatrix(A, M, K, N, 'A', schema);
    storeMatrix(B, M, K, N, 'B', schema);
    storeMatrix(C, M, K, N, 'C', schema);

    return {A, B, C};
}


template std::tuple<float*, float*, float*> getMatrices<float>(size_t, size_t, size_t, int, struct LCG*);
template std::tuple<double*, double*, double*> getMatrices<double>(size_t, size_t, size_t, int, struct LCG*);
