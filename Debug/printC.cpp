#include <iostream>
#include <fstream>

void writeMatrixToFileInt(const std::string& filename, int *matrix, int nodes, int colum) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < colum; ++j) {
            file << matrix[i*colum+j];
            
            // If it's not the last column, add a space
            if (j != colum - 1) {
                file << " ";
            }
        }
        file << std::endl;  // New line for each row
    }

    file.close();
}