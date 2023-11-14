#include <iostream>
#include <fstream>

void writeMatrixToFile(const std::string& filename, double *matrix, int nodes, int col) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < col; ++j) {
            file << matrix[i*col+j];
            
            // If it's not the last column, add a space
            if (j != col - 1) {
                file << "\t\t\t\t\t ";
            }
        }
        file << std::endl;  // New line for each row
    }

    file.close();
}