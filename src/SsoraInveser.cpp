#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include <iostream>
#include <vector>

typedef double Number;

int main() {
    SparseMatrix<Number> mat = SparseMatrix<Number>::triDiagonal(1, -1, 2, -1);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";

    mat = SparseMatrix<Number>::triDiagonal(2, -1, 2, -1);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";
    
    mat = SparseMatrix<Number>::triDiagonal(3, -1, 2, -1);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";

    mat = std::move(SparseMatrix<Number>::triDiagonal(6, -1, 2, -1));
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";

    std::unique_ptr<Number[]> entries = std::unique_ptr<Number[]>(new Number[64]);
    std::unique_ptr<int[]> cols = std::unique_ptr<int[]>(new int[64]);
    std::unique_ptr<int[]> rowPtrs = std::unique_ptr<int[]>(new int[9]);

    for (int i = 0; i < 64; i++) {
	cols[i] = i % 8;
    }

    for (int i = 0; i < 9; i++) {
	rowPtrs[i] = i * 8;
    }
    
    entries[0] = 9;
    entries[1] = 3;
    entries[2] = 6;
    entries[3] = 9;
    entries[4] = 12;
    entries[5] = 15;
    entries[6] = 18;
    entries[7] = 21;

    entries[8] = 3;
    entries[9] = 37;
    entries[10] = 26;
    entries[11] = 33;
    entries[12] = 46;
    entries[13] = 11;
    entries[14] = 24;
    entries[15] = -11;

    entries[16] = 6;
    entries[17] = 26;
    entries[18] = 21;
    entries[19] = 22;
    entries[20] = 26;
    entries[21] = 24;
    entries[22] = 35;
    entries[23] = 14;

    entries[24] = 9;
    entries[25] = 33;
    entries[26] = 22;
    entries[27] = 86;
    entries[28] = 81;
    entries[29] = -8;
    entries[30] = 25;
    entries[31] = -30;
    
    entries[32] = 12;
    entries[33] = 46;
    entries[34] = 26;
    entries[35] = 81;
    entries[36] = 215;
    entries[37] = -68;
    entries[38] = -71;
    entries[39] = -115;

    entries[40] = 15;
    entries[41] = 11;
    entries[42] = 24;
    entries[43] = -8;
    entries[44] = -68;
    entries[45] = 275;
    entries[46] = 131;
    entries[47] = 120;
    
    entries[48] = 18;
    entries[49] = 24;
    entries[50] = 35;
    entries[51] = 25;
    entries[52] = -71;
    entries[53] = 131;
    entries[54] = 222;
    entries[55] = 203;
    
    entries[56] = 21;
    entries[57] = -11;
    entries[58] = 14;
    entries[59] = -30;
    entries[60] = -115;
    entries[61] = 120;
    entries[62] = 203;
    entries[63] = 241;

    mat = SparseMatrix<Number>(8, entries, cols, rowPtrs);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";
    auto matDiag = mat.getAllDiagonals();
    for (int i = 0; i < 8; i++) {
	std::cout << "mat[" << i << "] = " << matDiag[i] << "\n";
    }
    
    return 0;
}


     
