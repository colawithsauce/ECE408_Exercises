#include "ece408.hpp"
#include "fill_matrix.hpp"
#include <gtest/gtest.h>

TEST(matrix_multi_tile, correctness_of_square__multi_eye)
{
    int M = 16;
    double* matA_h = new double[M * M];
    double* matB_h = new double[M * M];
    double* matC_h = new double[M * M];

    fill_randomly(matA_h, M, M);
    fill_eye(matB_h, M, M);

    matrix_multi_tile((const double*)matA_h, (const double*)matB_h, (double*)matC_h, M, M, M);

    ASSERT_TRUE(matrix_same(matA_h, matC_h, M, M));
}

TEST(matrix_multi_tile, correctness_of_square__large_matrix)
{
    int M = 1024;
    double* matA_h = new double[M * M];
    double* matB_h = new double[M * M];
    double* matC_h = new double[M * M];

    fill_randomly(matA_h, M, M);
    fill_eye(matB_h, M, M);

    matrix_multi_tile((const double*)matA_h, (const double*)matB_h, (double*)matC_h, M, M, M);

    ASSERT_TRUE(matrix_same(matA_h, matC_h, M, M));
}

TEST(matrix_multi_tile, correctness_of_not_square_1)
{
    double A[3][4] = { 0.8851664842189177, 0.9768207526431404, 0.32847186349921464, 0.9552227448759539, 0.9352328627640409, 0.6764536989920251, 0.948111409671085, 0.13209170510687795, 0.14243240013658698, 0.6567905523879392, 0.344911334274699, 0.3734305748808653 };

    double B[4][3] = { 0.942444520235116, 0.13000221198291373, 0.28102813916562575, 0.3985900141025147, 0.08160662702912336, 0.276142309710385, 0.19583575462668168, 0.059716338462446505, 0.5121862777595527, 0.6804390140232334, 0.1476914782829044, 0.9609758346922692 };

    double C[3][3] = { 1.9378686580775226, 0.3554820440131474, 1.604682984408352, 1.4265872390433758, 0.2529120066307212, 1.0621708283527134, 0.7176674941564154, 0.147864344338038, 0.7569117833546364 };

    double* matC = (double*)malloc(sizeof(double) * 3 * 3);

    matrix_multi_tile((const double*)A, (const double*)B, (double*)matC, 3, 4, 3);

    ASSERT_TRUE(matrix_same(matC, (const double*)C, 3, 3));
}

TEST(matrix_multi_tile, correctness_of_not_square_2)
{
    double A[3][4] = { 0.5255498673806541, 0.5632248055177753, 0.9066744897120185, 0.218152452605887, 0.45268786443022013, 0.17498146500595668, 0.8508821304010632, 0.9170248806935053, 0.5623262513506225, 0.02880696408354433, 0.9592019919703658, 0.6586257117477086 };

    double B[4][4] = { 0.6096028831502072, 0.8486709843101432, 0.5358317047171188, 0.5114128118970882, 0.9196510601000598, 0.329124417442474, 0.45253260650871363, 0.678664380000182, 0.19914731286037257, 0.013198342682737896, 0.45279792058112367, 0.2606603853876446, 0.11757897931018046, 0.7618634445607829, 0.24856739718547782, 0.7135571813156518 };

    double C[3][4] = { 1.0445589348401385, 0.8095589388578133, 1.001249781563181, 1.04301191999476, 0.7141554564186251, 1.1516516965173138, 0.9349694756058016, 1.2264050142135186, 0.6377510972780351, 1.001153778529845, 0.9123858706807344, 1.0271241771345185 };

    double* matC = (double*)malloc(sizeof(double) * 3 * 4);

    matrix_multi_tile((const double*)A, (const double*)B, (double*)matC, 3, 4, 4);

    ASSERT_TRUE(matrix_same(matC, (const double*)C, 3, 4));
}
