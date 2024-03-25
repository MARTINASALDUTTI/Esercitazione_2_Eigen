#include <iostream>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

bool SolveSystem(const Matrix2d& Matrix,
                 double& detM,
                 double& condM,
                 double& ErrRelPALU,
                 double& ErrRelQR,
                 const Vector2d& b)
{
    JacobiSVD<Matrix2d> svd(Matrix);
    Vector2d SingularValuesMatrix = svd.singularValues();
    condM = SingularValuesMatrix.maxCoeff()/ SingularValuesMatrix.minCoeff();

    detM =Matrix.determinant();

    if (SingularValuesMatrix.minCoeff()< 1e-16)
    {
        ErrRelPALU= -1;
        ErrRelQR = -1;
        return false;
    }

    Vector2d ExactSol;
    ExactSol << -1.00e+0, -1.00e+0;

    //calcolo la soluzione con la fattorizzazione LU
    Vector2d xPALU = Matrix.fullPivLu().solve(b);

    //calcolo la soluzione con la fattorizzazione QR
    Vector2d xQR = Matrix.fullPivHouseholderQr().solve(b);

    //calcolo errore relativo
    ErrRelPALU = (ExactSol-xPALU).norm() / ExactSol.norm();
    ErrRelQR =  (ExactSol-xQR).norm() / ExactSol.norm();
    return true;
}

int main()
{
    Matrix2d A1, A2, A3;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b1, b2, b3;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    double detA1, condA1, ErrRelPALU_A1, ErrRelQR_A1;
    if (SolveSystem(A1,detA1, condA1, ErrRelPALU_A1,ErrRelQR_A1, b1))
        cout << scientific << "A1 -detA1: " << detA1 << ", CondA1: " << 1.0/condA1 << ", PALU relative error: " <<  ErrRelPALU_A1 <<", QR relative error: " << ErrRelQR_A1<< endl;
    else
        cout << scientific << "A1 -detA1: " << detA1 << ", CondA1: " << 1.0/condA1 << " Matrix is singular" << endl;
    double detA2, condA2, ErrRelPALU_A2, ErrRelQR_A2;
    if (SolveSystem(A2,detA2, condA2, ErrRelPALU_A2, ErrRelQR_A2, b2))
        cout << scientific << "A2 -detA2: " << detA2 << ", CondA2: " << 1.0/condA2 << ", relative error: " << ErrRelPALU_A2 <<", QR relative error: " << ErrRelQR_A2 << endl;
    else
        cout << scientific << "A2 -detA2: " << detA2 << ", CondA2: " << 1.0/condA2 << " Matrix is singular"  << endl;
    double detA3, condA3, ErrRelPALU_A3, ErrRelQR_A3;
    if (SolveSystem(A3,detA3, condA3, ErrRelPALU_A3, ErrRelQR_A3,b3))
        cout << scientific << "A3 -detA3: " << detA3 << ", CondA3: " << 1.0/condA3 << ", relative error: " << ErrRelPALU_A3 << ", QR relative error: " << ErrRelQR_A3 << endl;
    else
        cout << scientific << "A3 -detA3: " << detA3 << ", CondA1: " << 1.0/condA3 << " Matrix is singular"  << endl;

    return 0;
}

