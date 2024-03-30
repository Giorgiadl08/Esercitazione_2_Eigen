#include <iostream>
#include "Eigen/Eigen"


using namespace std;
using namespace Eigen;


//funzione che risolve il sist con fattorizzazione PALU
void solveSystemWithPALU(const MatrixXd& A, const VectorXd& b, VectorXd& x, double& errRel, const VectorXd& sol)
{
    PartialPivLU<MatrixXd> palu(A);
    x = palu.solve(b);

    errRel = (sol - x).norm() / sol.norm();
}

//funzione che risolve il sist con fattorizzazione QR
void solveSystemWithQR(const MatrixXd& A, const VectorXd& b, VectorXd& x, double& errRel, const VectorXd& sol)
{
    HouseholderQR<MatrixXd> qr(A);
    x = qr.solve(b);

    errRel = (sol - x).norm() / sol.norm();
}

int main()
{
    //definisco la sol esatta per tutti e 3 i sistemi
    VectorXd exactSolution(2);
    exactSolution << -1.0e+0, -1.0e+00;


    //definisco i sistemi
    MatrixXd A1(2, 2);
    VectorXd b1(2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    MatrixXd A2(2, 2);
    VectorXd b2(2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    MatrixXd A3(2, 2);
    VectorXd b3(2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    //risolvo ogni sistema con PALU e QR
    VectorXd x1_pal, x2_pal, x3_pal, x1_qr, x2_qr, x3_qr;
    double error1_pal, error2_pal, error3_pal, error1_qr, error2_qr, error3_qr;

    solveSystemWithPALU(A1, b1, x1_pal, error1_pal, exactSolution);
    solveSystemWithQR(A1, b1, x1_qr, error1_qr, exactSolution);
    solveSystemWithPALU(A2, b2, x2_pal, error2_pal, exactSolution);
    solveSystemWithQR(A2, b2, x2_qr, error2_qr, exactSolution);
    solveSystemWithPALU(A3, b3, x3_pal, error3_pal, exactSolution);
    solveSystemWithQR(A3, b3, x3_qr, error3_qr, exactSolution);

    // Output
    cout << "Solution for System 1 with PALU: " << x1_pal.transpose() << ", Relative error: " << error1_pal << endl;
    cout << "Solution for System 1 with QR: " << x1_qr.transpose() << ", Relative error: " << error1_qr << endl;
    cout << "Solution for System 2 with PALU: " << x2_pal.transpose() << ", Relative error: " << error2_pal << endl;
    cout << "Solution for System 2 with QR: " << x2_qr.transpose() << ", Relative error: " << error2_qr << endl;
    cout << "Solution for System 3 with PALU: " << x3_pal.transpose() << ", Relative error: " << error3_pal << endl;
    cout << "Solution for System 3 with QR: " << x3_qr.transpose() << ", Relative error: " << error3_qr << endl;

    return 0;
}
