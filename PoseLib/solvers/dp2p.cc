// Copyright (c) 2023, Vaclav Vavra
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "dp2p.h"
#include "PoseLib/misc/univariate.h"
#include <vector>
#include <math.h>

static int solve_quadratic_real_wrapper(double a, double b, double c, double roots[2]) {
    if (c == 0) {
        roots[0] = 0;
        if (a == 0) {
            return 1;
        } else {
            roots[1] = -b / a;
            return 2;
        }
    } else {
        return poselib::univariate::solve_quadratic_real(a, b, c, roots);
    }
}

static bool solve_for_second_row(const Eigen::Vector3d &X_d, const Eigen::Vector3d &x_d, double r31, double r33, Eigen::Matrix3d *R) {

    // X_d(0) * r_21 + X_d(1) * r_22 + X_d(2) * r_23 = x_d(1) ->
    // coeff * first + X_d(1) * r_22 = x_d(1)

    bool solve_first_r21 = true;
    double a = 0;
    double coeff = 0;

    assert(r31 != 0 || r33 != 0);
    // r31 * r21 + r33 * r23 = 0
    if (r33 == 0) {
        solve_first_r21 = false;
        // r21 = 0 * r23
        a = 0;
        coeff = X_d(2);
    } else {
        solve_first_r21 = true;
        // r23 = (-r31/r33) * r21
        a = -r31/r33;
        coeff = a * X_d(2) + X_d(0);
    }

    if (X_d(1) == 0) {
        if (coeff == 0) {
            return false;
        } else {
            // coeff * first = x_d(1)
            double first = x_d(1) / coeff;
            double second = a * first;
            double r_21 = solve_first_r21 ? first : second;
            double r_23 = solve_first_r21 ? second : first;

            double A = 1;
            double C = first * first + second * second - 1;

            double roots[2];
            double solutions = 0;

            if (C > 0) {
                return false;
            } else if (C == 0) {
                roots[0] = 0;
                solutions = 1;
            } else {
                double sq = std::sqrt(-C / A);
                roots[0] = sq;
                roots[1] = -sq;
                solutions = 2;
            }
            for (int i = 0; i < solutions; i++) {
                double r_22 = roots[i];
                R->row(1) = Eigen::Vector3d({r_21, r_22, r_23});
                R->row(0) = R->row(1).cross(R->row(2));
                double tol = 1e-6;
                double diff = R->row(0).dot(X_d) - x_d(0);
                if (abs(diff) < tol) {
                    return true;
                }
            }
            return false;
        }
    } else {
        // coeff * first + X_d(1) * r_22 = x_d(1)
        // r_22 = r22_a * first + r22_b
        double r22_a = -coeff / X_d(1);
        double r22_b = x_d(1) / X_d(1);

        // r21^2 + r22^2 + r23^2 = 1
        // A * first^2 + B * first + C = 0
        double A = 1 + r22_a * r22_a + a * a;
        double B = 2 * r22_a * r22_b;
        double C = r22_b * r22_b - 1.0;

        double roots[2];
        const int sols = solve_quadratic_real_wrapper(A, B, C, roots);

        for (int i = 0; i < sols; i++) {
            double first = roots[i];
            double second = a * first;

            double r_22 = r22_a * first + r22_b;
            double r_21 = solve_first_r21 ? first : second;
            double r_23 = solve_first_r21 ? second : first;
            R->row(1) = Eigen::Vector3d({r_21, r_22, r_23});
            R->row(0) = R->row(1).cross(R->row(2));

            double tol = 1e-6;
            double diff = R->row(0).dot(X_d) - x_d(0);

            if (abs(diff) < tol) {
                return true;
            }
        }
        return false;
    }

}


static int dp2p_z_hor_fixed_lambda(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                         poselib::CameraPoseVector *output) {

    Eigen::Vector3d X_d = X[1] - X[0];
    Eigen::Vector3d x_d = x[1] - x[0];

    assert(X_d(0) != 0.0 || X_d(2) != 0.0);

    double a = 0.0;
    double b = 0.0;

    // X_d(0) * r31 + X_d(2) * r33 = x_d(2)
    bool solve_r31;
    if (X_d(0) != 0.0) {
        // r31 = (x_d(2) - X_d(2) * r33) / X_d(0)
        // solve for r33
        solve_r31 = false;
        a = -X_d(2) / X_d(0);
        b = x_d(2) / X_d(0);
    } else {
        // r33 = (x_d(2) - X_d(0) * r31) / X_d(2)
        // solve for r31
        solve_r31 = true;
        a = -X_d(0) / X_d(2);
        b = x_d(2) / X_d(2);
    }
    double A = (1 + a * a);
    double B = 2 * a * b;
    double C = b * b - 1.0;

    double roots[2];
    const int sols = solve_quadratic_real_wrapper(A, B, C, roots);

    int solutions = 0;
    for (int i = 0; i < sols; i++) {

        double first = roots[i];
        double second = a * first + b;
        double r31 = solve_r31 ? first : second;
        double r33 = solve_r31 ? second : first;

        Eigen::Matrix3d R;
        R.row(2) = Eigen::Vector3d({r31, 0.0, r33});

        bool add = solve_for_second_row(X_d, x_d, r31, r33, &R);
        if (add) {
            Eigen::Vector3d t = x[0] - R * X[0];
            output->emplace_back(R, t);
            solutions++;
        }
    }
    return solutions;
}


static int second_norm(double c, double a, double cos_gamma, double or_b_norm, bool bothRoots, double norms[2]) {

    double a_cos_gamma = a * cos_gamma;
    double quarter_D = a_cos_gamma * a_cos_gamma + c * c - a * a;
    if (quarter_D < 0) {
        return 0;
    } else if (quarter_D == 0.0) {
        norms[0] = a_cos_gamma;
        return 1;
    } else {
        double sqrt_d = std::sqrt(quarter_D);
        double b1 = a_cos_gamma - sqrt_d;
        double b2 = a_cos_gamma + sqrt_d;
        if (b1 < 0) {
            norms[0] = b2;
            return 1;
        } else if (bothRoots) {
            norms[0] = b1;
            norms[1] = b2;
            return 2;
        } else {
            double b1_d = std::abs(b1 - or_b_norm);
            double b2_d = std::abs(b2 - or_b_norm);
            norms[0] = b1_d < b2_d ? b1 : b2;
            return 1;
        }
    }
}

int poselib::dp2p_z_hor(const LambdaComputation lambdaComputation, const std::vector<Eigen::Vector3d> &x,
                                   const std::vector<Eigen::Vector3d> &X, CameraPoseVector *output, const bool bothRoots) {

    double all_solutions = 0;
    std::vector<Eigen::Vector3d> x_in;

    Eigen::Vector3d X_d = X[1] - X[0];
    double X_d_norm = X_d.norm();
    double x1_norm = x[0].norm();
    double x2_norm = x[1].norm();

    if (lambdaComputation == LambdaComputation::RATIO || lambdaComputation == LambdaComputation::BOTH) {
        double delta = X_d_norm / std::sqrt(x[0].squaredNorm() + x[1].squaredNorm() - 2 * x[0].dot(x[1]));
        x_in.emplace_back(delta * x[0]);
        x_in.emplace_back(delta * x[1]);
        all_solutions += dp2p_z_hor_fixed_lambda(x_in, X, output);
        x_in.clear();
    }
    if (lambdaComputation == LambdaComputation::ONE_FROM_OTHER || lambdaComputation == LambdaComputation::BOTH) {

        double cos_gamma = x[0].transpose() / (x1_norm * x2_norm) * x[1];

        double new_norms[2];
        int solutions = second_norm(X_d_norm, x1_norm, cos_gamma, x2_norm, bothRoots, new_norms);
        for (int i = 0; i < solutions; i++) {
            x_in.emplace_back(x[0]);
            x_in.emplace_back(new_norms[i] / x2_norm * x[1]);
            all_solutions += dp2p_z_hor_fixed_lambda(x_in, X, output);
            x_in.clear();
        }

        solutions = second_norm(X_d_norm, x2_norm, cos_gamma, x1_norm, bothRoots, new_norms);
        for (int i = 0; i < solutions; i++) {
            x_in.emplace_back(new_norms[i] / x1_norm * x[0]);
            x_in.emplace_back(x[1]);
            all_solutions += dp2p_z_hor_fixed_lambda(x_in, X, output);
            x_in.clear();
        }
    }
    return all_solutions;
}
