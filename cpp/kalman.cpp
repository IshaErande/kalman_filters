#include <iostream>
#include <eigen3/Eigen/Dense>  // Ensure Eigen is installed for matrix operations
#include <vector>

using namespace Eigen;

class KalmanFilter {
private:
    Vector2d x;           // State vector [position, velocity]
    Matrix2d P;           // Covariance matrix
    double acc_var;       // Acceleration variance (process noise)

public:
    // Constructor to initialize the filter
    KalmanFilter(double init_x, double init_v, double acc_var)
        : x(init_x, init_v), acc_var(acc_var) {
        P = Matrix2d::Identity();  // Initialize covariance matrix as identity
    }

    // Init function to reinitialize filter parameters if needed
    void init(double init_x, double init_v, double acc_var) {
        x = Vector2d(init_x, init_v);
        P = Matrix2d::Identity();
        this->acc_var = acc_var;
    }

    // Predict function to estimate the new state based on elapsed time (dt)
    void predict(double dt) {
        Matrix2d F;
        F << 1, dt,
             0, 1;

        x = F * x;

        Vector2d G;
        G << 0.5 * dt * dt, dt;

        P = F * P * F.transpose() + G * G.transpose() * acc_var;
    }

    // Update function that combines measurements from the camera and encoder
    double update(double camera_distance, double encoder_distance, double meas_var_camera, double meas_var_encoder) {
        RowVector2d H;
        H << 1, 0;

        double combined_measurement = (camera_distance + encoder_distance) / 2.0;
        double combined_meas_var = (meas_var_camera + meas_var_encoder) / 2.0;
        Matrix<double, 1, 1> R;
        R << combined_meas_var;

        VectorXd z(1);
        z << combined_measurement;
        VectorXd y = z - H * x;

        Matrix<double, 1, 1> S = H * P * H.transpose() + R;
        Vector2d K = P * H.transpose() * S.inverse();

        x = x + K * y;
        P = (Matrix2d::Identity() - K * H) * P;

        return x(0);  // Position estimate
    }

    double getPosition() const { return x(0); }
    double getVelocity() const { return x(1); }
};

// Test case simulating sensor data with noise
void testKalmanFilter() {
    KalmanFilter kf(0.0, 0.0, 1.0);
    double dt = 1.0;

    std::vector<double> camera_distances = {5.0, 5.3, 5.5, 5.1, 5.2};
    std::vector<double> encoder_distances = {5.1, 5.4, 5.2, 5.0, 5.3};
    double meas_var_camera = 0.1;
    double meas_var_encoder = 0.2;

    std::cout << "Time Step | Camera Dist | Encoder Dist | Filtered Dist | Estimated Vel\n";
    std::cout << "---------------------------------------------------------------------\n";

    for (size_t i = 0; i < camera_distances.size(); ++i) {
        // Prediction step
        kf.predict(dt);

        // Update step with the measurements
        double filtered_distance = kf.update(camera_distances[i], encoder_distances[i], meas_var_camera, meas_var_encoder);

        // Output results
        std::cout << i + 1 << "         | " << camera_distances[i] << "         | " 
                  << encoder_distances[i] << "         | " << filtered_distance 
                  << "        | " << kf.getVelocity() << "\n";
    }
}

int main() {
    testKalmanFilter();
    return 0;
}
