#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "ShaderClass.h"
#include <vector>

#define PI                              3.141592    // PI approximate value
#define H                               0.005        // time step length for numerical integration
#define ROD_LENGHT                      0.8         // length of rod 
#define GRAVITY                         9.81        // gravitational constant
#define THETA_INIT_VALUE                PI / 4      // Initial angle
#define ANGULAR_VELOCITY_INIT_VALUE     2           // Initial angular velocity
#define TIME_INIT_VALUE                 0           // Initial time
#define RADIUS                          0.15        // Radius of pendulum circle

// pendulum rod
const float rod[] = {
    -0.01f, 0.0f,
    -0.01f, ROD_LENGHT,
    0.01f, 0.0f,
    0.01f, ROD_LENGHT,
};

// vertices and indices of pendulum's circle
std::vector<glm::vec3> vertices;
std::vector<unsigned int> indices;

unsigned int VBO;
unsigned int circleVAO;
unsigned int rodVAO;
unsigned int EBO;

// Function for angular velocity (f function for numerical integration)
float f(float time, float theta, float omega) {
    return omega;
}

// Function for angular acceleration (g function for numerical integration)
float g(float time, float theta, float omega) {
    return -(GRAVITY / ROD_LENGHT) * sin(theta);
}

void RungeKuttaIntegration(float& theta, float& angular_velocity, float& time) {
    float k1_1 = H * f(time, theta, angular_velocity);
    float k1_2 = H * g(time, theta, angular_velocity);
    float k2_1 = H * f(time + H / 2, theta + k1_1 / 2, angular_velocity + k1_2 / 2);
    float k2_2 = H * g(time + H / 2, theta + k1_1 / 2, angular_velocity + k1_2 / 2);
    float k3_1 = H * f(time + H / 2, theta + k2_1 / 2, angular_velocity + k2_2 / 2);
    float k3_2 = H * g(time + H / 2, theta + k2_1 / 2, angular_velocity + k2_2 / 2);
    float k4_1 = H * f(time + H, theta + k3_1, angular_velocity + k3_2);
    float k4_2 = H * g(time + H, theta + k3_1, angular_velocity + k3_2);

    // Update theta and omega
    theta += (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6;
    angular_velocity += (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6;

    if (theta > 2 * PI) theta -= 2 * PI;
    if (theta < -2 * PI) theta += 2 * PI;

    time += H;
}

void EulerIntegration(float& theta, float& angular_velocity, float& time) {
    theta += H * angular_velocity;
    angular_velocity += H * g(time, theta, angular_velocity);

    if (theta > 2 * PI) theta -= 2 * PI;
    if (theta < -2 * PI) theta += 2 * PI;

    time += H;
}

void VerletIntegration(float& theta, float& angular_velocity, float& time, float& theta_old) {
    float theta_new = 2 * theta - theta_old + H * H * g(time, theta, angular_velocity);
    angular_velocity = (theta_new - theta_old) / (2 * H);
    theta_old = theta;
    theta = theta_new;

    if (theta > 2 * PI) theta -= 2 * PI;
    if (theta < -2 * PI) theta += 2 * PI;

    time += H;
}

void buildCircle(float radius, int vCount) {
    float angle = 360.0f / vCount;

    int triangleCount = vCount - 2;

    std::vector<glm::vec3> temp;

    for (int i = 0; i < vCount; i++) {
        float currentAngle = angle * i;
        float x = radius * cos(glm::radians(currentAngle));
        float y = radius * sin(glm::radians(currentAngle));
        float z = 0.0f;

        vertices.push_back(glm::vec3(x, y, z));
    }

    // push indexes of each triangle points
    for (int i = 0; i < triangleCount; i++) {
        indices.push_back(0);
        indices.push_back(i + 1);
        indices.push_back(i + 2);
    }
}

int main(int argc, char** argv) {
    if (!glfwInit()) {
        return -1;
    }

    int initWidth = 800;
    int initHeight = 600;
    int* width = &initWidth;
    int* height = &initHeight;
    GLFWwindow* window = glfwCreateWindow(*width, *height, "Pendulum", NULL, NULL);
    if (window == NULL) {
        std::cout << "Error. I could not create a window at all!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    Shader shaderProgram("default.vert", "default.frag");

    //draws pendulum
    buildCircle(0.1, 128);
    glGenVertexArrays(1, &circleVAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(circleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);

    glGenVertexArrays(1, &rodVAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(rodVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rod), rod, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Initial values for Runge-Kutta
    float theta_rk = THETA_INIT_VALUE;
    float angular_velocity_rk = ANGULAR_VELOCITY_INIT_VALUE;
    float time_rk = TIME_INIT_VALUE;

    // Initial values for Euler
    float theta_euler = THETA_INIT_VALUE;
    float angular_velocity_euler = ANGULAR_VELOCITY_INIT_VALUE;
    float time_euler = TIME_INIT_VALUE;

    // Initial values for Verlet
    float theta_verlet = THETA_INIT_VALUE;
    float angular_velocity_verlet = ANGULAR_VELOCITY_INIT_VALUE;
    float time_verlet = TIME_INIT_VALUE;
    float theta_old_verlet = THETA_INIT_VALUE - ANGULAR_VELOCITY_INIT_VALUE * H;

    while (!glfwWindowShouldClose(window)) {
        glfwGetWindowSize(window, width, height);
        glViewport(0, 0, *width, *height);

        glClearColor(0.0f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shaderProgram.Activate();

        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 view = glm::mat4(1.0f);

        // Runge-Kutta Integration and drawing
        RungeKuttaIntegration(theta_rk, angular_velocity_rk, time_rk);

        view = glm::rotate(glm::mat4(1.0f), theta_rk, glm::vec3(0.0f, 0.0f, 1.0f));
        view = glm::translate(view, glm::vec3(0.0f, -ROD_LENGHT, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "color"), 1.0f, 0.0f, 0.0f);  // Red color

        glBindVertexArray(circleVAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glBindVertexArray(rodVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Euler Integration and drawing
        EulerIntegration(theta_euler, angular_velocity_euler, time_euler);

        view = glm::rotate(glm::mat4(1.0f), theta_euler, glm::vec3(0.0f, 0.0f, 1.0f));
        view = glm::translate(view, glm::vec3(0.0f, -ROD_LENGHT, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "color"), 0.0f, 1.0f, 0.0f);  // Green color

        glBindVertexArray(circleVAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glBindVertexArray(rodVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Verlet Integration and drawing
        VerletIntegration(theta_verlet, angular_velocity_verlet, time_verlet, theta_old_verlet);

        view = glm::rotate(glm::mat4(1.0f), theta_verlet, glm::vec3(0.0f, 0.0f, 1.0f));
        view = glm::translate(view, glm::vec3(0.0f, -ROD_LENGHT, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "color"), 0.0f, 0.0f, 1.0f);  // Blue color

        glBindVertexArray(circleVAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glBindVertexArray(rodVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    shaderProgram.Delete();
}
