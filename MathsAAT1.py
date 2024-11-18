# Importing necessary libraries
import numpy as np

# Function to perform the Gauss-Seidel iteration
def gauss_seidel(coeff_matrix, const_vector, tolerance, max_iterations):
    # Initial guess (starting with zero for each variable)
    n = len(const_vector)
    x = np.zeros(n)
    
    # List to store values for each iteration
    iteration_values = []
    
    # Iterate until the solution converges or max_iterations is reached
    for k in range(max_iterations):
        x_old = np.copy(x)  # Keep a copy of the previous values
        for i in range(n):
            # Summing all terms except the ith term
            sum1 = sum(coeff_matrix[i][j] * x[j] for j in range(n) if j != i)
            # Updating the ith variable
            x[i] = (const_vector[i] - sum1) / coeff_matrix[i][i]
        
        # Storing values for the current iteration
        iteration_values.append(np.copy(x))
        
        # Check for convergence by comparing current and previous values
        if np.allclose(x, x_old, atol=tolerance):
            return k+1, x, iteration_values  # Return when converged

    return max_iterations, x, iteration_values  # If max_iterations reached

# Main program
def main():
    # Getting user input for the number of equations
    print("Gauss-Seidel Method Solver")
    n = int(input("Enter the number of equations: "))
    
    # Initializing coefficient matrix and constant vector
    coeff_matrix = []
    const_vector = []
    
    # Taking input for the coefficient matrix
    print("Enter the coefficients row by row:")
    for i in range(n):
        row = list(map(float, input(f"Enter coefficients for equation {i+1}, separated by spaces: ").split()))
        coeff_matrix.append(row)
    
    # Taking input for the constant vector
    print("Enter the constant terms:")
    for i in range(n):
        const = float(input(f"Enter constant term for equation {i+1}: "))
        const_vector.append(const)
    
    # Converting lists to numpy arrays
    coeff_matrix = np.array(coeff_matrix)
    const_vector = np.array(const_vector)
    
    # Getting user input for tolerance and max iterations
    tolerance = float(input("Enter the tolerance level (e.g., 0.0001): "))
    max_iterations = int(input("Enter the maximum number of iterations: "))
    
    # Solving using Gauss-Seidel method
    iterations, solution, iteration_values = gauss_seidel(coeff_matrix, const_vector, tolerance, max_iterations)
    
    # Outputting the number of iterations and final solutions
    print(f"\nSolution found after {iterations} iterations:")
    for i, value in enumerate(solution):
        print(f"x{i+1} = {value:.4f}")
    
    # Printing the values from each iteration in table form
    print("\nIteration values (up to 4 decimal places):")
    for k, values in enumerate(iteration_values):
        print(f"Iteration {k+1}: ", end="")
        for value in values:
            print(f"{value:.4f} ", end="")
        print()

# Running the main program
if __name__ == "__main__":
    main()