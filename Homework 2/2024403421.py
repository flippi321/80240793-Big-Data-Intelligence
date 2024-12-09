import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

google_colab = False

local_root = 'Project2-data'
colab_root = '/content/drive/MyDrive/Colab Notebooks/Project2-data/'
root = colab_root if google_colab else local_root

# Load data
users = pd.read_csv(f'{root}/users.txt', sep=' ', names=['user_id'], header=None)
movies = pd.read_csv(f'{root}/movie_titles.txt', names=['movie_id', 'year', 'title'], header=None, on_bad_lines='skip')
movie_ids = movies['movie_id']

ratings_train = pd.read_csv(f'{root}/netflix_train.txt', sep=' ', names=['user_id', 'movie_id', 'rating', 'date'], header=None)
ratings_test = pd.read_csv(f'{root}/netflix_test.txt', sep=' ', names=['user_id', 'movie_id', 'rating', 'date'], header=None)

# Create matrix
matrix = ratings_train.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Convert to numpy matrix for computation
vector_matrix = matrix.values

# Map user and movie IDs to matrix indices
user_id_to_index = {user_id: idx for idx, user_id in enumerate(matrix.index)}
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(matrix.columns)}


# --- Collaborative Filtering ---

def compute_user_similarities(matrix):
    norm_matrix = np.linalg.norm(matrix, axis=1)
    similarity_matrix = np.dot(matrix, matrix.T) / (norm_matrix[:, None] * norm_matrix)
    
    # Avoid division by zero
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return similarity_matrix

def userCF(matrix, user_id, movie_id, similarity_matrix, filter_only_rated=True):
    # Map user_id and movie_id to matrix indices
    user_index = user_id_to_index.get(user_id, None)
    movie_index = movie_id_to_index.get(movie_id, None)

    user_ratings = matrix.iloc[user_index].to_numpy()

    # Similar users to the target user
    similarities = similarity_matrix[user_index]

    # Only consider users who have rated the movie
    if filter_only_rated:
        rated_users = user_ratings != 0
        similarities = similarities[rated_users]
        user_ratings = user_ratings[rated_users]
    
    # Compute weighted sum of ratings for the movie
    numerator = np.dot(similarities, user_ratings)
    denominator = np.sum(np.abs(similarities))  # Sum of similarities

    if denominator == 0:
        return 0  # If no similar users, return 0 (or can return the average rating)
    return numerator / denominator

def userCF_rmse(similarity_matrix, matrix, ratings_test):
    # Create a matrix for the predicted ratings
    predicted_ratings = np.zeros_like(matrix.values)
    
    # Get the non-zero entries (rated items) in the test set
    test_ratings = ratings_test[['user_id', 'movie_id', 'rating']]
    
    # Map user and movie IDs to matrix indices
    user_index_test = test_ratings['user_id'].map(user_id_to_index)
    movie_index_test = test_ratings['movie_id'].map(movie_id_to_index)
    
    # Filter out rows where user or movie is not found in the matrix
    valid_test_ratings = test_ratings[~user_index_test.isna() & ~movie_index_test.isna()]
    user_index_test = user_index_test[valid_test_ratings.index]
    movie_index_test = movie_index_test[valid_test_ratings.index]
    actual_ratings = valid_test_ratings['rating'].values
    
    # Get predicted ratings for test data
    predicted_ratings = np.array([
        userCF(matrix, user_id, movie_id, similarity_matrix, filter_only_rated=True)
        for user_id, movie_id in zip(valid_test_ratings['user_id'], valid_test_ratings['movie_id'])
    ])
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))
    return rmse

def compare_predictions(predictions=10, similarity_matrix=None):
    print(f"\nPredicting {predictions} random values")

    for i in range(predictions):
        actual_rating = 0

        while(actual_rating == 0):
            movie_index = random.randint(0, len(matrix.columns) - 1)
            user_index = random.randint(0, len(matrix.index) - 1)
            movie_id = matrix.columns[movie_index]
            user_id = matrix.index[user_index]
            actual_rating = matrix.iloc[user_index, movie_index]

        predicted_rating = userCF(matrix, user_id, movie_id, similarity_matrix, filter_only_rated=True)
        actual_rating = matrix.iloc[user_index, movie_index]
        print(f"user {user_id}, movie {movie_id}: {predicted_rating:.1f} (actual: {actual_rating:.1f})")

print("Computing user similarities")
similarity_matrix = compute_user_similarities(matrix)
print("Done computing similarities :) \n")

# Evaluate RMSE for a specific test set size
print("Computing RMSE")
test_rmse = userCF_rmse(similarity_matrix=similarity_matrix, matrix=matrix, ratings_test=ratings_test)
print(f"Test RMSE: {test_rmse}")

# We will now test 10 random values and see what is predicted
compare_predictions(predictions=10, similarity_matrix=similarity_matrix)

# --- Gradient Descent ---
def matrix_decomposition(X_train, X_test, k=50, lambda_=0.5, alpha=1e-3, max_iter=100, tolerance=1e-1):
    # Ensure input matrices are CuPy arrays
    X_train = cp.asarray(X_train)
    X_test = cp.asarray(X_test)

    # Initialize U and V matrices with small random values using CuPy (GPU)
    num_users, num_movies = X_train.shape
    U = cp.random.rand(num_users, k)
    V = cp.random.rand(num_movies, k)

    # Create an indicator matrix A (1 for known values, 0 for unknown values) using CuPy
    A = (X_train > 0).astype(int)

    # To store the previous value of the objective function for convergence check
    prev_loss = float('inf')

    for iteration in range(max_iter):
        # Compute the predicted ratings matrix (U * V.T)
        X_pred = cp.dot(U, V.T)

        # Calculate the loss function J (with Frobenius norm and regularization)
        error = A * (X_train - X_pred)
        loss = 0.5 * cp.sum(error**2) + lambda_ * (cp.sum(U**2) + cp.sum(V**2))

        # Compute gradients with respect to U and V
        grad_U = -cp.dot(error, V) + 2 * lambda_ * U
        grad_V = -cp.dot(error.T, U) + 2 * lambda_ * V

        # Update U and V using gradient descent
        U -= alpha * grad_U
        V -= alpha * grad_V

        # Check for convergence by comparing the change in loss function
        if cp.abs(prev_loss - loss) < tolerance:
            print(f"Convergence reached at iteration {iteration + 1}")
            break

        # Our model is overfitting
        if (loss - prev_loss) >= loss*0.01 and iteration > 1000:
            print(f"Model overfitted with loss increasing at {iteration + 1}")
            break

        # Print the loss at every iteration to monitor the progress
        if iteration == 0 or iteration % 100 == 0:
            rmse = calculate_RMSE(X_pred, X_test)
            print(f"     {iteration}: {loss:.4f}, RMSE: {rmse:.4f}")

        if loss == cp.inf or cp.isnan(loss):
            print(f"Loss too high at iteration {iteration + 1}")
            break

        prev_loss = loss

    # Compute RMSE on the test set
    rmse = calculate_RMSE(X_pred, X_test)

    return U, V, X_pred, rmse

def logging_matrix_decomposition(X_train, X_test, k=50, lambda_=0.5, alpha=1e-3, max_iter=100, tolerance=1e-1):
    X_train = cp.asarray(X_train)
    X_test = cp.asarray(X_test)

    num_users, num_movies = X_train.shape
    U = cp.random.rand(num_users, k)
    V = cp.random.rand(num_movies, k)

    A = (X_train > 0).astype(int)
    prev_loss = float('inf')
    losses = []
    rmses = []

    for iteration in range(max_iter):
        X_pred = cp.dot(U, V.T)
        error = A * (X_train - X_pred)
        loss = 0.5 * cp.sum(error**2) + lambda_ * (cp.sum(U**2) + cp.sum(V**2))

        # Convergence and Error Checks
        if cp.abs(prev_loss - loss) < tolerance:
            print(f"Convergence reached at iteration {iteration + 1}")
            break
        if loss == cp.inf or cp.isnan(loss):
            print(f"Loss too high at iteration {iteration + 1}")
            break

        # Compute RMSE and Append Tracking Values
        rmse = calculate_RMSE(X_pred, X_test)
        losses.append(float(loss))
        rmses.append(float(rmse))

        # Gradient Updates
        grad_U = -cp.dot(error, V) + 2 * lambda_ * U
        grad_V = -cp.dot(error.T, U) + 2 * lambda_ * V
        U -= alpha * grad_U
        V -= alpha * grad_V

        # Print Progress
        if iteration == 0 or iteration % 100 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.4f}, RMSE = {rmse:.4f}")

        prev_loss = loss

    final_rmse = calculate_RMSE(X_pred, X_test)
    return U, V, X_pred, final_rmse, losses, rmses

def calculate_RMSE(X_pred, X_test):
    mask = X_test > 0
    error = (X_pred - X_test) ** 2
    rmse = cp.sqrt(cp.sum(error[mask]) / cp.sum(mask))

    return rmse


def check_predictions(X_pred, X_test, n=5):
    # Ensure that the datasets are still in CuPy for filtering
    non_zero_indices = cp.where(X_test > 0)

    # Check if there are enough non-zero ratings
    if len(non_zero_indices[0]) < n:
        print("Not enough non-zero ratings to sample.")
        return

    # Convert the indices of non-zero ratings to NumPy for sampling
    non_zero_indices_np = (cp.asnumpy(non_zero_indices[0]), cp.asnumpy(non_zero_indices[1]))

    # Sample n random indices
    random_indices = random.sample(range(len(non_zero_indices_np[0])), n)

    # Convert the prediction and test arrays to NumPy
    X_pred_np = cp.asnumpy(X_pred)
    X_test_np = cp.asnumpy(X_test)

    print(f" --- Testing {n} random Movie-User Pairs ---")
    for idx in random_indices:
        user = non_zero_indices_np[0][idx]
        movie = non_zero_indices_np[1][idx]

        # Check if the indices are within the valid range
        if movie >= len(movie_ids) or user >= len(users['user_id']):
            print(f"Skipping invalid index: Movie index {movie}, User index {user}")
            continue

        # Display predicted and actual values for the sampled pair
        print(
            f"Movie: {movie_ids[movie]}, User: {users['user_id'][user]}: Predicted: {X_pred_np[user, movie]:.2f}, Actual: {X_test_np[user, movie]}"
        )

# Create X_test from the test data (same as X_train structure)
X_train = matrix.values  # Use the pivot table from earlier
X_test = ratings_test.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# The testdata actually miss some values
# We take every movie and user found in the training set, not present in the testset, and add these with the values 0
X_test = X_test.reindex(columns=matrix.columns, fill_value=0)

# Setup
max_iter = 10000
best_rmse = float('inf')
X_test_np = X_test.values   # To speed up calculations I've used numpy arrays instead of pandas dataframes
X_test_cp = cp.asarray(X_test_np)

print("Starting plotting for specified values")
k = 50
lambda_ = 0.01
alpha = 1e-4
max_iter = 5000

U, V, X_pred, final_rmse, losses, rmses = logging_matrix_decomposition(X_train, X_test_cp, k=k, lambda_=lambda_, alpha=alpha, max_iter=max_iter)

# Plotting
iterations = list(range(len(losses)))

plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(iterations, losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()

# Plot RMSE
plt.subplot(1, 2, 2)
plt.plot(iterations, rmses, label='Test RMSE', color='orange')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Test RMSE over Iterations')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Final RMSE on Test Set: {final_rmse:.4f}")

# Testing the algorithm with different k and alpha
print("Started algorithm")

max_iter = 10000

for a_0 in [5e-5, 7e-5, 1e-5]:
    for k in range(5, 15, 1):
        for lambda_ in [10, 1, 1e-1]:
          # New best parameters found: k=10, alpha=5e-05, lambda=1, max_iter=10000, RMSE=0.8111
          print(f"     k={k}, alpha={a_0}, lambda={lambda_}, max_iter={max_iter}")
          U, V, X_pred, test_rmse = matrix_decomposition(X_train, X_test_cp, k, lambda_, alpha=a_0, max_iter=max_iter, tolerance=100)

          # Track the best combination based on RMSE
          if test_rmse < best_rmse:
              best_rmse = test_rmse
              best_k = k
              best_lambda = lambda_
              best_alpha = a_0
              print(f"New best parameters found: k={best_k}, alpha={best_alpha}, lambda={best_lambda}, max_iter={max_iter}, RMSE={best_rmse:.4f}")
              check_predictions(X_pred, X_test_cp, n=5)







