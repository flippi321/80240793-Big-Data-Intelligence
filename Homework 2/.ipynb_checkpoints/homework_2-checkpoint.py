import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- Data pre-processing ----- 

users = pd.read_csv('Project2-data/users.txt', sep=' ', names=['user_id'], header=None)
movies = pd.read_csv('Project2-data/movie_titles.txt', names=['movie_id', 'year', 'title'], header=None, on_bad_lines='skip')
movie_ids = movies['movie_id']

ratings_train = pd.read_csv('Project2-data/netflix_train.txt', sep=' ', names=['user_id', 'movie_id', 'rating', 'date'], header=None)
ratings_test = pd.read_csv('Project2-data/netflix_test.txt', sep=' ', names=['user_id', 'movie_id', 'rating', 'date'], header=None)

# We create an empty matrix of the size [users, movie_ids]
# Here empty values are set to 0 and assume there is maximum one review per movie per user
matrix = ratings_train.pivot(index='user_id', columns='movie_id', values='rating').fillna(-1)

# ----- Collaborative filtering -----
def userCF(matrix, user_id, movie_id, filter_only_rated=True):
    user_i_ratings = matrix.loc[user_id]
    k = 0
    top = 0
    bottom = 0

    # To hinder users being treated as a Series, we convert them to a DataFrame
    user_df = users['user_id']
    user_df_filtered = user_df[user_df != user_id]

    # We estimate the similarity sum between the user and all other users
    for user_k_id in user_df_filtered:
        user_k_ratings = matrix.loc[user_k_id]

        # If filter_only_rated, we don't account for people who haven't rated the movie
        if filter_only_rated and (user_k_ratings[movie_id] == -1):
            continue

        # Calculate the cosine similarity between the ratings
        similarity = np.dot(user_i_ratings, user_k_ratings) / (np.linalg.norm(user_i_ratings) * np.linalg.norm(user_k_ratings))

        top += (similarity * user_k_ratings[movie_id])
        bottom += similarity   

        # We increment k to see how many people
        k+=1
    
    return top/bottom

# Check the difference from the testset and our estimate
def checkError(estimated, actual):
    return np.absolute(estimated-actual)

# ----- Testing the userCF function -----
def userCF_rmse(test_size=10):
    # Sample a subset of the ratings_test for faster computation
    sampled_test = ratings_test.sample(n=test_size, random_state=42)
    
    # Calculate deviations in a vectorized form
    deviations = []
    for _, data_real in sampled_test.iterrows():
        data_estimated = userCF(matrix, data_real['user_id'], data_real['movie_id'])
        deviation = (data_real['rating'] - data_estimated) ** 2
        deviations.append(deviation)
    
    return np.sqrt(np.mean(deviations))


print("----- RMSE -----")
for i in range(100, 1000, 100):
    print(f" size %4s: %10.2f" % (i, userCF_rmse(i)))