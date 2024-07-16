import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'               #Support large floating point operations
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'     # Support CPU optimizations for ternsorflow

# Define the directory where your text files are stored
directory = os.getcwd() + "/text_files"  # Replace with your actual directory

# Load text files
documents = []
filenames = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            documents.append(file.read())
            filenames.append(filename)

# Check the loaded documents
# for doc, fname in zip(documents, filenames):
#     print(f"Document: {fname}")
#     print(doc[:200])  # Print the first 200 characters
#     print("\n")




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(X)

# Creating a dataframe of the cosine similarity matrix
import pandas as pd
sim_df_tfidf = pd.DataFrame(cosine_sim_matrix, index=filenames, columns=filenames)


# Display the similarity matrix
# print("TF-IDF Cosine Similarity Matrix:")
# print(sim_df_tfidf)
#Here in the output it shows 1--> Similar 0--> No similar


from sentence_transformers import SentenceTransformer, util    #semantically transform sentences

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for each document
embeddings = model.encode(documents)

# Calculate similarity matrix
sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()

# Display the similarity matrix
sim_df_embeddings = pd.DataFrame(sim_matrix, index=filenames, columns=filenames)

# Display the similarity matrix
# print("Sentence Transformers Similarity Matrix:")
# print(sim_df_embeddings)





import numpy as np
combined_sim_matrix = (sim_df_tfidf + sim_df_embeddings) / 2
# Example threshold (adjust as per your needs)
threshold = 0
# Analyze the combined similarity matrix
num_docs = len(combined_sim_matrix)
for i in range(num_docs):
    for j in range(i + 1, num_docs):  # Only need to check upper triangle (excluding diagonal)
        similarity_score = combined_sim_matrix.iloc[i, j]
        similarity_percentage = similarity_score * 100
        print(f"Similarity between document {i+1} and document {j+1}: {similarity_percentage:.2f}%")
        if similarity_score >= threshold:
            print(f"Potential plagiarism detected between document {i+1} and document {j+1} with similarity score {similarity_score}")