import numpy as np

def predict_word_from_embedding(embedding,phoc_levels,word_length):
    if word_length not in phoc_levels:
        print("Word length not found in the phoc_levels")
        return
    
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    level_index = phoc_levels.index(word_length) # index of the word_length in the phoc_levels array
    
    starting_index = np.sum(phoc_levels[:level_index])*len(unigrams) # starting index for the word_length in the embedding vector
    word_hist = embedding[starting_index:starting_index + word_length*len(unigrams)] # histogram of the word embedding
    
    word = ""
    for index in range(word_length):
        alphabet_hist = word_hist[index*len(unigrams):(index+1)*len(unigrams)] # histogram for the 
        alphabet_index = np.argmax(alphabet_hist)
        alphabet = unigrams[alphabet_index]
        word = word + alphabet
    
    return word