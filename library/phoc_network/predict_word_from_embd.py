import numpy as np
import editdistance as ed

def predict_word_from_embedding(embedding,phoc_levels,word_length):
    '''
    Given an embedding(array), phoc_levels(list of levels for phocnet) and the word length
    returns the word mapped from the embedding
    '''
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

def find_string_distances(embeddings,words,phoc_levels,word_length):
    '''
    Given embeddings (2d array of shape mxn where m is the number of words and n is the size of embedding), array of words,list of levels for phocnet and word length, find the words mapped from each row of embeddings, calculates distance between mapped word and actual word and returns this list of distances
    '''
    mapped_words = []
    for i in range(embeddings.shape[0]):
        mapped_word = predict_word_from_embedding(embeddings[i],phoc_levels,word_length)
        mapped_words.append(mapped_word)

    distance_array = [ed.distance(words[i],mapped_words[i]) for i in range(len(words))]

    return distance_array

