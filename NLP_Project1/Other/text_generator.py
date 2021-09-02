import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_generator(seed_text, next_words, tokenizer, 
                   model, max_sequence_len):
    '''
    Returns a block of text generated using a trained model.
    
    Parameters:
        seed_text (str): several words to start the block of text
        next_words (int): number of generated words desired
        tokenizer: tokenizer of the trained model
        model: trained model for text generation
        max_sequence_len: the biggest length of the sentences fed to the model
    '''
    for w in range(next_words):
        # Tokenize the previous words (i.e. the starting words when the 
        # text generation first begins)
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad sequences
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, 
                                   padding='pre')
        # Generate model predictions based on the padded sequences
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            # Look for the corresponding word of the predicted word index 
            # output by the model above
            if index == predicted:
                output_word = word
                break
        # Add the predicted word
        seed_text += " " + output_word
    return seed_text

def interactive_text(models, tokenizers, max_seq):
    model_num = int(input("Please input the number label of the model that you "+\
                          "wish to use for generating text\n"+\
                          "(1: Model of under 20; 2: Model of between 20 and "+\
                          "30; 3: Model of above 30)"))
    model = models[model_num-1]
    tokenizer = tokenizers[model_num-1]
    max_sequence_len = max_seq[model_num-1]
        
    seed_text = input("Please input some words that you want to start the "+\
                      "block of text with (in lower case):")
    next_len = input("Please input the number of words you wish to generate:")
    next_words = int(next_len)
    
    
    print("--------------------------------------".center(os.get_terminal_size().columns))
    print("Please see below for texts generated".center(os.get_terminal_size().columns))
    print("--------------------------------------".center(os.get_terminal_size().columns))
    # Extract the summary table from the MCMC model
    text = text_generator(seed_text, next_words, tokenizer, 
                             model, max_sequence_len)
    print(text)
    return text