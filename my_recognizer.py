import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def best_score(log_likelihoods):
        return max(log_likelihoods, key = log_likelihoods.get)

    probabilities = []
    guesses = []
    for word_index in range(0, len(test_set.get_all_Xlengths())):
        word_feature, length = test_set.get_item_Xlengths(word_index)
        log_likelihoods = {}

        # Calculate Log Likelihood score for each word
        for word, model in models.items():
            try:
                score = model.score(word_feature, length)
                log_likelihoods[word] = score
            except:
                log_likelihoods[word] = float("-inf")
                continue

        probabilities.append(log_likelihoods)
        guesses.append(best_score(log_likelihoods))

    return probabilities, guesses
