
# coding: utf-8

# # Artificial Intelligence Engineer Nanodegree - Probabilistic Models
# ## Project: Sign Language Recognition System
# - [Introduction](#intro)
# - [Part 1 Feature Selection](#part1_tutorial)
#     - [Tutorial](#part1_tutorial)
#     - [Features Submission](#part1_submission)
#     - [Features Unittest](#part1_test)
# - [Part 2 Train the models](#part2_tutorial)
#     - [Tutorial](#part2_tutorial)
#     - [Model Selection Score Submission](#part2_submission)
#     - [Model Score Unittest](#part2_test)
# - [Part 3 Build a Recognizer](#part3_tutorial)
#     - [Tutorial](#part3_tutorial)
#     - [Recognizer Submission](#part3_submission)
#     - [Recognizer Unittest](#part3_test)
# - [Part 4 (OPTIONAL) Improve the WER with Language Models](#part4_info)

# <a id='intro'></a>
# ## Introduction
# The overall goal of this project is to build a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models.  In particular, this project employs  [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).  In this video, the right-hand x and y locations are plotted as the speaker signs the sentence.
# [![ASLR demo](http://www-i6.informatik.rwth-aachen.de/~dreuw/images/demosample.png)](https://drive.google.com/open?id=0B_5qGuFe-wbhUXRuVnNZVnMtam8)
# 
# The raw data, train, and test sets are pre-defined.  You will derive a variety of feature sets (explored in Part 1), as well as implement three different model selection criterion to determine the optimal number of hidden states for each word model (explored in Part 2). Finally, in Part 3 you will implement the recognizer and compare the effects the different combinations of feature sets and model selection criteria.  
# 
# At the end of each Part, complete the submission cells with implementations, answer all questions, and pass the unit tests.  Then submit the completed notebook for review!

# <a id='part1_tutorial'></a>
# ## PART 1: Data
# 
# ### Features Tutorial
# ##### Load the initial database
# A data handler designed for this database is provided in the student codebase as the `AslDb` class in the `asl_data` module.  This handler creates the initial [pandas](http://pandas.pydata.org/pandas-docs/stable/) dataframe from the corpus of data included in the `data` directory as well as dictionaries suitable for extracting data in a format friendly to the [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) library.  We'll use those to create models in Part 2.
# 
# To start, let's set up the initial database and select an example set of features for the training set.  At the end of Part 1, you will create additional feature sets for experimentation. 

# In[2]:

import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database
print(asl.df.shape)
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame


# In[3]:

asl.df.ix[98,1]  # look at the data available for an individual frame


# The frame represented by video 98, frame 1 is shown here:
# ![Video 98](http://www-i6.informatik.rwth-aachen.de/~dreuw/database/rwth-boston-104/overview/images/orig/098-start.jpg)

# ##### Feature selection for training the model
# The objective of feature selection when training a model is to choose the most relevant variables while keeping the model as simple as possible, thus reducing training time.  We can use the raw features already provided or derive our own and add columns to the pandas dataframe `asl.df` for selection. As an example, in the next cell a feature named `'grnd-ry'` is added. This feature is the difference between the right-hand y value and the nose y value, which serves as the "ground" right y value. 

# In[4]:

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary


# ##### Try it!

# In[5]:

from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# test the code
test_features_tryit(asl)


# In[6]:

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]


# ##### Build the training set
# Now that we have a feature list defined, we can pass that list to the `build_training` method to collect the features for all the words in the training set.  Each word in the training set has multiple examples from various videos.  Below we can see the unique words that have been loaded into the training set:

# In[7]:

training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))
training.get_all_sequences()


# In[8]:

training.get_all_Xlengths()


# The training data in `training` is an object of class `WordsData` defined in the `asl_data` module.  in addition to the `words` list, data can be accessed with the `get_all_sequences`, `get_all_Xlengths`, `get_word_sequences`, and `get_word_Xlengths` methods. We need the `get_word_Xlengths` method to train multiple sequences with the `hmmlearn` library.  In the following example, notice that there are two lists; the first is a concatenation of all the sequences(the X portion) and the second is a list of the sequence lengths(the Lengths portion).

# In[9]:

training.get_word_Xlengths('CHOCOLATE')


# ###### More feature sets
# So far we have a simple feature set that is enough to get started modeling.  However, we might get better results if we manipulate the raw values a bit more, so we will go ahead and set up some other options now for experimentation later.  For example, we could normalize each speaker's range of motion with grouped statistics using [Pandas stats](http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats) functions and [pandas groupby](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html).  Below is an example for finding the means of all speaker subgroups.

# In[10]:

df_means = asl.df.groupby('speaker').mean()
df_means


# To select a mean that matches by speaker, use the pandas [map](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) method:

# In[11]:

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df.head()


# ##### Try it!

# In[12]:

from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()
# test the code
test_std_tryit(df_std)


# <a id='part1_submission'></a>
# ### Features Implementation Submission
# Implement four feature sets and answer the question that follows.
# - normalized Cartesian coordinates
#     - use *mean* and *standard deviation* statistics and the [standard score](https://en.wikipedia.org/wiki/Standard_score) equation to account for speakers with different heights and arm length
#     
# - polar coordinates
#     - calculate polar coordinates with [Cartesian to polar equations](https://en.wikipedia.org/wiki/Polar_coordinate_system#Converting_between_polar_and_Cartesian_coordinates)
#     - use the [np.arctan2](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.arctan2.html) function and *swap the x and y axes* to move the $0$ to $2\pi$ discontinuity to 12 o'clock instead of 3 o'clock;  in other words, the normal break in radians value from $0$ to $2\pi$ occurs directly to the left of the speaker's nose, which may be in the signing area and interfere with results.  By swapping the x and y axes, that discontinuity move to directly above the speaker's head, an area not generally used in signing.
# 
# - delta difference
#     - as described in Thad's lecture, use the difference in values between one frame and the next frames as features
#     - pandas [diff method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html) and [fillna method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html) will be helpful for this one
# 
# - custom features
#     - These are your own design; combine techniques used above or come up with something else entirely. We look forward to seeing what you come up with! 
#     Some ideas to get you started:
#         - normalize using a [feature scaling equation](https://en.wikipedia.org/wiki/Feature_scaling)
#         - normalize the polar coordinates
#         - adding additional deltas
# 

# In[13]:

# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_orig = ['right-x', 'right-y', 'left-x', 'left-y']

for feat_new, feat_orig in zip(features_norm, features_orig):
    mean_feat = asl.df['speaker'].map(df_means[feat_orig])
    std_feat = asl.df['speaker'].map(df_std[feat_orig])
    asl.df[feat_new] = (asl.df[feat_orig] - mean_feat)/std_feat
    
asl.df.head()    


# In[14]:

# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
grnd_rx, grnd_ry = asl.df['grnd-rx'], asl.df['grnd-ry']
grnd_lx, grnd_ly = asl.df['grnd-lx'], asl.df['grnd-ly']

asl.df['polar-rr'] = np.hypot(grnd_rx, grnd_ry)
asl.df['polar-lr'] = np.hypot(grnd_lx, grnd_ly)
asl.df['polar-rtheta'] = np.arctan2(grnd_rx, grnd_ry)
asl.df['polar-ltheta'] = np.arctan2(grnd_lx, grnd_ly)

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']


# In[15]:

# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
features_ref = ['right-x', 'right-y', 'left-x', 'left-y']

for index, feature in enumerate(features_delta):
    asl.df[feature] = asl.df[features_ref[index]].diff().fillna(0)

asl.df.head()


# In[16]:

# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like

features_norm_grnd = ['norm-grnd-rx', 'norm-grnd-ry', 'norm-grnd-lx','norm-grnd-ly']
features_grnd = ['grnd-rx', 'grnd-ry', 'grnd-lx','grnd-ly']

df_std = asl.df.groupby('speaker').std()
df_mean = asl.df.groupby('speaker').mean()

for feature, root_feat in zip(features_norm_grnd, features_grnd):
    asl.df[feature] = (asl.df[root_feat] - asl.df['speaker'].map(df_means[root_feat])) / asl.df['speaker'].map(df_std[root_feat])

# Creating Delta features from Normalized Grounded Features
features_delta_norm_grnd = ['delta-norm-grnd-rx', 'delta-norm-grnd-ry', 'delta-norm-grnd-lx', 'delta-norm-grnd-ly']
asl.df[features_delta_norm_grnd] = asl.df[features_norm_grnd].fillna(0).diff().fillna(0)

# TODO define a list named 'features_custom' for building the training set
features_custom = features_delta_norm_grnd + features_polar


# **Question 1:**  What custom features did you choose for the features_custom set and why?
# 
# **Answer 1:** I choose delta features using normalized ground features along with the polar features. I think these two sets of features are quite different from each other. Polar features could be more aligned to how people may perceive the signs. Whereas Also adding delta features helps to understand relative movement of the hands.

# <a id='part1_test'></a>
# ### Features Unit Testing
# Run the following unit tests as a sanity check on the defined "ground", "norm", "polar", and 'delta"
# feature sets.  The test simply looks for some valid values but is not exhaustive.  However, the project should not be submitted if these tests don't pass.

# In[17]:

import unittest
# import numpy as np

class TestFeatures(unittest.TestCase):

    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))
                         
suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)


# <a id='part2_tutorial'></a>
# ## PART 2: Model Selection
# ### Model Selection Tutorial
# The objective of Model Selection is to tune the number of states for each word HMM prior to testing on unseen data.  In this section you will explore three methods: 
# - Log likelihood using cross-validation folds (CV)
# - Bayesian Information Criterion (BIC)
# - Discriminative Information Criterion (DIC) 

# ##### Train a single word
# Now that we have built a training set with sequence data, we can "train" models for each word.  As a simple starting example, we train a single word using Gaussian hidden Markov models (HMM).   By using the `fit` method during training, the [Baum-Welch Expectation-Maximization](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) (EM) algorithm is invoked iteratively to find the best estimate for the model *for the number of hidden states specified* from a group of sample seequences. For this example, we *assume* the correct number of hidden states is 3, but that is just a guess.  How do we know what the "best" number of states for training is?  We will need to find some model selection technique to choose the best parameter.

# In[18]:

import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))


# The HMM model has been trained and information can be pulled from the model, including means and variances for each feature and hidden state.  The [log likelihood](http://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution) for any individual sample or group of samples can also be calculated with the `score` method.

# In[19]:

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()
    
show_model_stats(demoword, model)


# ##### Try it!
# Experiment by changing the feature set, word, and/or num_hidden_states values in the next cell to see changes in values.  

# In[20]:

my_testword = 'CHOCOLATE'
model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
show_model_stats(my_testword, model)
print("logL = {}".format(logL))


# In[21]:

# experimenting with different feaures sets and states
feature_set = [features_polar, features_delta_norm_grnd, features_custom]
feature_set_name = ['features_polar', 'features_delta_norm_grnd', 'features_custom']
for i, features in enumerate(feature_set):
    for j in range(2,6): 
        model, logL = train_a_word(my_testword, j, features) # Experiment here with different parameters
        print('\n\nStats for model with features', feature_set_name[i], 'and states', j, '\n\n', '#'*100)
        show_model_stats(my_testword, model)
        print("logL = {}".format(logL))


# ##### Visualize the hidden states
# We can plot the means and variances for each state and feature.  Try varying the number of states trained for the HMM model and examine the variances.  Are there some models that are "better" than others?  How can you tell?  We would like to hear what you think in the classroom online.

# In[22]:

get_ipython().magic('matplotlib inline')


# In[23]:

import math
from matplotlib import (cm, pyplot as plt, mlab)

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()
        
visualize(my_testword, model)


# #####  ModelSelector class
# Review the `ModelSelector` class from the codebase found in the `my_model_selectors.py` module.  It is designed to be a strategy pattern for choosing different model selectors.  For the project submission in this section, subclass `SelectorModel` to implement the following model selectors.  In other words, you will write your own classes/functions in the `my_model_selectors.py` module and run them from this notebook:
# 
# - `SelectorCV `:  Log likelihood with CV
# - `SelectorBIC`: BIC 
# - `SelectorDIC`: DIC
# 
# You will train each word in the training set with a range of values for the number of hidden states, and then score these alternatives with the model selector, choosing the "best" according to each strategy. The simple case of training with a constant value for `n_components` can be called using the provided `SelectorConstant` subclass as follow:

# In[24]:

from my_model_selectors import SelectorConstant

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
word = 'VEGETABLE' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))


# ##### Cross-validation folds
# If we simply score the model with the Log Likelihood calculated from the feature sequences it has been trained on, we should expect that more complex models will have higher likelihoods. However, that doesn't tell us which would have a better likelihood score on unseen data.  The model will likely be overfit as complexity is added.  To estimate which topology model is better using only the training data, we can compare scores using cross-validation.  One technique for cross-validation is to break the training set into "folds" and rotate which fold is left out of training.  The "left out" fold scored.  This gives us a proxy method of finding the best model to use on "unseen data". In the following example, a set of word sequences is broken into three folds using the [scikit-learn Kfold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) class object. When you implement `SelectorCV`, you will use this technique.

# In[25]:

from sklearn.model_selection import KFold

training = asl.build_training(features_ground) # Experiment here with different feature sets
word = 'VEGETABLE' # Experiment here with different words
word_sequences = training.get_word_sequences(word)
split_method = KFold()
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds


# **Tip:** In order to run `hmmlearn` training using the X,lengths tuples on the new folds, subsets must be combined based on the indices given for the folds.  A helper utility has been provided in the `asl_utils` module named `combine_sequences` for this purpose.

# ##### Scoring models with other criterion
# Scoring model topologies with **BIC** balances fit and complexity within the training set for each word.  In the BIC equation, a penalty term penalizes complexity to avoid overfitting, so that it is not necessary to also use cross-validation in the selection process.  There are a number of references on the internet for this criterion.  These [slides](http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf) include a formula you may find helpful for your implementation.
# 
# The advantages of scoring model topologies with **DIC** over BIC are presented by Alain Biem in this [reference](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf) (also found [here](https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf)).  DIC scores the discriminant ability of a training set for one word against competing words.  Instead of a penalty term for complexity, it provides a penalty if model liklihoods for non-matching words are too similar to model likelihoods for the correct word in the word set.

# <a id='part2_submission'></a>
# ### Model Selection Implementation Submission
# Implement `SelectorCV`, `SelectorBIC`, and `SelectorDIC` classes in the `my_model_selectors.py` module.  Run the selectors on the following five words. Then answer the questions about your results.
# 
# **Tip:** The `hmmlearn` library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.

# In[26]:

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit


# In[27]:

# autoreload for automatically reloading changes made in my_model_selectors and my_recognizer
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[28]:

# TODO: Implement SelectorCV in my_model_selector.py
from my_model_selectors import SelectorCV

training = asl.build_training(features_polar)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))


# In[29]:

# TODO: Implement SelectorBIC in module my_model_selectors.py
from my_model_selectors import SelectorBIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))


# In[30]:

# TODO: Implement SelectorDIC in module my_model_selectors.py
from my_model_selectors import SelectorDIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))


# **Question 2:**  Compare and contrast the possible advantages and disadvantages of the various model selectors implemented.
# 
# **Answer 2:** 
# 
# DIC is a discriminant selector method meaning that it compares training set for one word against competing words. It doesn't have a penalty term for complexity. However, it provides a penalty if liklihoods for non-matching words are too similar to  likelihoods for the correct word. This allows DIC to make the best performing selection strategy more often. DIC method has much higher pre-processing time as it needs to create all possible combination for words and number of states. As the number of words increase, processing time becomes very high. 
# 
# In the BIC equation, a penalty term penalizes complexity to avoid overfitting. This strategy tries to ensure that 'law of parsimony' is followed and simpler model is selected over similarly performing complex model. BIC tries to select model that maximizes the gain by improving within-class statistics. 
# 
# Cross validation checks the model performance on the data that is never used in the training. Hence, it allows to create models that generalize well to new data by means of random sampling. More are the number of cross-validation set, lesser is the bias in measuring the performance of the model. However, due to low number of samples in the test set (out of fold data for cross validation) variance increases. A good value of cross validation folds is usally 5 or 10 as it is able to balance bias vs variance trade-off. 

# <a id='part2_test'></a>
# ### Model Selector Unit Testing
# Run the following unit tests as a sanity check on the implemented model selectors.  The test simply looks for valid interfaces  but is not exhaustive. However, the project should not be submitted if these tests don't pass.

# In[41]:

from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)


# <a id='part3_tutorial'></a>
# ## PART 3: Recognizer
# The objective of this section is to "put it all together".  Using the four feature sets created and the three model selectors, you will experiment with the models and present your results.  Instead of training only five specific words as in the previous section, train the entire set with a feature set and model selector strategy.  
# ### Recognizer Tutorial
# ##### Train the full training set
# The following example trains the entire set with the example `features_ground` and `SelectorConstant` features and model selector.  Use this pattern for you experimentation and final submission cells.
# 
# 

# In[42]:

from my_model_selectors import SelectorConstant

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorBIC)
print("Number of word models returned = {}".format(len(models)))


# ##### Load the test set
# The `build_test` method in `ASLdb` is similar to the `build_training` method already presented, but there are a few differences:
# - the object is type `SinglesData` 
# - the internal dictionary keys are the index of the test word rather than the word itself
# - the getter methods are `get_all_sequences`, `get_all_Xlengths`, `get_item_sequences` and `get_item_Xlengths`

# In[43]:

test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))


# <a id='part3_submission'></a>
# ### Recognizer Implementation Submission
# For the final project submission, students must implement a recognizer following guidance in the `my_recognizer.py` module.  Experiment with the four feature sets and the three model selection methods (that's 12 possible combinations). You can add and remove cells for experimentation or run the recognizers locally in some other way during your experiments, but retain the results for your discussion.  For submission, you will provide code cells of **only three** interesting combinations for your discussion (see questions below). At least one of these should produce a word error rate of less than 60%, i.e. WER < 0.60 . 
# 
# **Tip:** The hmmlearn library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.

# In[44]:

# TODO implement the recognize method in my_recognizer
from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorCV


# In[46]:

# TODO Choose a feature set and model selector
features = features_custom # change as needed
model_selector = SelectorDIC # change as needed

# Recognize the test set and display the result with the show_errors method
print("----------------------------------")
print("Features: {}".format(features))
print("Model Selector: {}".format(model_selector))
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)


# In[47]:

# TODO Choose a feature set and model selector
features = features_delta_norm_grnd # change as needed
model_selector = SelectorBIC # change as needed

# Recognize the test set and display the result with the show_errors method
print("----------------------------------")
print("Features: {}".format(features))
print("Model Selector: {}".format(model_selector))
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)


# In[48]:

# TODO Choose a feature set and model selector
# TODO Recognize the test set and display the result with the show_errors method
# Experimenting with feature sets and model selectors
features = features_custom + features_norm # change as needed
model_selector = SelectorCV # change as needed

# Recognize the test set and display the result with the show_errors method
print("----------------------------------")
print("Features: {}".format(features))
print("Model Selector: {}".format(model_selector))
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)


# **Question 3:**  Summarize the error results from three combinations of features and model selectors.  What was the "best" combination and why?  What additional information might we use to improve our WER?  For more insight on improving WER, take a look at the introduction to Part 4.
# 
# **Answer 3:** 
# 
# I tested the following four feature sets with all three selection methods(DIC, BIC and CV). 
# 1. Features_polar - polar features
# 2. Features_delta_norn_grnd - delta of normalized ground features
# 3. Features_custom  - combination of Features_delta_norn_grnd and Features_polar
# 4. Features_norm + Features_custom - combination of Features_custom and Features_norm (normalized features)
# 
# 
# ![title](Model_Comparison.jpg)
# 
# 
# **The best combination** : The best feature and method combination was Features_custom with DIC method. WER for this combination was 0.4269 which is much lower than all other combination tried. 
# 
# It could be due to the reason that these are two different types of features that capture the trajectory of the motion very well. Interesting thing is that combination works much better that the individual features. Another interesting fact is that adding more features to the custom feature did not help. In fact, it made the model worse. The reason for the same is that newly added feature set (Features_norm) is already covered to a great extent in features delta norm ground.
# 
# In terms of model selector, DIC seems to be working the best. The reason for the same is that it compares training set for one word against all competing words and then chooses the best option. Due to this comparison it also takes longer than other methods to run.
# 
# **How to improve WER** : Further improvements could be brought using Language Models. Each word has some probability of occurrence adjacent to specific other words. Sign language word recognition could use this probability along with HMM to identify words. 

# <a id='part3_test'></a>
# ### Recognizer Unit Tests
# Run the following unit tests as a sanity check on the defined recognizer.  The test simply looks for some valid values but is not exhaustive. However, the project should not be submitted if these tests don't pass.

# In[50]:

from asl_test_recognizer import TestRecognize
suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
unittest.TextTestRunner().run(suite)


# <a id='part4_info'></a>
# ## PART 4: (OPTIONAL)  Improve the WER with Language Models
# We've squeezed just about as much as we can out of the model and still only get about 50% of the words right! Surely we can do better than that.  Probability to the rescue again in the form of [statistical language models (SLM)](https://en.wikipedia.org/wiki/Language_model).  The basic idea is that each word has some probability of occurrence within the set, and some probability that it is adjacent to specific other words. We can use that additional information to make better choices.
# 
# ##### Additional reading and resources
# - [Introduction to N-grams (Stanford Jurafsky slides)](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)
# - [Speech Recognition Techniques for a Sign Language Recognition System, Philippe Dreuw et al](https://www-i6.informatik.rwth-aachen.de/publications/download/154/Dreuw--2007.pdf) see the improved results of applying LM on *this* data!
# - [SLM data for *this* ASL dataset](ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-boston-104/lm/)
# 
# ##### Optional challenge
# The recognizer you implemented in Part 3 is equivalent to a "0-gram" SLM.  Improve the WER with the SLM data provided with the data set in the link above using "1-gram", "2-gram", and/or "3-gram" statistics. The `probabilities` data you've already calculated will be useful and can be turned into a pandas DataFrame if desired (see next cell).  
# Good luck!  Share your results with the class!

# In[51]:

# create a DataFrame of log likelihoods for the test word items
df_probs = pd.DataFrame(data=probabilities)
df_probs.head()


# In[ ]:



