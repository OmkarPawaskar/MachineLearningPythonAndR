#Natural Language Processing

#Importing dataset
dataset_original = read.delim("Restaurant_Reviews.tsv",quote = '',stringsAsFactors = F)#delim has default parameter for tab"\t" which we need to seperate Review and Mark.
#Quote is empty which means we want to ignore all quotes

#Cleaning Texts
#install.packages('tm')-cleaning tool
#install.packages('SnowballC') - this contains stopwords and used for stemming
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
#as.character(corpus[[1]]) - to check the string in corpus
corpus = tm_map(corpus,content_transformer(tolower))#to convert string to lower case
#as.character(corpus[[841]])
corpus = tm_map(corpus,removeNumbers) # to remove numbers from reviews
corpus = tm_map(corpus,removePunctuation) # to remove Punctuations from reviews
corpus = tm_map(corpus,removeWords,stopwords()) #to remove all irrelevant words
#Stemming - getting root of each word
#Stemming or Lemmatization
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus,stripWhitespace) #removes extra spaces ie leave only 1 space between words

#Creating Bag of Model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999) #filter non frequent word out of sparse matrix

#For NLP , Classification models such as Naive Bayes and Decision tree are used Mostly
#We will use Random Forest Classification
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)