import numpy as np
from preprocess import process_data, partition_data
print('processing data...')
X, y = process_data(collapse=False, encode=False, normalize=True, predict_missing=True, k_predict=3)
_, y_classes = process_data(collapse=False, encode=True, normalize=True, predict_missing=True, k_predict=3)
 
partitioned_data = partition_data(X, y, partitions=[0.2,0.8]) 
train = partitioned_data[1]
valid = partitioned_data[0]

#model = FFNN((1000, 100, 100), num_iterations=500) 
#model = FFNN((1000, 10000, 1000, 100, 50), num_iterations = 500)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB

models = [
    [KNeighborsClassifier(3), KNeighborsClassifier(5), KNeighborsClassifier(7)],
    [RadiusNeighborsClassifier(4), RadiusNeighborsClassifier(5), RadiusNeighborsClassifier(10)],
    [GaussianNB(), MultinomialNB(), ComplementNB],
    
]

from cross_validation import cross_validation
print('len models:', len(models))
fitted = []
for m in models:
    r = cross_validation(X, y, m)
    print(r)
    model = m[np.argmin(r)]
    print('best is', np.argmin(r))
    if np.min(r) == 100000:
        continue
    fitted.append(model.fit(X = train[0], y = train[1]))


from evaluate import evaluate
for model in fitted:
    print('evaluation of model')
    evaluate(train[1], model.predict(train[0]))
    evaluate(valid[1], model.predict(valid[0]))

