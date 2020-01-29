import numpy as np
import cv2

# Image dataset
data = cv2.imread('digits.png')
gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
'''small = cv2.pyrDown(gray)
cv2.imshow('data', small)'''
print(gray.shape)

# hsplit(data, sections)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x=np.array(cells)

train = x[:, :70].reshape(-1, 400).astype(np.float32)
test = x[:, 70:100].reshape(-1, 400).astype(np.float32)

k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_labels = np.repeat(k, 350)[:, np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]

model = cv2.ml.KNearest_create()
model.train(train, train_labels)
ret, res, neighbours, dist = knn.find_nearest(test, k=3)

matches = res == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*(100.0/res.size)
print(accuracy)

