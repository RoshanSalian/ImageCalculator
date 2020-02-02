import numpy as np
import cv2


''' ----------TRAINING MODEL---------- '''
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
model.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, res, neighbours, dist = model.findNearest(test, k=3)

matches = res == test_labels
correct = np.count_nonzero(matches)
accuracy = correct*(100.0/res.size)
print(accuracy)

''' -----------PREPARING INPUT-------------'''
def x_cord_contour(contour):
    if(cv2.contourArea(contour)>10):
        M = cv2.moments(contour)
        return(int(M['m10']/M['m00']))
        
def makeSquare(not_square):
    BLACK = [0, 0, 0]
    image_shape = not_square.shape
    height = image_shape[0]
    width = image_shape[1]
    if(height == width):
        sqaure = not_square
        return sqaure
    else:
        double_size = cv2.resize(not_square, (2*height, 2*width), interpolation = cv2.INTER_CUBIC)
        height = height*2
        width = width*2
        #cv2.imshow(double_size)
        if(height > width):
            pad =   (height-width)//2
            double_size_square = cv2.copyMakeBorder(image, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = (width - height)//2
            double_size_square = cv2.copyMakeBorder(double_size, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        
    return double_size_square
    
def resize_to_pixel(dimesions, image):
        buffer_pix = 4
        dimensions = dimesions-buffer_pix
        squared = image
        r = float(dimensions) / squared.shape[1]
        dim = (dimensions, int(squared.shape[0] * r))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        img_dim2 = resized.shape
        height_r = img_dim2[0]
        width_r = img_dim2[1]
        BLACK = [0,0,0]
        if (height_r > width_r):
            resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
        if (height_r < width_r):
            resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
        p = 2
        ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
        return ReSizedImg
    

'''------------New Image Classification---------------'''
image = cv2.imread('test1.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(blur, 30, 150)

contours, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=x_cord_contour, reverse = False)
full_name = []
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    
    if w>5 and h>25:
        roi = blur[y:y+h, x:x+w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        sqaured = makeSquare(roi)
        final = resize_to_pixel(20, sqaured)
        final_array = final.reshape((1, 840))
        # final_array = final_array[:, 1:440]
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = model.findNearest(final_array, k=1)
        number = str(int(float(result[0])))
        #important
        full_name.append(number)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(image, number, (x, y+200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        
print(full_name)
