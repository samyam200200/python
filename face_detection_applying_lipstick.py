#!/usr/bin/env python
# coding: utf-8

# ## HOG features

# In[1]:


import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data,exposure
import cv2


# ## Reading images using OpenCV

# In[2]:


image = cv2.imread('leo_cap.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#SIngle channel hog uses cv2.COLOR_BGR2GRAY
# Feature descriptor and the gradients (hog) is shown
fd, hog_image=hog(image, orientations=8, pixels_per_cell=(16,16),
                 cells_per_block=(1,1), visualize=True, multichannel=True)

# plot input and hog image
fig, (ax1,ax2)=plt.subplots(1,2, figsize=(8,4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input Image')

# Rescaling histogram for better display
hog_image_rescaled=exposure.rescale_intensity(hog_image, in_range=(0,10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Rescaled Hog Image')
plt.show()


# In[3]:


len(fd)


# In[4]:


image.shape


# In[5]:


import face_recognition

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


image=cv2.imread('leo_cap.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[7]:


face_locations=face_recognition.face_locations(image)
#Uses sliding window classifier

number_of_faces=len(face_locations)
print('Found {} face(s) in input image' .format(number_of_faces))


# In[8]:


#Adding rectangle to recognizing faces
plt.imshow(image)
ax=plt.gca()

#repeat for all faces found
for face_location in face_locations:
    #Print the location of each face in the image
    top,right,bottom, left=face_location
    x,y,w,h=left, top, right, bottom
    print('A face is located at pixel location Top: {}, left: {}, Bottom: {}, Right:{}' .format(x,y,w,h))
    
    #adding rectangle as a box around the face
    rect=Rectangle((x,y), w-x, h-y, fill=False, color='red')
    ax.add_patch(rect)
    
#plotting the output
plt.show()


# In[9]:


image=cv2.imread('images.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
face_locations=face_recognition.face_locations(image)
#Uses sliding window classifier

number_of_faces=len(face_locations)
print('Found {} face(s) in input image' .format(number_of_faces))
#Adding rectangle to recognizing faces
plt.imshow(image)
ax=plt.gca()

#repeat for all faces found
for face_location in face_locations:
    #Print the location of each face in the image
    top,right,bottom, left=face_location
    x,y,w,h=left, top, right, bottom
    print('A face is located at pixel location Top: {}, left: {}, Bottom: {}, Right:{}' .format(x,y,w,h))
    
    #adding rectangle as a box around the face
    rect=Rectangle((x,y), w-x, h-y, fill=False, color='red')
    ax.add_patch(rect)
    
#plotting the output
plt.show()


# ## Face Recognition
# lOcate and extract the faces (face detection)
# 
# Represent the face as features - encoding and decoding (Autoencoders) - if decoder can construct the embedding / encoding
# 
# Compare with known faces - present in the created database - then create the encoding and use the test image afterwards and compare the encodings
# 
# Compute euclidean distance and apply threshold (distance rule) - 0.6 is the arbitrary limit

# In[10]:


import face_recognition

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


image=cv2.imread('idris.jpg')
face_idris=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image=cv2.imread('test.jpg')
clooney=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image=cv2.imread('leo_cap.jpg')
leonardo=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[12]:


#creating face encodings
#taking the first element of the output list
face_idris_encoding=face_recognition.face_encodings(face_idris)[0]
clooney_encoding=face_recognition.face_encodings(clooney)[0]
leonardo_encoding=face_recognition.face_encodings(leonardo)[0]

# CReating the database as list
known_face_encodings=[
    face_idris_encoding,
    clooney_encoding,
    leonardo_encoding
]


# In[13]:


image=cv2.imread('clo2.jpg')
unknown_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(unknown_image)

#Encoding the unkown image
unknown_face_encodings=face_recognition.face_encodings(unknown_image)


# In[14]:


#Scipy used to compute distance
from scipy.spatial import distance
   
#run for loop for all the encodings found in the unkonw image
for unknown_face_encoding in unknown_face_encodings:
    results=[]
    for known_face_encoding in known_face_encodings:
        d=distance.euclidean(known_face_encoding, unknown_face_encoding)
        results.append(d)
    threshold=0.6
    results=np.array(results)<=threshold
    
    name='Unknown'
    
    if results[0]:
        name='Idris'
    elif results[1]:
        name='George Clooney'
    elif results[2]:
        name='Leonardo DiCaprio'
    
    print(f"found {name} in the photo!")


# ## Identifying facial landmarks
# - 68 point face landmark model
# - eyes nose chin, nose tip
# - used for modifying images
# - used for improving accuracy - aligning the face towards the front
# - returns chin structure, left eye-brow, right-eyebrow, nose-bridge, nose-tip,
# left-eye, right-eye, top-lip, bottom-lip

# In[15]:


# reading leonardo dicaprio
image=cv2.imread('leo_cap.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[19]:


face_landmarks_list=face_recognition.face_landmarks(image)
print(face_landmarks_list)


# In[18]:


import matplotlib.lines as mlines
from matplotlib.patches import Polygon

plt.imshow(image)
ax=plt.gca()

#Run forloop for all the faces in the facial landmarks list
for face_landmarks in face_landmarks_list:
    left_eyebrow_pts=face_landmarks['left_eyebrow']
    pre_x, pre_y=left_eyebrow_pts[0]
    for (x,y) in left_eyebrow_pts[1:]:
        l=mlines.Line2D([pre_x,x], [pre_y,y], color='red')
        ax.add_line(l)
        pre_x,pre_y=x,y
    
    right_eyebrow_pts=face_landmarks['right_eyebrow']
    pre_x, pre_y=right_eyebrow_pts[0]
    for (x,y) in right_eyebrow_pts[1:]:
        l=mlines.Line2D([pre_x,x], [pre_y,y], color='red')
        ax.add_line(l)
        pre_x,pre_y=x,y
    
    p=Polygon(face_landmarks['top_lip'], facecolor='lightsalmon', edgecolor='orangered')
    ax.add_patch(p)
    p=Polygon(face_landmarks['bottom_lip'], facecolor='lightsalmon', edgecolor='orangered')
    ax.add_patch(p)
    
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




