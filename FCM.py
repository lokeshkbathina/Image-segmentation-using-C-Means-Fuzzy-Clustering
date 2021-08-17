import matplotlib.pyplot as plt
from time import time
from fcmeans import FCM


path = r'C:\Users\Lokesh\PycharmProjects\pythonProject18\L4.jpg'
img = plt.imread(path)/255
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)


size = img.shape
print(size)

new_time = time()
ip = input("Enter the number of clusters that you want to clusterize the image into = ")
fcm = FCM(n_clusters = int(ip))

pic_n = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
print(pic_n.shape)

fcm.fit(pic_n)
fcm_centers = fcm.centers
fcm_labels = fcm.u.argmax(axis=1)

pic2show = fcm_centers[fcm_labels]

cluster_pic = pic2show.reshape(img.shape[0], img.shape[1], img.shape[2])
print(cluster_pic.shape)


figure = plt.figure(figsize=(20, 20))
subplot1 = figure.add_subplot(121)
subplot1.imshow(img)
subplot2 = figure.add_subplot(122)
subplot2.imshow(cluster_pic)
plt.show()
