from keras.models import load_model ,Model
import numpy as np
from PIL import Image
import cv2
import os

'''对图片进行测试'''
model=load_model('mnist_mode.h5')

'''将图片转换为所需要的格式'''
imgfile='Data/dmsp/image_dmsp_hist_aver'
for file_name in os.listdir(imgfile):
    img_name=imgfile+'\\'+file_name
    img =cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    img=cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_CONSTANT)
    img = np.array(img)
    (c, b) = np.shape(img)
    Data = []

    for i in range(5, c - 5):
        for j in range(5, b - 5):
            data = img[i - 5:i + 6, j - 5:j + 6]
            data = np.array([data])
            Data.append(data)
    Data = np.array(Data)
    Data = Data.reshape(-1, 11, 11, 1)

    Data = Data / 255.0

    result = model.predict_classes(Data, batch_size=500, verbose=0)
    result = result.reshape(228, 200)
    result=result*255
    file_name=file_name.strip('.bmp')
    '''用来输出结果图'''
    # cv2.imwrite('Data\\uvi_result'+'\\'+file_name+'_AMET.bmp',result)
    '''用来输出每个点的概率值'''
    intermediate_layer_model = Model(input=model.input, output=model.get_layer('logits').output)
    intermediate_output = intermediate_layer_model.predict(Data, batch_size=500, verbose=0)
    logits_img = [x[1] for x in intermediate_output]
    logits_img = (np.array(logits_img)) * 255
    logits_img.astype(int)
    logits_img = logits_img.reshape(228, 200)
    cv2.imwrite('Data/dmsp/image_dmsp_logits'+'/'+file_name+'.bmp',logits_img)






# print(len(result))
# result=result.reshape(218,190)
# result=result*255
# matplotlib.image.imsave('er_zhi_tu_2.png',result,cmap='gray')
# matplotlib.image.imsave('gai_lv_tu_2.png',logits_img,cmap='gray')


# result=np.pad(result,((3,3),(3,3)),mode='constant' )
# (c,b)=np.shape(result)
# final_result=list()
# for i in range(3,c-3):
#     for j in range(3,b-3):
#         data=result[i-3:i+4,j-3:j+4]
#         data=np.array(data)
#         final_result.append(data)
# final_result=np.array(final_result)
# final_result=final_result.reshape(-1,7,7,1)
# result1=model1.predict_classes(final_result,batch_size=500,verbose=0)
# result1=result1.reshape(218,190)
# plt.imshow(result,cmap='gray')
# # plt.imshow(result,cmap='gray')
# plt.show()



