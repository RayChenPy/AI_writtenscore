import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
import os
from flask import Flask, redirect, url_for, request, render_template, send_file
#from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet import (ResNet50, preprocess_input)
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as Xception_pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Activation,Flatten, GlobalAveragePooling2D)
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"]="1"

numMap = pd.read_csv('./Flask_t1/models/num_class_map_3.csv')

#-------------------------1 wei
#看圖片的
def displayIMG(windowName,img):
    cv2.namedWindow( windowName, cv2.WINDOW_NORMAL )
    cv2.resizeWindow(windowName, 112, 112)
    cv2.imshow(windowName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#看圖片的特定區域(左上角座標跟 W H值)
def displayIMG_part(windowName,img,A,Y):
    (x, y, w, h) = A
    if Y=="y":
        displayIMG(windowName, img[y:y + h, x:x + w]) 
    return img[y:y + h, x:x + w]
#旋轉
def ImageRotate(image):
    height, width = image.shape[:2]    
    center = (width / 2, height / 2)   
    angle = 20  
    scale = 1                       
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(height, width), borderValue=(255, 255, 255))
    return image_rotation
#去背景十字使用+去除邊框  x是座標,y是閥值
def remove_cross_1(x,y):
    if x<y:
        x=0
    else:
        x=255
    return x
#結合成一個function
def remove_cross(image,y):
    image2=image[::,::,1].copy()
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if i<=5 or j<=5 or i>=image2.shape[0]-5 or j>=image2.shape[1]-5:
                image2[i][j]=255
            else:
                image2[i][j]=remove_cross_1(image2[i][j],y)
    return image2
#算重心與縮放
#算重心
def find_center_1(x):
    if x==0:
        x=1
    else:
        x=0
    return x
#結合成一個找重心function
def find_center(image2):
    image3=image2[::,::,1].copy()
    for i in range(image3.shape[0]):
        for j in range(image3.shape[1]):
            image3[i][j]=find_center_1(image3[i][j])
    target=np.argwhere(image3==1).mean(axis=0)
    return target
#比對重心距離
def compare_center(ans,write):
    ans_target=find_center(ans)
    write_target=find_center(write)
    target=((ans_target[0]-write_target[[0]])**2+(ans_target[1]-write_target[[1]])**2)**0.5
    return target  
#縮放找邊框
#找邊框
def find_boundingbox(image2):
    #化成01矩陣
    image3=image2[::,::,1].copy()
    for i in range(image3.shape[0]):
        for j in range(image3.shape[1]):
            image3[i][j]=find_center_1(image3[i][j])
    
    x,y=[],[]
    for j in range(2):
        target_sum=image3.sum(axis=j)
        z=["",""]
        for i in range(len(target_sum)):
            if target_sum[i] >0 and z[0]=="":
                z[0]=i
            elif target_sum[i] ==0 and z[0]!="" and target_sum[i+1:len(target_sum)].sum()==0:
                z[1]=i
                break
        if j==0:
            x=z.copy()
        elif j==1:
            y=z.copy()
    return x,y
#縮放成跟答案一樣大小的圖片
def image_resize(ans,write):
    ans_x,ans_y=find_boundingbox(ans)
    write_x,write_y=find_boundingbox(write)
    image=write[write_y[0]:write_y[1],write_x[0]:write_x[1],::]
    image=cv2.resize(image,(ans_x[1]-ans_x[0],ans_y[1]-ans_y[0]))
    return image
#比對縮放大小
def compare_size(ans,write):
    ans_x,ans_y=find_boundingbox(ans)
    write_x,write_y=find_boundingbox(write)
    ans_line=((ans_x[0]-ans_x[1])**2+(ans_y[0]-ans_y[1])**2)**0.5
    write_line=((write_x[0]-write_x[1])**2+(write_y[0]-write_y[1])**2)**0.5
    target=write_line/ans_line
    return target
#移動重心並新生圖檔(112,112)，重心都會在56,56
def move_to_center(image):
    img_white = np.ones((112, 112, 3), np.uint8) #生成一个空白圖
    img_white = img_white*255
    xy=find_center(image)
    y=int(round(56-xy[0],0))
    x=int(round(56-xy[1],0)) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_white[i+y][j+x]=image[i][j]
    return img_white
#找格子
def find_lattice(image1):
    image1 = cv2.resize(image1,(1240,1755))
    image2 = image1.copy()
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image,50,150)
    ret, binary = cv2.threshold(canny,127,255,cv2.THRESH_BINARY)  
    #方格資訊存起來
    contours, hierarchy,= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    #創一個矩陣存座標
    arrayA = np.zeros(4, dtype=int).reshape(1,4)
    #叫出每一個方格來篩選
    for (i, c) in enumerate(contours):

        (x, y, w, h) = cv2.boundingRect(c)
        #標準格子篩選:w,h 要大於80小於150，且要是方格
        if min(w,h)>=80 and abs(w-h)<=20 and max(w,h)<=150:
        #把滿足的格子加入矩陣
            arrayB = np.array([(x, y, w, h)])
            arrayA = np.append(arrayA, arrayB, axis=0)
    #把0矩陣刪掉
    arrayA = np.delete(arrayA, 0, axis=0)
    #對x排序
    arrayA=arrayA[np.lexsort(arrayA[:,::-1].T)]
    #透過X座標來矩陣分界整理，用 A(list)放，因為要用X座標差，所以起始一個0
    A=[0]
    for i in range(1,len(arrayA)):
    #X座標差超過90是就另一個格子，避免相近格子重複取
        if arrayA[i][0]-arrayA[i-1][0]>=80:
            #加入A中
            A.append(i)
    A.append(len(arrayA))
    #找答案圖，用B存起來，C存手寫
    B,C=[],[]
    #由前一塊可知A就是個別行的分隔，所以用for 取出
    for i in range(len(A)-1):
        arrayB=arrayA[A[i]:A[i+1]]
        #每個行的群組，最上面的加入B，視為"答案"
        B.append(arrayB[arrayB.argmin(axis=0)[1]])
        #接下來找手寫圖y座標，用小C存起來，大C append([])的目的是順勢造出共幾行的群組，像這樣:A[0]-A[1] = 第一群 = C[0]
        C.append([])
        c=arrayB[:,1].tolist()
        #整理手寫圖座標，存入大C
        #整理方式是排序，並要求Y座標相差大於20才是另一個字，避免重複取
        c.sort()
        for j in range(len(c)-1):
            if c[j+1]-c[j]>=20:
                C[i].append(c[j+1])
    #用小d取出手寫區對應座標的X W H值，先前是用答案的X W H值直接帶入，但有發生切到字的問題
    #原理就是去比對Y座標有沒有一樣，一樣就拉出來
    e=[]
    for k in range(len(B)):
        e.append([])
        for j in range(len(arrayA)):
            if abs(B[k][0]-arrayA[j][0])<30:
                for z in range(len(C[i])):
                    if arrayA[j][1]==C[k][z]:
                        e[k].append(arrayA[j])
    #篩選座標
    d=[]
    for i in range(len(e)):
        d.append([])
        check_list=[]
        for j in range(len(e[i])):
            if e[i][j][1] not in check_list:
                d[i].append(e[i][j])
                check_list.append(e[i][j][1])
    #畫答案方框
    for (x, y, w, h) in B:
        cv2.rectangle(image2, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #畫手寫方框
    for i in range(len(d)):
        for j in range(len(d[i])):
            (x, y, w, h)=(d[i][j][0],  d[i][j][1], d[i][j][2], d[i][j][3])
            cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image2, B, d
#整合成大function
def pretreatment(name):
    #讀取圖片
    image1 = cv2.imread(name)
    #找格子
    image2,ans,write = find_lattice(image1)
    #拉出答案並去十字後存檔
    name_list,number_list=[],[]
    for i in range(len(ans)):
        img=displayIMG_part("Original",image2,ans[i],"n")    #讀取小圖片(img),n改成y可看圖片
        img=remove_cross(img,120)                            #去十字
        name="ans"+str(i+1)+".png"                           #檔案名
        cv2.imwrite(name, img)                               #存圖片
        name_list.append([name])                             #加入list
    #拉出手寫並去十字後存檔
    for i in range(len(write)):
        for j in range(len(write[i])):
            img=displayIMG_part("Original",image2,write[i][j],"n")  #n改成y可看圖片
            img=remove_cross(img,120)                               #去十字
            name="write"+str(i+1)+"-"+str(j+1)+".png"               #檔案名
            cv2.imwrite(name, img)                                  #存圖片
            name_list[i].append([name])                             #加入list
    name_list_score=name_list.copy()                                #Ray新增for score
    #手寫圖片優化處理(重心平移到中間，縮放跟答案相同大小)
    for i in range(len(name_list)):
        img_ans=cv2.imread(name_list[i][0])
        number_list.append([name_list[i][0]])
        for j in range(1,len(name_list[i])):
            #0是答案,後面是手寫
            img=cv2.imread(name_list[i][j][0])                #讀手寫圖片
            number_list[i].append([compare_center(img_ans,img),compare_size(img_ans,img)]) #重心距離差異 #手寫是答案的幾倍大
            img=image_resize(img_ans,img)                     #手寫圖片縮放跟答案相同
            img=move_to_center(img)                           #手寫圖片移動到中心
            cv2.imwrite(name_list[i][j][0], img)              #存手寫圖片
        img_ans=move_to_center(img_ans)                       #答案圖片移動到中心
        cv2.imwrite(name_list[i][0], img_ans)                 #存答案圖片
    return name_list,number_list,name_list_score
#評分用數據
#number_list      []                            []                     [] 
#           答案與手寫群(0~看幾組)    0:答案檔名，1之後是對應手寫的數據        0:重心距離差  1:手寫是答案的幾倍
#把圖片弄成矩陣
def get_testdata(name_list):
    x=[]
    for i in range(len(name_list)):
        for j in range(0,len(name_list[i])):
            if j==0:                    
                img = cv2.imread(name_list[i][j])
            else:
                img = cv2.imread(name_list[i][j][0])      
            x.append(img)
    X=np.array(x,dtype="float32")
    return X
#-------------------------------Ray
def displacement(number_list):
    displacement_list=[]
    for i in number_list:
        for j in i[1:]:
            displacement_list.append(float(j[0]))
    mean=np.mean(displacement_list)
    count=0
    displacement_character=[]
    for i,j in enumerate(displacement_list):
        if j >15:
            count+=1
            displacement_character.append(i)
    return mean,count,displacement_character
#Ray新增
def size(number_list):
    size_list=[]
    for i in number_list:
        for j in i[1:]:
            size_list.append(float(j[1]))
    mean=np.mean(size_list)
    count=0
    size_character=[]
    for i,j in enumerate(size_list):
        if j <0.7:
            count+=1
            size_character.append(i)
    return mean,count,size_character
#Ray新增
def count_error(predict):
    count_error=0
    ans=[]
    for i in predict:
        ans.append(i[0])

    for number,i in enumerate(predict):
        for each in i[1:]:
            if each != ans[number]:
                count_error+=1
    return count_error
#Ray新增
def score(y_score,y_pred):
    #儲存score的list(去除標楷體)
    y_pred_score=[]
    for i,each in enumerate(y_score):
        if i % 7!=0:
            y_pred_score.append(float(each))
    #儲存答案的list
    ans=[]
    counter=0
    for i,each in enumerate(y_pred):
        if i % 7==0:
            ans.append(each)
            counter+=1
        elif i % 7 !=0 and i%7<6:
            ans.append(y_pred[counter*7-7])

    #先將predict的list抽掉標楷體
    predict_final=y_pred.copy()
    ypred_final=[]
    for i,each in enumerate(predict_final):
        if i % 7!=0:
            ypred_final.append(each)
    
    #只儲存答對的score的list    
    y_pred_score_final=[]
    error=0
    for i,each in enumerate(y_pred_score):
        if ypred_final[i] == ans[i]:
            y_pred_score_final.append(each)
    for i,each in enumerate(y_pred_score):
        if ypred_final[i] != ans[i]:
            error+=1
            
    return y_pred_score_final,error
#Ray新增
def predict(book):
    #預處理function,回傳兩個list,第一個是檔名list,第二個是算成績list,第三個是只做去十字的單字矩陣
    name_list,number_list,name_list_score=pretreatment(book)   
    #認字model讀入&preprocess
    x_test=get_testdata(name_list)
    x_test_score=x_test.copy()
    x_test = preprocess_input(x_test)
    #認字model建立&讀weight    
    pre_model = ResNet50(weights=None, include_top=False,
                     input_shape=(112, 112, 3))
    x = GlobalAveragePooling2D()(pre_model.output)
    x = Dropout(0.25)(x)
    outputs = Dense(4803, activation='softmax')(x)
    model_resnet = Model(inputs=pre_model.inputs, outputs=outputs)
    model_resnet.load_weights('./Flask_t1/models/basic_model-best-model-character2.h5')
    y_pred = model_resnet.predict(x_test)
    y_pred = y_pred.argmax(-1)
    
    #評分model讀入&preprocess 
    x_test_score = Xception_pre(x_test_score)
    #評分model建立&讀weight 
    pre_model_score = Xception(weights=None, include_top=False,
                     input_shape=(112, 112, 3))
    x_score = GlobalAveragePooling2D()(pre_model_score.output)
    outputs_score = Dense(1, activation='linear')(x_score)
    model_score = Model(inputs=pre_model_score.inputs, outputs=outputs_score)
    model_score.load_weights('./Flask_t1/models/basic_model-best-model-score.h5')
    y_pred_score = model_score.predict(x_test_score)
    
    y_pred_score_final,error=score(y_pred_score,y_pred)
    
#     y_pred_score_final=[]
#     for i,each in enumerate(y_pred_score):
#         if i % 7!=0:
#             y_pred_score_final.append(float(each))
    y_pred_score_mean=np.mean(y_pred_score_final)
    
    dis_mean,dis_count,dis_character=displacement(number_list)
    size_mean,size_count,size_character=size(number_list)
    
    return y_pred,y_pred_score,y_pred_score_final,y_pred_score_mean,dis_count,size_count,error,number_list

def getScore(img_path,number_list):
    #讀取存的檔案
    evaluate = cv2.imread('./Flask_t1/eval/score_1.jpg')
    #讀取評分
    evaluateNew = cv2.resize(evaluate, (20, 20))
    dis_mean,dis_count,dis_character=displacement(number_list) #寫偏
    size_mean,size_count,size_character=size(number_list) #寫太小
    
    image, B, d = find_lattice(cv2.imread(img_path))
    d = np.array(d).reshape(-1,4)
    dis_character_axis = []
    for i in dis_character:
        dis_character_axis.append(d[i])
    size_character_axis=[]
    for i in size_character:
        size_character_axis.append(d[i])

    img_icon = add_icon(img_path,dis_character_axis,size_character_axis)
    img_icon.save('./Flask_t1/result/output.jpg')
    # img_icon = add_icon(img_path,dis_character,size_character)
    # img_icon.save('./Flask_t1/eval/img_icon.jpg')
    #img = cv2.imread('./Flask_t1/eval/img_icon.jpg')  
    #img = cv2.imread(img_path)
    #img[0:evaluateNew.shape[0], 0:evaluateNew.shape[1]] = evaluateNew
    
    # 寫入圖檔
    # cv2.imwrite('./Flask_t1/result/output.jpg', img)

def model_predict(img_path, fileName):
    data = {'id':[img_path],'class':[""]}
    #Ray新增
    predict_final,predict_score,predict_score_final,predict_score_mean,dis_count,size_count,error,number_list=predict(img_path)
    #Ray新增
    index=[1912]*7+[1000]*7+[1423]*7+[1808]*7+[1149]*7+[64]*7
    predict_score_pd=[]
    for i in predict_score:
        predict_score_pd.append(float(i))
    predict_score_pd[0]="NA"
    predict_score_pd[7]="NA"
    predict_score_pd[14]="NA"
    predict_score_pd[21]="NA"
    predict_score_pd[28]="NA"
    predict_score_pd[35]="NA"
    check=pd.DataFrame(np.array(predict_final),index=index)
    check['score']=predict_score_pd
    Allstring=dict()

    SS = predict_score_mean
    SS = SS - error*0.5
    SS = SS - dis_count*0.05    
    SS = SS - size_count*0.1   
    AA = '吃藤條'
    
    if SS >8:
        AA = '甲上'
    elif SS >= 6 and SS <=8:
        AA = '甲'
    elif SS <6 :
        AA = '甲下'
    if AA == '吃藤條':
        SS = '吃鴨蛋'

    Allstring['s1'] = f'原始得分:{SS}'
    Allstring['s2']  = f'你的平均得分為:{AA}'
    #print(f'dis_mean{dis_mean}')
    Allstring['s3']  = f'你有{error}個字寫錯囉'
    Allstring['s4']  = f'你有{dis_count}個字寫偏囉'
    #print(f'dis_character{dis_character}')
    #print(f'size_mean{size_mean}')
    Allstring['s5']  = f'你有{size_count}個字寫太小囉'
    #print(f'size_character{size_character}')

    # y_pred = model.predict(data_test_generator_1)
    # y_pred_final = pd.DataFrame(y_pred.argmax(-1),columns=['label'])
    # num = y_pred_final.iloc[0]['label']

    getScore(img_path,number_list)
    #存output
    # img_icon = add_icon('./Flask_t1/result/output.jpg',[[200,200,200,200]],[[400,400,400,400]])
    # img_icon.save('./Flask_t1/result/output.jpg')

    print(check)
    return Allstring
def add_icon(img_path, list_1, list_2):
    img = Image.open(img_path) #生字簿
    img = img.resize((1240,1755))
    ring = Image.open('./Flask_t1/picture/ring_red.png')
    frame = Image.open('./Flask_t1/picture/frame_blue.png')
    print('list_1:',list_1)
    print('list_2:',list_2)
    for i in list_1:  
        x,y,w,h = i
        location = [x,y,x+h,y+w]
        ring = ring.resize((h,w))
        img.paste(ring,location,ring) #第三個參數把透明部分顯示出來
    for i in list_2:  
        x,y,h,w = i
        location = [x,y,x+h,y+w]
        frame = frame.resize((h,w))
        img.paste(frame,location,frame) #第三個參數把透明部分顯示出來
    return img

@app.route('/')
def index():
    return render_template("index.html")
    #return render_template("uploadPhoto.html")

@app.route('/download', methods=['GET', 'POST'])
def download_site():
    basepath = os.path.dirname(__file__)
    path = os.path.join(basepath,'result','output.jpg') 
    return send_file(path, as_attachment=True)


@app.route('/uploadPhoto')
def uploadPhoto():
    return render_template("uploadPhoto.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        
        file_path = os.path.join(basepath, 'picture', f.filename) 
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, f.filename)#, model
        # pred_class = preds.argmax(axis=-1)           
           
        return preds
    return "辨識失敗"

if __name__ == '__main__':    
    app.run(debug=True, port=5000)
